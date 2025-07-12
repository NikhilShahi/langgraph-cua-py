import time
from typing import Any, Dict, Optional

from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from openai.types.responses.response_computer_tool_call import ResponseComputerToolCall
from scrapybara.types import ComputerResponse, InstanceGetStreamUrlResponse
from hyperbrowser.models import Coordinate

from ..types import CUAState, get_configuration_with_defaults
from ..utils import get_instance, is_computer_tool_call, get_hyperbrowser_client

# Copied from the OpenAI example repository
# https://github.com/openai/openai-cua-sample-app/blob/eb2d58ba77ffd3206d3346d6357093647d29d99c/computers/scrapybara.py#L10
CUA_KEY_TO_SCRAPYBARA_KEY = {
    "/": "slash",
    "\\": "backslash",
    "arrowdown": "Down",
    "arrowleft": "Left",
    "arrowright": "Right",
    "arrowup": "Up",
    "backspace": "BackSpace",
    "capslock": "Caps_Lock",
    "cmd": "Meta_L",
    "delete": "Delete",
    "end": "End",
    "enter": "Return",
    "esc": "Escape",
    "home": "Home",
    "insert": "Insert",
    "option": "Alt_L",
    "pagedown": "Page_Down",
    "pageup": "Page_Up",
    "tab": "Tab",
    "win": "Meta_L",
}


def take_computer_action(state: CUAState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Executes computer actions based on the tool call in the last message.

    Args:
        state: The current state of the CUA agent.
        config: The runnable configuration.

    Returns:
        A dictionary with updated state information.
    """
    message: AnyMessage = state.get("messages", [])[-1]
    assert message.type == "ai", "Last message must be an AI message"
    tool_outputs = message.additional_kwargs.get("tool_outputs")

    if not is_computer_tool_call(tool_outputs):
        # This should never happen, but include the check for proper type safety.
        raise ValueError("Cannot take computer action without a computer call in the last message.")

    # Cast tool_outputs as list[ResponseComputerToolCall] since is_computer_tool_call is true
    tool_outputs: list[ResponseComputerToolCall] = tool_outputs

    instance_id = state.get("instance_id")
    if not instance_id:
        raise ValueError("Instance ID not found in state.")

    configuration = get_configuration_with_defaults(config)
    provider = configuration.get("provider")
    environment = configuration.get("environment")
    auth_state_id = configuration.get("auth_state_id")
    authenticated_id = state.get("authenticated_id")

    if provider == "scrapybara":
        instance = get_instance(instance_id, config)
    elif provider == "hyperbrowser":
        client = get_hyperbrowser_client(configuration.get("hyperbrowser_api_key"))
        instance = client.sessions.get(instance_id)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if (
        environment == "web"
        and auth_state_id is not None
        and (
            (authenticated_id is None)
            or (authenticated_id is not None and authenticated_id != auth_state_id)
        )
    ):
        instance.authenticate(auth_state_id=auth_state_id)
        authenticated_id = auth_state_id

    stream_url: Optional[str] = state.get("stream_url")
    if not stream_url:
        # If the stream_url is not yet defined in state, fetch it, then write to the custom stream
        # so that it's made accessible to the client (or whatever is reading the stream) before any actions are taken.
        if provider == "scrapybara":
            stream_url_response: InstanceGetStreamUrlResponse = instance.get_stream_url()
            stream_url = stream_url_response.stream_url
        elif provider == "hyperbrowser":
            stream_url = instance.live_url
        else:
            raise ValueError(f"Unknown provider: {provider}")

        writer = get_stream_writer()
        writer({"stream_url": stream_url})

    output = tool_outputs[-1]
    action = output.get("action")
    tool_message: Optional[ToolMessage] = None

    try:
        computer_response: Optional[ComputerResponse] = None
        action_type = action.get("type")

        if action_type == "click":
            if provider == "scrapybara":
                computer_response = instance.computer(
                    action="click_mouse",
                    button="middle" if action.get("button") == "wheel" else action.get("button"),
                    coordinates=[action.get("x"), action.get("y")],
                )
            elif provider == "hyperbrowser":
                computer_response = client.computer_action.click(
                    instance,
                    x=action.get("x"),
                    y=action.get("y"),
                    button="middle" if action.get("button") == "wheel" else action.get("button"),
                    return_screenshot=True,
                )
        elif action_type == "double_click":
            if provider == "scrapybara":
                computer_response = instance.computer(
                    action="click_mouse",
                    button="left",
                    coordinates=[action.get("x"), action.get("y")],
                    num_clicks=2,
                )
            elif provider == "hyperbrowser":
                computer_response = client.computer_action.click(
                    instance,
                    x=action.get("x"),
                    y=action.get("y"),
                    button="left",
                    num_clicks=2,
                    return_screenshot=True,
                )
        elif action_type == "drag":
            if provider == "scrapybara":
                computer_response = instance.computer(
                    action="drag_mouse",
                    path=[[point.get("x"), point.get("y")] for point in action.get("path")],
                )
            elif provider == "hyperbrowser":
                computer_response = client.computer_action.drag(
                    instance,
                    path=[
                        Coordinate(x=point.get("x"), y=point.get("y"))
                        for point in action.get("path")
                    ],
                    return_screenshot=True,
                )
        elif action_type == "keypress":
            mapped_keys = [
                CUA_KEY_TO_SCRAPYBARA_KEY.get(key.lower(), key.lower())
                for key in action.get("keys")
            ]
            if provider == "scrapybara":
                computer_response = instance.computer(action="press_key", keys=mapped_keys)
            elif provider == "hyperbrowser":
                computer_response = client.computer_action.press_keys(
                    instance, keys=mapped_keys, return_screenshot=True
                )
        elif action_type == "move":
            if provider == "scrapybara":
                computer_response = instance.computer(
                    action="move_mouse", coordinates=[action.get("x"), action.get("y")]
                )
            elif provider == "hyperbrowser":
                computer_response = client.computer_action.move_mouse(
                    instance, x=action.get("x"), y=action.get("y"), return_screenshot=True
                )
        elif action_type == "screenshot":
            if provider == "scrapybara":
                computer_response = instance.computer(action="take_screenshot")
            elif provider == "hyperbrowser":
                computer_response = client.computer_action.screenshot(instance)
        elif action_type == "wait":
            # Sleep for 2000ms (2 seconds)
            time.sleep(2)
            # Take a screenshot after waiting
            if provider == "scrapybara":
                computer_response = instance.computer(action="take_screenshot")
            elif provider == "hyperbrowser":
                computer_response = client.computer_action.screenshot(instance)
        elif action_type == "scroll":
            if provider == "scrapybara":
                computer_response = instance.computer(
                    action="scroll",
                    delta_x=action.get("scroll_x") // 20,
                    delta_y=action.get("scroll_y") // 20,
                    coordinates=[action.get("x"), action.get("y")],
                )
            elif provider == "hyperbrowser":
                computer_response = client.computer_action.scroll(
                    instance,
                    x=action.get("x"),
                    y=action.get("y"),
                    scroll_x=action.get("scroll_x"),
                    scroll_y=action.get("scroll_y"),
                    return_screenshot=True,
                )
        elif action_type == "type":
            if provider == "scrapybara":
                computer_response = instance.computer(action="type_text", text=action.get("text"))
            elif provider == "hyperbrowser":
                computer_response = client.computer_action.type_text(
                    instance, text=action.get("text"), return_screenshot=True
                )
        else:
            raise ValueError(f"Unknown computer action received: {action}")

        if computer_response:
            output_content = {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{computer_response.base_64_image if provider == 'scrapybara' else computer_response.screenshot}",
            }
            tool_message = {
                "role": "tool",
                "content": [output_content],
                "tool_call_id": output.get("call_id"),
                "additional_kwargs": {"type": "computer_call_output"},
            }
    except Exception as e:
        print(f"\n\nFailed to execute computer call: {e}\n\n")
        print(f"Computer call details: {output}\n\n")

    return {
        "messages": tool_message if tool_message else None,
        "instance_id": instance.id,
        "stream_url": stream_url,
        "authenticated_id": authenticated_id,
    }
