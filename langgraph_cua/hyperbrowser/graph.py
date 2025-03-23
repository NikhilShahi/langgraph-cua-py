from langgraph.graph import END, START, StateGraph

from .nodes import call_model, take_browser_action
from .types import CUAState
from .utils import is_computer_tool_call


def take_action_or_end(state: CUAState):
    """
    Routes to the take_computer_action node if a computer call is present
    in the last message, otherwise routes to END.

    Args:
        state: The current state of the thread.

    Returns:
        "take_computer_action" or END depending on if a computer call is present.
    """
    if not state.get("messages", []):
        return END

    last_message = state.get("messages", [])[-1]
    additional_kwargs = getattr(last_message, "additional_kwargs", None)

    if not additional_kwargs:
        return END

    tool_outputs = additional_kwargs.get("tool_outputs")
    tool_calls = getattr(last_message, "tool_calls", [])


    if not is_computer_tool_call(tool_outputs) and len(tool_calls) == 0:
        return END

    return "take_browser_action"


def reinvoke_model_or_end(state: CUAState):
    """
    Routes to the call_model node if the last message is a tool message,
    otherwise routes to END.

    Args:
        state: The current state of the thread.

    Returns:
        "call_model" or END depending on if the last message is a tool message.
    """
    messages = state.get("messages", [])
    if messages and getattr(messages[-1], "type", None) == "tool":
        return "call_model"

    return END


workflow = StateGraph(CUAState)

workflow.add_node("call_model", call_model)
workflow.add_node("take_browser_action", take_browser_action)

workflow.add_edge(START, "call_model")
workflow.add_conditional_edges("call_model", take_action_or_end)
workflow.add_conditional_edges("take_browser_action", reinvoke_model_or_end)

graph = workflow.compile()
graph.name = "Computer Use Agent"


# TODO: What else do I need to do to this to match the other create functions?
def create_cua():
    return graph


__all__ = ["create_cua", graph]
