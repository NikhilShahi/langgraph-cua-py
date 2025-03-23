import os
from typing import Any, Dict, Union

from hyperbrowser import AsyncHyperbrowser
from hyperbrowser.models import CreateSessionParams, ScreenConfig
from langchain_core.runnables import RunnableConfig
from hyperbrowser.models import SessionDetail
from .types import get_configuration_with_defaults


DEFAULT_DISPLAY_WIDTH = 1024
DEFAULT_DISPLAY_HEIGHT = 800


def get_hyperbrowser_client(api_key: str) -> AsyncHyperbrowser:
    """
    Gets the Hyperbrowser client, using the API key provided.

    Args:
        api_key: The API key for Hyperbrowser.

    Returns:
        The Hyperbrowser client.
    """
    if not api_key:
        raise ValueError(
            "Hyperbrowser API key not provided. Please provide one in the configurable fields, "
            "or set it as an environment variable (HYPERBROWSER_API_KEY)"
        )
    client = AsyncHyperbrowser(api_key=api_key)
    return client


async def init_or_load(
    inputs: Dict[str, Any], config: RunnableConfig
) -> SessionDetail:
    """
    Initializes or loads a session based on the inputs provided.

    Args:
        inputs: Dictionary containing session_id for the browser session.
        config: The configuration for the runnable.

    Returns:
        The initialized or loaded session.
    """

    session_id = inputs.get("session_id")

    configuration = get_configuration_with_defaults(config)
    hyperbrowser_api_key = configuration.get("hyperbrowser_api_key")

    if not hyperbrowser_api_key:
        raise ValueError(
            "Hyperbrowser API key not provided. Please provide one in the configurable fields, "
            "or set it as an environment variable (HYPERBROWSER_API_KEY)"
        )

    client = get_hyperbrowser_client(hyperbrowser_api_key)

    if session_id:
        return await client.sessions.get(session_id)

    return await client.sessions.create(
        params=CreateSessionParams(
            solve_captchas=True,
            screen=ScreenConfig(
                width=DEFAULT_DISPLAY_WIDTH,
                height=DEFAULT_DISPLAY_HEIGHT
            )
        )
    )

async def init_or_load_without_config(
    inputs: Dict[str, Any]
) -> SessionDetail:
    """
    Initializes or loads a session based on the inputs provided.
    """
    session_id = inputs.get("session_id")

    client = get_hyperbrowser_client(os.getenv("HYPERBROWSER_API_KEY"))

    if session_id:
        return await client.sessions.get(session_id)

    return await client.sessions.create(
        params=CreateSessionParams(
            solve_captchas=True,
            screen=ScreenConfig(
                width=DEFAULT_DISPLAY_WIDTH,
                height=DEFAULT_DISPLAY_HEIGHT
            )
        )
    )


def is_computer_tool_call(tool_outputs: Any) -> bool:
    """
    Checks if the given tool outputs are a computer call.

    Args:
        tool_outputs: The tool outputs to check.

    Returns:
        True if the tool outputs are a computer call, false otherwise.
    """
    if not tool_outputs or not isinstance(tool_outputs, list):
        return False

    return all(output.get("type") == "computer_call" for output in tool_outputs)
