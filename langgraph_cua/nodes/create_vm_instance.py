from langchain_core.runnables.config import RunnableConfig
from scrapybara.client import BrowserInstance, UbuntuInstance, WindowsInstance
from hyperbrowser.models import CreateSessionParams

from ..types import CUAState
from ..utils import get_configuration_with_defaults, get_scrapybara_client, get_hyperbrowser_client

# Copied from the OpenAI example repository
# https://github.com/openai/openai-cua-sample-app/blob/eb2d58ba77ffd3206d3346d6357093647d29d99c/utils.py#L13
BLOCKED_DOMAINS = [
    "maliciousbook.com",
    "evilvideos.com",
    "darkwebforum.com",
    "shadytok.com",
    "suspiciouspins.com",
    "ilanbigio.com",
]


def create_vm_instance(state: CUAState, config: RunnableConfig):
    instance_id = state.get("instance_id")
    configuration = get_configuration_with_defaults(config)
    provider = configuration.get("provider")
    scrapybara_api_key = configuration.get("scrapybara_api_key")
    hyperbrowser_api_key = configuration.get("hyperbrowser_api_key")
    session_params = configuration.get("session_params")
    timeout_hours = configuration.get("timeout_hours")
    environment = configuration.get("environment")

    if instance_id is not None:
        # If the instance_id already exists in state, do nothing.
        return {}

    if provider == "scrapybara":
        if not scrapybara_api_key:
            raise ValueError(
                "Scrapybara API key not provided. Please provide one in the configurable fields, "
                "or set it as an environment variable (SCRAPYBARA_API_KEY)"
            )
    elif provider == "hyperbrowser":
        if not hyperbrowser_api_key:
            raise ValueError(
                "Hyperbrowser API key not provided. Please provide one in the configurable fields, "
                "or set it as an environment variable (HYPERBROWSER_API_KEY)"
            )

    if provider == "scrapybara":
        client = get_scrapybara_client(scrapybara_api_key)
    elif provider == "hyperbrowser":
        client = get_hyperbrowser_client(hyperbrowser_api_key)

    instance: UbuntuInstance | BrowserInstance | WindowsInstance

    if provider == "scrapybara":
        if environment == "ubuntu":
            instance = client.start_ubuntu(timeout_hours=timeout_hours)
        elif environment == "windows":
            instance = client.start_windows(timeout_hours=timeout_hours)
        elif environment == "web":
            blocked_domains = [
                domain.replace("https://", "").replace("www.", "") for domain in BLOCKED_DOMAINS
            ]
            instance = client.start_browser(
                timeout_hours=timeout_hours, blocked_domains=blocked_domains
            )
        else:
            raise ValueError(
                f"Invalid environment. Must be one of 'web', 'ubuntu', or 'windows'. Received: {environment}"
            )
        stream_url = instance.get_stream_url().stream_url
    elif provider == "hyperbrowser":
        instance = client.sessions.create(
            session_params if session_params else CreateSessionParams()
        )
        stream_url = instance.live_url

    return {
        "instance_id": instance.id,
        "stream_url": stream_url,
    }
