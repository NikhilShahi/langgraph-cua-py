import ast
import json

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from langgraph_cua.hyperbrowser import create_cua

# Load environment variables from .env file
load_dotenv()


@pytest.mark.asyncio
async def test_browser_interaction():
    """
    Test that the agent can interact with the browser.
    """
    graph = create_cua()

    # Create input messages similar to the TypeScript test
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "input_text",
                    "text": "what is current price of nvidia stock?"
                }
            ]
        ),
    ]

    # Stream the graph execution
    stream = graph.astream({"messages": messages}, {"streamMode": ["custom", "updates"]})

    # Process the stream updates
    async for update in stream:
        print("\n---UPDATE---\n")

        if "take_browser_action" in update:
            print("Computer Action:")
            # Check for tool message in the messages field
            tool_message = update.get("take_browser_action", {}).get("messages")
            if tool_message:
                # Extract content from the tool message
                content = tool_message.content

                # Handle the case where content is an array
                if isinstance(content, list) and len(content) > 0:
                    # Use the first item in the array
                    content_item = content[0]
                else:
                    # Use content directly if it's not an array
                    content_item = content

                # Try to parse content if it's a string
                parsed_content = None
                if isinstance(content_item, str):
                    try:
                        # Try parsing as JSON first
                        parsed_content = json.loads(content_item)
                    except json.JSONDecodeError:
                        try:
                            # Try parsing as Python literal (for string representations of dicts)
                            parsed_content = ast.literal_eval(content_item)
                        except (SyntaxError, ValueError):
                            # If both fail, keep content as is
                            parsed_content = None
                else:
                    # If content is already a dict, use it directly
                    parsed_content = content_item if isinstance(content_item, dict) else None

                # Handle image_url specially - truncate to 100 chars
                if (
                    parsed_content
                    and isinstance(parsed_content, dict)
                    and parsed_content.get("image_url")
                ):
                    image_url = parsed_content["image_url"]
                    # Create a copy to avoid modifying the original
                    content_copy = parsed_content.copy()
                    content_copy["image_url"] = (
                        image_url[:100] + "..." if len(image_url) > 100 else image_url
                    )
                    print(f"Tool Message ID: {tool_message.tool_call_id}")
                    # Print the truncated content explicitly
                    print(f"Content type: {content_copy.get('type')}")
                    print(f"Image URL (truncated): {content_copy['image_url']}")
                else:
                    # Just print the first 200 characters of the content if we couldn't parse it
                    if isinstance(content_item, str) and len(content_item) > 200:
                        print(f"Tool Message (truncated content): {content_item[:200]}...")
                    else:
                        print(f"Tool Message: {tool_message}")
        elif "call_model" in update:
            print("Model Call:")
            if update.get("call_model", {}).get("messages"):
                messages = update["call_model"]["messages"]
                if "tool_outputs" in messages.additional_kwargs:
                    print(messages.additional_kwargs["tool_outputs"])
                else:
                    print(messages.content)
        else:
            print(update)
