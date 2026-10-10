"""
Anthropic Claude API examples covering messaging, system prompts,
extended thinking, tool use, vision, streaming, and token counting.

Requires: pip install anthropic
Set environment variable: ANTHROPIC_API_KEY=your_key
"""

from __future__ import annotations

import base64
import json
import os
import sys
from typing import Any

try:
    import anthropic
    from anthropic import Anthropic, AnthropicError, RateLimitError
except ImportError:
    print("Error: anthropic package not installed. Run: pip install anthropic")
    sys.exit(1)


def get_client() -> Anthropic:
    """Create and return an Anthropic client with API key validation."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it with: export ANTHROPIC_API_KEY=your_key"
        )
    return Anthropic(api_key=api_key)


def example_basic_messaging(client: Anthropic) -> None:
    """Basic message request to Claude."""
    print("\n--- Basic Messaging ---")
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "What are the three laws of robotics?"}
        ],
    )
    print(f"Response: {response.content[0].text}")
    print(f"Tokens - input: {response.usage.input_tokens}, output: {response.usage.output_tokens}")


def example_system_prompt(client: Anthropic) -> None:
    """Message with a system prompt to set Claude's behavior."""
    print("\n--- System Prompt ---")
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=(
            "You are an expert technical writer. Use clear, concise language. "
            "Format responses with headers and bullet points when appropriate."
        ),
        messages=[
            {
                "role": "user",
                "content": "Explain the difference between TCP and UDP.",
            }
        ],
    )
    print(f"Response:\n{response.content[0].text}")


def example_extended_thinking(client: Anthropic) -> None:
    """Demonstrate extended thinking (thinking blocks) for complex reasoning."""
    print("\n--- Extended Thinking ---")
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000,
        },
        messages=[
            {
                "role": "user",
                "content": (
                    "A farmer has 100 meters of fencing. What is the maximum "
                    "rectangular area he can enclose? Show your reasoning."
                ),
            }
        ],
    )

    for block in response.content:
        if block.type == "thinking":
            print(f"Thinking:\n{block.thinking}\n")
        elif block.type == "text":
            print(f"Answer:\n{block.text}")


def example_tool_use(client: Anthropic) -> None:
    """Demonstrate tool use / function calling with Claude."""
    print("\n--- Tool Use ---")
    tools = [
        {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given ticker symbol",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g., AAPL, GOOGL",
                    }
                },
                "required": ["ticker"],
            },
        }
    ]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": "What is the current price of Apple stock?"}
        ],
    )

    for block in response.content:
        if block.type == "tool_use":
            print(f"Tool call: {block.name}")
            print(f"Input: {json.dumps(block.input, indent=2)}")

            # Simulate tool result
            tool_result = {"price": 178.52, "currency": "USD", "ticker": "AAPL"}
            followup = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                tools=tools,
                messages=[
                    {"role": "user", "content": "What is the current price of Apple stock?"},
                    {"role": "assistant", "content": response.content},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(tool_result),
                            }
                        ],
                    },
                ],
            )
            print(f"Final response: {followup.content[0].text}")


def example_vision(client: Anthropic) -> None:
    """Analyze an image using Claude's vision capabilities."""
    print("\n--- Vision (Image Analysis) ---")
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": image_url,
                        },
                    },
                    {"type": "text", "text": "Describe this image in detail."},
                ],
            }
        ],
    )
    print(f"Description: {response.content[0].text}")


def example_vision_local_image(client: Anthropic) -> None:
    """Analyze a local image file."""
    print("\n--- Vision (Local Image) ---")
    image_path = os.environ.get("SAMPLE_IMAGE_PATH")

    if not image_path or not os.path.exists(image_path):
        print("Skipping: Set SAMPLE_IMAGE_PATH to an image file to test.")
        return

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    suffix = Path(image_path).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_types.get(suffix, "image/jpeg")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": "What do you see in this image?"},
                ],
            }
        ],
    )
    print(f"Description: {response.content[0].text}")


def example_long_context(client: Anthropic) -> None:
    """Handle long context with extended context window."""
    print("\n--- Long Context Handling ---")
    # Simulate a long document
    long_text = (
        "This is a technical document about distributed systems. " * 500
        + "\n\nKey concepts covered: "
        + ", ".join([
            "consensus algorithms",
            "replication strategies",
            "partition tolerance",
            "eventual consistency",
            "Byzantine fault tolerance",
        ])
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"Summarize the key concepts in this document:\n\n{long_text}",
            }
        ],
    )
    print(f"Summary: {response.content[0].text}")
    print(f"Input tokens used: {response.usage.input_tokens}")


def example_streaming(client: Anthropic) -> None:
    """Stream response tokens in real-time."""
    print("\n--- Streaming ---")
    print("Response: ", end="", flush=True)

    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Write a short poem about AI."}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print("\n")


def example_streaming_with_tools(client: Anthropic) -> None:
    """Stream responses that include tool calls."""
    print("\n--- Streaming with Tool Use ---")
    tools = [
        {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"],
            },
        }
    ]

    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=[{"role": "user", "content": "Calculate 15 * 23 + 47"}],
    ) as stream:
        for event in stream:
            if hasattr(event, "type"):
                print(f"Event: {event.type}")


def example_token_counting(client: Anthropic) -> None:
    """Count tokens for a given message."""
    print("\n--- Token Counting ---")
    messages = [
        {
            "role": "user",
            "content": "Explain the theory of relativity in simple terms.",
        }
    ]

    # Use the count_tokens method
    token_count = client.messages.count_tokens(
        model="claude-sonnet-4-20250514",
        messages=messages,
    )
    print(f"Input tokens (counted): {token_count.input_tokens}")

    # Actual API call token usage
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=messages,
    )
    print(f"Input tokens (actual):  {response.usage.input_tokens}")
    print(f"Output tokens (actual): {response.usage.output_tokens}")


def example_multi_turn(client: Anthropic) -> None:
    """Multi-turn conversation with context."""
    print("\n--- Multi-Turn Conversation ---")
    conversation = [
        {"role": "user", "content": "I'm learning Python. What's a list?"},
    ]

    response1 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=conversation,
    )
    print(f"Turn 1 - Claude: {response1.content[0].text[:200]}...")

    conversation.append({"role": "assistant", "content": response1.content[0].text})
    conversation.append({"role": "user", "content": "How is that different from a tuple?"})

    response2 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=conversation,
    )
    print(f"Turn 2 - Claude: {response2.content[0].text[:200]}...")


def example_error_handling(client: Anthropic) -> None:
    """Demonstrate proper error handling."""
    print("\n--- Error Handling ---")
    try:
        # This will fail with invalid model
        client.messages.create(
            model="invalid-model-name",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
        )
    except anthropic.NotFoundError as e:
        print(f"Model not found: {e.message}")
    except anthropic.AuthenticationError as e:
        print(f"Authentication error: {e.message}")
    except RateLimitError:
        print("Rate limited - wait and retry")
    except AnthropicError as e:
        print(f"API error: {e.message}")


def run_all_examples() -> None:
    """Run all Claude API examples."""
    print("=" * 60)
    print("Anthropic Claude API Examples")
    print("=" * 60)

    try:
        client = get_client()
    except EnvironmentError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    examples = [
        example_basic_messaging,
        example_system_prompt,
        example_extended_thinking,
        example_tool_use,
        example_vision,
        example_vision_local_image,
        example_long_context,
        example_streaming,
        example_streaming_with_tools,
        example_token_counting,
        example_multi_turn,
        example_error_handling,
    ]

    for example in examples:
        try:
            example(client)
        except Exception as e:
            print(f"Error in {example.__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    run_all_examples()
