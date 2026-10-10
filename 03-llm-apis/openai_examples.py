"""
Comprehensive OpenAI API examples covering chat completions, streaming,
structured output, function calling, embeddings, vision, audio, and more.

Requires: pip install openai
Set environment variable: OPENAI_API_KEY=your_key
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
    from openai.types.chat import ChatCompletionToolParam
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)


def get_client() -> OpenAI:
    """Create and return an OpenAI client with API key validation."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it with: export OPENAI_API_KEY=your_key"
        )
    return OpenAI(api_key=api_key)


def example_chat_basic(client: OpenAI) -> None:
    """Basic chat completion request."""
    print("\n--- Basic Chat Completion ---")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ],
    )
    message = response.choices[0].message.content
    print(f"Response: {message}")
    print(f"Tokens used: {response.usage.total_tokens}")


def example_chat_system_prompt(client: OpenAI) -> None:
    """Chat completion with a system prompt to set behavior."""
    print("\n--- Chat with System Prompt ---")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful Python programming assistant. "
                    "Always provide code examples and explain briefly."
                ),
            },
            {"role": "user", "content": "How do I read a file in Python?"},
        ],
    )
    print(f"Response:\n{response.choices[0].message.content}")


def example_multi_turn(client: OpenAI) -> None:
    """Multi-turn conversation maintaining context."""
    print("\n--- Multi-Turn Conversation ---")
    messages = [
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "What is the derivative of x^2?"},
    ]

    response1 = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages
    )
    assistant_msg = response1.choices[0].message
    print(f"Turn 1 - User: {messages[-1]['content']}")
    print(f"Turn 1 - Assistant: {assistant_msg.content}")

    messages.append({"role": "assistant", "content": assistant_msg.content})
    messages.append({"role": "user", "content": "And what about x^3?"})

    response2 = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages
    )
    print(f"Turn 2 - User: {messages[-1]['content']}")
    print(f"Turn 2 - Assistant: {response2.choices[0].message.content}")


def example_streaming(client: OpenAI) -> None:
    """Stream response tokens as they are generated."""
    print("\n--- Streaming Response ---")
    print("Response: ", end="", flush=True)
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Write a haiku about coding."}],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


def example_json_mode(client: OpenAI) -> None:
    """Request structured JSON output from the model."""
    print("\n--- JSON Mode / Structured Output ---")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Respond with valid JSON only.",
            },
            {
                "role": "user",
                "content": (
                    "List 3 programming languages with their name, year created, "
                    "and primary use case. Return as a JSON array."
                ),
            },
        ],
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    print(json.dumps(result, indent=2))


def example_function_calling(client: OpenAI) -> None:
    """Demonstrate function calling / tool use."""
    print("\n--- Function Calling ---")

    tools: list[ChatCompletionToolParam] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name, e.g., 'San Francisco, CA'",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [
        {"role": "user", "content": "What's the weather in Tokyo?"}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, tools=tools, tool_choice="auto"
    )

    assistant_message = response.choices[0].message
    if assistant_message.tool_calls:
        for tool_call in assistant_message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            print(f"Function called: {tool_call.function.name}")
            print(f"Arguments: {json.dumps(args, indent=2)}")

        # Simulate function result
        weather_data = {"temperature": 22, "condition": "Partly cloudy", "unit": "celsius"}
        messages.append(assistant_message.model_dump())
        messages.append({
            "role": "tool",
            "tool_call_id": assistant_message.tool_calls[0].id,
            "content": json.dumps(weather_data),
        })

        final_response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=tools
        )
        print(f"Final response: {final_response.choices[0].message.content}")


def example_embeddings(client: OpenAI) -> None:
    """Generate text embeddings."""
    print("\n--- Embeddings API ---")
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaps above a sleepy hound.",
        "Python is a popular programming language.",
    ]

    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    embeddings = [item.embedding for item in response.data]

    print(f"Generated {len(embeddings)} embeddings")
    print(f"Dimension: {len(embeddings[0])}")

    # Compute cosine similarity between first two (similar) vs first and third (different)
    import math

    def cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    sim_01 = cosine_similarity(embeddings[0], embeddings[1])
    sim_02 = cosine_similarity(embeddings[0], embeddings[2])
    print(f"Similarity (fox sentences): {sim_01:.4f}")
    print(f"Similarity (fox vs python): {sim_02:.4f}")


def example_vision(client: OpenAI) -> None:
    """Analyze an image using the vision-capable model."""
    print("\n--- Vision API (Image Analysis) ---")
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        max_tokens=300,
    )
    print(f"Description: {response.choices[0].message.content}")


def example_audio_transcription(client: OpenAI) -> None:
    """Transcribe audio using the Whisper API.

    Note: Requires an actual audio file. This example creates a placeholder
    and will skip if no file is provided.
    """
    print("\n--- Audio Transcription (Whisper) ---")
    audio_path = os.environ.get("SAMPLE_AUDIO_PATH")

    if not audio_path or not Path(audio_path).exists():
        print("Skipping: Set SAMPLE_AUDIO_PATH to an audio file to test.")
        return

    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )
    print(f"Transcription: {response.text}")


def example_list_models(client: OpenAI) -> None:
    """List available OpenAI models."""
    print("\n--- Model Listing ---")
    models = client.models.list()
    chat_models = [m.id for m in models.data if "gpt" in m.id.lower()]
    print("Available GPT models:")
    for model in sorted(chat_models)[:10]:
        print(f"  - {model}")
    print(f"  ... ({len(chat_models)} total GPT models)")


def example_usage_tracking(client: OpenAI) -> None:
    """Show token usage from a completion request."""
    print("\n--- Usage Tracking ---")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Explain quantum computing in one paragraph."}],
    )
    usage = response.usage
    if usage:
        print(f"Prompt tokens:     {usage.prompt_tokens}")
        print(f"Completion tokens: {usage.completion_tokens}")
        print(f"Total tokens:      {usage.total_tokens}")
        print(f"Model:             {response.model}")


def example_retry_with_backoff() -> None:
    """Demonstrate error handling with exponential backoff retry."""
    print("\n--- Error Handling with Retries ---")
    client = get_client()
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello!"}],
            )
            print(f"Success on attempt {attempt + 1}: {response.choices[0].message.content}")
            return
        except RateLimitError:
            wait_time = 2 ** attempt
            print(f"Rate limited. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
        except APITimeoutError:
            print(f"Request timed out. Retrying... (attempt {attempt + 1}/{max_retries})")
            time.sleep(1)
        except APIConnectionError:
            print(f"Connection error. Retrying... (attempt {attempt + 1}/{max_retries})")
            time.sleep(2)

    print("All retries exhausted.", file=sys.stderr)


def run_all_examples() -> None:
    """Run all OpenAI API examples."""
    print("=" * 60)
    print("OpenAI API Examples")
    print("=" * 60)

    try:
        client = get_client()
    except EnvironmentError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    examples = [
        example_chat_basic,
        example_chat_system_prompt,
        example_multi_turn,
        example_streaming,
        example_json_mode,
        example_function_calling,
        example_embeddings,
        example_vision,
        example_audio_transcription,
        example_list_models,
        example_usage_tracking,
    ]

    for example in examples:
        try:
            example(client)
        except Exception as e:
            print(f"Error in {example.__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    run_all_examples()
