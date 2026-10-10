"""
Streaming response demonstration across multiple LLM providers.

Compares streaming vs non-streamed latency, shows real-time token display,
and demonstrates streaming with function calling.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    import httpx
except ImportError:
    httpx = None


@dataclass
class StreamingMetrics:
    """Metrics collected during a streaming response."""
    provider: str
    model: str
    first_token_time: float = 0.0
    total_time: float = 0.0
    token_count: int = 0
    tokens_per_second: float = 0.0
    full_text: str = ""

    def summary(self) -> str:
        return (
            f"Provider: {self.provider} | Model: {self.model}\n"
            f"  Time to first token: {self.first_token_time:.3f}s\n"
            f"  Total time: {self.total_time:.3f}s\n"
            f"  Tokens generated: {self.token_count}\n"
            f"  Tokens/second: {self.tokens_per_second:.1f}"
        )


def stream_openai(
    prompt: str,
    model: str = "gpt-4o-mini",
    system_prompt: str = "",
) -> StreamingMetrics:
    """Stream a response from OpenAI and collect metrics."""
    if OpenAI is None:
        raise ImportError("openai package not installed")

    client = OpenAI()
    metrics = StreamingMetrics(provider="OpenAI", model=model)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    print(f"\n[OpenAI {model}] Streaming:")
    print("-" * 50)

    start_time = time.perf_counter()
    first_token_received = False

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            if not first_token_received:
                metrics.first_token_time = time.perf_counter() - start_time
                first_token_received = True
            print(token, end="", flush=True)
            metrics.token_count += 1
            metrics.full_text += token

    metrics.total_time = time.perf_counter() - start_time
    if metrics.token_count > 0:
        metrics.tokens_per_second = metrics.token_count / metrics.total_time

    print("\n")
    return metrics


def stream_claude(
    prompt: str,
    model: str = "claude-sonnet-4-20250514",
    system_prompt: str = "",
) -> StreamingMetrics:
    """Stream a response from Anthropic Claude and collect metrics."""
    if Anthropic is None:
        raise ImportError("anthropic package not installed")

    client = Anthropic()
    metrics = StreamingMetrics(provider="Claude", model=model)

    print(f"\n[Claude {model}] Streaming:")
    print("-" * 50)

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    start_time = time.perf_counter()
    first_token_received = False

    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            if not first_token_received:
                metrics.first_token_time = time.perf_counter() - start_time
                first_token_received = True
            print(text, end="", flush=True)
            metrics.token_count += 1
            metrics.full_text += text

    metrics.total_time = time.perf_counter() - start_time
    if metrics.token_count > 0:
        metrics.tokens_per_second = metrics.token_count / metrics.total_time

    print("\n")
    return metrics


def stream_ollama(
    prompt: str,
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
) -> StreamingMetrics:
    """Stream a response from a local Ollama instance."""
    if httpx is None:
        raise ImportError("httpx package not installed. Run: pip install httpx")

    metrics = StreamingMetrics(provider="Ollama", model=model)

    print(f"\n[Ollama {model}] Streaming:")
    print("-" * 50)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
    }

    start_time = time.perf_counter()
    first_token_received = False

    with httpx.Client(timeout=60.0) as client:
        with client.stream("POST", f"{base_url}/api/generate", json=payload) as response:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        if not first_token_received:
                            metrics.first_token_time = time.perf_counter() - start_time
                            first_token_received = True
                        print(token, end="", flush=True)
                        metrics.token_count += 1
                        metrics.full_text += token

                    if data.get("done"):
                        break

    metrics.total_time = time.perf_counter() - start_time
    if metrics.token_count > 0:
        metrics.tokens_per_second = metrics.token_count / metrics.total_time

    print("\n")
    return metrics


def non_stream_openai(prompt: str, model: str = "gpt-4o-mini") -> StreamingMetrics:
    """Get a non-streamed response from OpenAI for comparison."""
    if OpenAI is None:
        raise ImportError("openai package not installed")

    client = OpenAI()
    metrics = StreamingMetrics(provider="OpenAI (non-streamed)", model=model)

    print(f"\n[OpenAI {model}] Non-streamed:")
    print("-" * 50)

    start_time = time.perf_counter()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )

    metrics.total_time = time.perf_counter() - start_time
    text = response.choices[0].message.content or ""
    metrics.full_text = text
    metrics.token_count = response.usage.completion_tokens if response.usage else len(text.split())
    metrics.first_token_time = metrics.total_time  # All tokens arrive at once
    if metrics.token_count > 0:
        metrics.tokens_per_second = metrics.token_count / metrics.total_time

    print(f"{text}\n")
    return metrics


def stream_with_function_calling() -> None:
    """Demonstrate streaming with function/tool calls."""
    print("\n--- Streaming with Function Calling ---")
    print("-" * 50)

    if OpenAI is None:
        print("Skipping: openai package not installed")
        return

    client = OpenAI()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=tools,
        stream=True,
        stream_options={"include_usage": True},
    )

    collected_content = ""
    tool_calls = {}

    print("Streaming with tool call:")
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        if delta.content:
            print(delta.content, end="", flush=True)
            collected_content += delta.content

        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls:
                    tool_calls[idx] = {"id": tc.id, "name": "", "arguments": ""}
                if tc.function:
                    if tc.function.name:
                        tool_calls[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls[idx]["arguments"] += tc.function.arguments

    print("\n")
    if tool_calls:
        for idx, tc in tool_calls.items():
            print(f"Tool call {idx}: {tc['name']}({tc['arguments']})")


def compare_latency(prompt: str) -> None:
    """Compare streamed vs non-streamed latency side by side."""
    print("\n" + "=" * 60)
    print("Latency Comparison: Streamed vs Non-Streamed")
    print("=" * 60)
    print(f"Prompt: {prompt[:80]}...\n")

    results: list[StreamingMetrics] = []

    if OpenAI:
        try:
            results.append(non_stream_openai(prompt))
            results.append(stream_openai(prompt))
        except Exception as e:
            print(f"OpenAI error: {e}")

    if Anthropic:
        try:
            # Non-streamed Claude
            client = Anthropic()
            start = time.perf_counter()
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = time.perf_counter() - start
            text = response.content[0].text
            metrics = StreamingMetrics(
                provider="Claude (non-streamed)",
                model="claude-sonnet-4-20250514",
                total_time=elapsed,
                first_token_time=elapsed,
                token_count=response.usage.output_tokens,
            )
            if metrics.token_count > 0:
                metrics.tokens_per_second = metrics.token_count / metrics.total_time
            results.append(metrics)
            print(f"\n[Claude non-streamed] Done in {elapsed:.3f}s\n")

            results.append(stream_claude(prompt))
        except Exception as e:
            print(f"Claude error: {e}")

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    for m in results:
        print(f"\n{m.summary()}")


def run_all_demos() -> None:
    """Run all streaming demonstrations."""
    print("=" * 60)
    print("Streaming Response Demonstration")
    print("=" * 60)

    prompt = "Explain the concept of recursion in programming with a simple example."

    # Basic streaming demos
    if OpenAI and os.environ.get("OPENAI_API_KEY"):
        try:
            stream_openai(prompt)
        except Exception as e:
            print(f"OpenAI streaming error: {e}")

    if Anthropic and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            stream_claude(prompt)
        except Exception as e:
            print(f"Claude streaming error: {e}")

    # Ollama (local) - always try
    try:
        stream_ollama(prompt)
    except Exception as e:
        print(f"Ollama streaming error: {e}")

    # Function calling stream
    if OpenAI and os.environ.get("OPENAI_API_KEY"):
        try:
            stream_with_function_calling()
        except Exception as e:
            print(f"Function calling stream error: {e}")

    # Latency comparison
    if (OpenAI and os.environ.get("OPENAI_API_KEY")) or (
        Anthropic and os.environ.get("ANTHROPIC_API_KEY")
    ):
        compare_latency("Write a short paragraph about the history of computers.")


if __name__ == "__main__":
    run_all_demos()
