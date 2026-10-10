"""
Function Calling / Tool Use Patterns

Demonstrates function calling across OpenAI, Claude, and Gemini APIs.
Includes parallel tool calls, chaining, and error handling.

Requirements:
    pip install openai anthropic

Usage:
    export OPENAI_API_KEY="your-key"
    export ANTHROPIC_API_KEY="your-key"
    python function_calling.py
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Tool definitions (mock + real)
# ---------------------------------------------------------------------------


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Supports basic arithmetic, exponents, and common math functions.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4").

    Returns:
        String representation of the result, or an error message.
    """
    allowed_names = {
        k: v for k, v in math.__dict__.items() if not k.startswith("_")
    }
    allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})

    sanitized = re.sub(r"[^0-9+\-*/().,%s ]" % "".join(allowed_names.keys()), "", expression)

    if not sanitized.strip():
        return "Error: Empty expression"

    try:
        result = eval(sanitized, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def web_search(query: str) -> str:
    """Mock web search that returns simulated results.

    In production, this would call a search API like Google, Bing, or Tavily.

    Args:
        query: Search query string.

    Returns:
        Mock search results as a formatted string.
    """
    mock_results = {
        "python": "Python is a high-level programming language created by Guido van Rossum in 1991. Latest version: 3.12.",
        "weather": "Today's weather: 72°F, partly cloudy. Tomorrow: 68°F, rain expected.",
        "population": "World population in 2024: approximately 8.1 billion people.",
        "ai": "Artificial Intelligence market projected to reach $1.8 trillion by 2030.",
    }

    query_lower = query.lower()
    for key, result in mock_results.items():
        if key in query_lower:
            return result

    return f"Search results for '{query}': [Mock data] No specific results found."


def database_query(sql: str) -> str:
    """Mock database query executor.

    In production, this would execute against a real database with proper
    parameterization and security controls.

    Args:
        sql: SQL query string.

    Returns:
        Query results as a JSON string or error message.
    """
    mock_tables = {
        "users": [
            {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
            {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25},
            {"id": 3, "name": "Charlie", "email": "charlie@example.com", "age": 35},
        ],
        "orders": [
            {"id": 101, "user_id": 1, "product": "Laptop", "amount": 999.99},
            {"id": 102, "user_id": 2, "product": "Mouse", "amount": 29.99},
            {"id": 103, "user_id": 1, "product": "Keyboard", "amount": 79.99},
        ],
    }

    sql_lower = sql.lower().strip()

    for table_name, data in mock_tables.items():
        if f"select" in sql_lower and table_name in sql_lower:
            if "count(*)" in sql_lower:
                return json.dumps({"result": len(data)})
            return json.dumps({"result": data[:10], "row_count": len(data)})

    if "insert" in sql_lower or "update" in sql_lower or "delete" in sql_lower:
        return json.dumps({"status": "success", "rows_affected": 1})

    return json.dumps({"error": "Unsupported query or table not found"})


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Supports +, -, *, /, **, and math functions like sqrt, sin, cos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate, e.g., '2 + 3 * 4' or 'sqrt(16)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information on a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "database_query",
            "description": "Execute a SQL query against the database. Supports SELECT, INSERT, UPDATE, DELETE.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL query to execute",
                    }
                },
                "required": ["sql"],
            },
        },
    },
]

TOOL_FUNCTIONS: dict[str, Callable[..., str]] = {
    "calculator": calculator,
    "web_search": web_search,
    "database_query": database_query,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Execute a tool by name with the given arguments.

    Args:
        name: Tool function name.
        arguments: Dictionary of arguments to pass to the tool.

    Returns:
        Tool execution result as a string.
    """
    if name not in TOOL_FUNCTIONS:
        return f"Error: Unknown tool '{name}'"

    try:
        result = TOOL_FUNCTIONS[name](**arguments)
        return result
    except TypeError as e:
        return f"Error: Invalid arguments for {name}: {e}"
    except Exception as e:
        return f"Error executing {name}: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# OpenAI Function Calling
# ---------------------------------------------------------------------------


def openai_function_calling() -> None:
    """Demonstrate OpenAI function calling with tool definitions."""
    try:
        from openai import OpenAI
    except ImportError:
        print("[OpenAI] Skipping — openai package not installed.")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[OpenAI] Skipping — OPENAI_API_KEY not set.")
        return

    client = OpenAI(api_key=api_key)

    print("\n" + "=" * 60)
    print("OPENAI FUNCTION CALLING")
    print("=" * 60)

    # --- Example 1: Single tool call ---
    print("\n--- Example 1: Single Tool Call ---")
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": "What is 15 * 37 + 128?"},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOL_DEFINITIONS,
        tool_choice="auto",
    )

    assistant_message = response.choices[0].message
    print(f"Model wants to call tools: {assistant_message.tool_calls is not None}")

    if assistant_message.tool_calls:
        for tool_call in assistant_message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            print(f"  Tool: {func_name}({func_args})")

            result = execute_tool(func_name, func_args)
            print(f"  Result: {result}")

            messages.append(assistant_message.model_dump())
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )
        print(f"Final answer: {final_response.choices[0].message.content}")
    else:
        print(f"Direct response: {assistant_message.content}")

    # --- Example 2: Parallel tool calls ---
    print("\n--- Example 2: Parallel Tool Calls ---")
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": "What is 100 / 4 and what is sqrt(144)?"},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOL_DEFINITIONS,
        tool_choice="auto",
    )

    assistant_message = response.choices[0].message
    if assistant_message.tool_calls:
        print(f"  Model requested {len(assistant_message.tool_calls)} tool calls:")
        messages.append(assistant_message.model_dump())

        for tool_call in assistant_message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            print(f"    - {func_name}({func_args})")

            result = execute_tool(func_name, func_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )
        print(f"  Final answer: {final_response.choices[0].message.content}")

    # --- Example 3: Chaining tool calls ---
    print("\n--- Example 3: Chaining Tool Calls ---")
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": "Search for information about Python, then calculate 2^10."},
    ]

    for iteration in range(5):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )

        assistant_message = response.choices[0].message
        if not assistant_message.tool_calls:
            print(f"  Final answer (after {iteration} tool calls): {assistant_message.content}")
            break

        messages.append(assistant_message.model_dump())
        for tool_call in assistant_message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            result = execute_tool(func_name, func_args)
            print(f"  Step {iteration + 1}: {func_name}({func_args}) → {result[:80]}...")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })


# ---------------------------------------------------------------------------
# Claude (Anthropic) Tool Use
# ---------------------------------------------------------------------------


def claude_tool_use() -> None:
    """Demonstrate Claude tool use with Anthropic API."""
    try:
        import anthropic
    except ImportError:
        print("[Claude] Skipping — anthropic package not installed.")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[Claude] Skipping — ANTHROPIC_API_KEY not set.")
        return

    client = anthropic.Anthropic(api_key=api_key)

    print("\n" + "=" * 60)
    print("CLAUDE (ANTHROPIC) TOOL USE")
    print("=" * 60)

    claude_tools = [
        {
            "name": "calculator",
            "description": "Evaluate a mathematical expression.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
        {
            "name": "web_search",
            "description": "Search the web for information.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    ]

    # --- Example 1: Single tool use ---
    print("\n--- Example 1: Single Tool Use ---")
    messages = [{"role": "user", "content": "What is (23 + 17) * 5?"}]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=claude_tools,
        messages=messages,
    )

    print(f"  Stop reason: {response.stop_reason}")

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            result = execute_tool(block.name, block.input)
            print(f"  Tool call: {block.name}({block.input}) → {result}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })
        elif block.type == "text":
            print(f"  Text: {block.text}")

    if response.stop_reason == "tool_use":
        messages.append({"role": "user", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        final_response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=claude_tools,
            messages=messages,
        )
        for block in final_response.content:
            if block.type == "text":
                print(f"  Final answer: {block.text}")

    # --- Example 2: Parallel tool use ---
    print("\n--- Example 2: Parallel Tool Use ---")
    messages = [{"role": "user", "content": "Calculate sqrt(256) and also search for 'Python programming'"}]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=claude_tools,
        messages=messages,
    )

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            result = execute_tool(block.name, block.input)
            print(f"  Tool: {block.name} → {result[:60]}...")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

    if response.stop_reason == "tool_use":
        messages.append({"role": "user", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        final_response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=claude_tools,
            messages=messages,
        )
        for block in final_response.content:
            if block.type == "text":
                print(f"  Final answer: {block.text}")


# ---------------------------------------------------------------------------
# Error Handling Patterns
# ---------------------------------------------------------------------------


def tool_calling_with_error_handling() -> None:
    """Demonstrate robust error handling for tool execution."""
    try:
        from openai import OpenAI
    except ImportError:
        print("[Error Handling] Skipping — openai package not installed.")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[Error Handling] Skipping — OPENAI_API_KEY not set.")
        return

    client = OpenAI(api_key=api_key)

    print("\n" + "=" * 60)
    print("ERROR HANDLING IN TOOL EXECUTION")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": "Divide 100 by 0 and also search for 'error handling best practices'"},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOL_DEFINITIONS,
    )

    assistant_message = response.choices[0].message
    if not assistant_message.tool_calls:
        print(f"  No tool calls: {assistant_message.content}")
        return

    messages.append(assistant_message.model_dump())

    for tool_call in assistant_message.tool_calls:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)

        print(f"  Calling: {func_name}({func_args})")
        result = execute_tool(func_name, func_args)
        print(f"  Result: {result}")

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })

    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOL_DEFINITIONS,
    )
    print(f"  Final answer: {final_response.choices[0].message.content}")


# ---------------------------------------------------------------------------
# Multi-Tool Agent Demo
# ---------------------------------------------------------------------------


def multi_tool_agent_demo() -> None:
    """End-to-end demo of an agent using multiple tools for a complex task."""
    try:
        from openai import OpenAI
    except ImportError:
        print("[Multi-Tool Agent] Skipping — openai package not installed.")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[Multi-Tool Agent] Skipping — OPENAI_API_KEY not set.")
        return

    client = OpenAI(api_key=api_key)

    print("\n" + "=" * 60)
    print("MULTI-TOOL AGENT DEMO")
    print("=" * 60)

    system_prompt = (
        "You are a research assistant with access to these tools:\n"
        "- calculator: evaluate math expressions\n"
        "- web_search: search the web for information\n"
        "- database_query: query the database\n\n"
        "Use tools as needed to answer the user's question. "
        "Always explain your reasoning and provide a clear final answer."
    )

    user_query = (
        "I need help with a business analysis. First, search for the current "
        "AI market size. Then query our database to get the total number of "
        "users and total order amount. Finally, calculate what percentage of "
        "the AI market we'd need to capture to reach $10 million in revenue, "
        "assuming our average order value is $500."
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    print(f"\nUser: {user_query}\n")
    print("Agent reasoning:")

    for step in range(10):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message

        if assistant_message.content:
            print(f"  [Step {step + 1}] {assistant_message.content}")

        if not assistant_message.tool_calls:
            print(f"\nFinal Answer:\n{assistant_message.content}")
            break

        messages.append(assistant_message.model_dump())

        for tool_call in assistant_message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            result = execute_tool(func_name, func_args)
            print(f"  [Tool] {func_name}({func_args}) → {result[:100]}...")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all function calling examples."""
    print("Function Calling / Tool Use Examples")
    print("=" * 60)

    openai_function_calling()
    claude_tool_use()
    tool_calling_with_error_handling()
    multi_tool_agent_demo()

    print("\n" + "=" * 60)
    print("All examples completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
