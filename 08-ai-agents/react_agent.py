"""
ReAct Agent from Scratch

A complete ReAct (Reason + Act) agent implementation without any frameworks.
Uses only the OpenAI API for LLM calls.

The agent follows the Thought -> Action -> Observation loop:
1. Thought: Reason about what to do next
2. Action: Call a tool
3. Observation: Receive the tool result
4. Repeat until Final Answer

Requirements:
    pip install openai

Usage:
    export OPENAI_API_KEY="your-key"
    python react_agent.py
"""

from __future__ import annotations

import json
import math
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Base Tool Class
# ---------------------------------------------------------------------------


class Tool(ABC):
    """Base class for all agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the tool."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema describing the tool's parameters."""

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Execute the tool with the given arguments.

        Returns:
            String result of the tool execution.
        """

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert tool to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


# ---------------------------------------------------------------------------
# Built-in Tools
# ---------------------------------------------------------------------------


class CalculatorTool(Tool):
    """Evaluate mathematical expressions safely."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate a mathematical expression. "
            "Supports +, -, *, /, **, sqrt, sin, cos, tan, log, pi, e."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate",
                }
            },
            "required": ["expression"],
        }

    def execute(self, expression: str = "") -> str:
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("_")
        }
        allowed_names.update({
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
        })

        sanitized = re.sub(
            r"[^0-9+\-*/().,%s ]" % "".join(allowed_names.keys()),
            "",
            expression,
        )

        if not sanitized.strip():
            return "Error: Empty expression"

        try:
            result = eval(sanitized, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"


class WikipediaLookupTool(Tool):
    """Mock Wikipedia lookup with simulated encyclopedia entries."""

    MOCK_DATA: dict[str, str] = {
        "python": (
            "Python is a high-level, general-purpose programming language. "
            "Created by Guido van Rossum and first released in 1991, Python's "
            "design philosophy emphasizes code readability with significant "
            "whitespace. Latest stable version: 3.12 (2023)."
        ),
        "machine learning": (
            "Machine learning is a subset of artificial intelligence that "
            "enables systems to learn from data. Key paradigms include "
            "supervised learning, unsupervised learning, and reinforcement "
            "learning. Popular algorithms: linear regression, decision trees, "
            "neural networks, random forests."
        ),
        "artificial intelligence": (
            "Artificial intelligence (AI) is intelligence demonstrated by "
            "machines, as opposed to natural intelligence displayed by humans. "
            "Key subfields: natural language processing, computer vision, "
            "robotics, expert systems. Major milestones: Deep Blue (1997), "
            "AlphaGo (2016), GPT-4 (2023)."
        ),
        "photosynthesis": (
            "Photosynthesis is the process by which green plants and some "
            "organisms convert light energy into chemical energy. The process "
            "occurs primarily in chloroplasts using chlorophyll. Overall "
            "equation: 6CO2 + 6H2O + light -> C6H12O6 + 6O2."
        ),
        "gravity": (
            "Gravity is a fundamental interaction that causes mutual attraction "
            "between all objects with mass. Newton's law: F = G * m1 * m2 / r^2. "
            "Einstein's general relativity describes gravity as spacetime "
            "curvature. G = 6.674e-11 N(m/kg)^2."
        ),
    }

    @property
    def name(self) -> str:
        return "wikipedia_lookup"

    @property
    def description(self) -> str:
        return "Look up information on a topic from an encyclopedia. Returns factual summaries."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to look up",
                }
            },
            "required": ["topic"],
        }

    def execute(self, topic: str = "") -> str:
        topic_lower = topic.lower().strip()
        for key, summary in self.MOCK_DATA.items():
            if key in topic_lower or topic_lower in key:
                return summary
        return f"No information found for topic: '{topic}'. Try another search term."


class MathEvaluatorTool(Tool):
    """Evaluate math with step-by-step breakdown."""

    @property
    def name(self) -> str:
        return "math_evaluator"

    @property
    def description(self) -> str:
        return (
            "Evaluate a mathematical expression and show the step-by-step "
            "breakdown of the calculation."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate",
                }
            },
            "required": ["expression"],
        }

    def execute(self, expression: str = "") -> str:
        steps = [f"Expression: {expression}"]

        try:
            parts = re.findall(r"[\d.]+|[+\-*/^]", expression)
            steps.append(f"Tokens: {parts}")

            calc_expr = expression.replace("^", "**")
            result = eval(calc_expr, {"__builtins__": {}}, {"__builtins__": {}})
            steps.append(f"Result: {result}")
            return " -> ".join(steps)
        except Exception as e:
            return f"Error evaluating '{expression}': {e}"


# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------


@dataclass
class AgentStep:
    """A single step in the agent's reasoning loop."""

    thought: str
    action: str | None = None
    action_input: dict[str, Any] | None = None
    observation: str | None = None
    is_final: bool = False
    final_answer: str | None = None


class ReActAgent:
    """ReAct agent that reasons and acts in a loop.

    Uses OpenAI function calling for structured tool invocation.
    Falls back to text parsing when function calling is unavailable.
    """

    SYSTEM_PROMPT = (
        "You are a helpful assistant that reasons step-by-step and uses tools "
        "to answer questions. For each question, follow this format:\n\n"
        "Thought: <your reasoning about what to do next>\n"
        "Action: <tool_name>\n"
        "Action Input: <JSON object with the tool's parameters>\n\n"
        "When you have enough information to answer, use:\n"
        "Thought: <your final reasoning>\n"
        "Final Answer: <your complete answer to the user>\n\n"
        "Important:\n"
        "- Always start with a Thought.\n"
        "- Use tools when you need information or calculations.\n"
        "- Only use Final Answer when you are confident in your response.\n"
        "- You may call multiple tools before giving a final answer."
    )

    def __init__(
        self,
        tools: list[Tool] | None = None,
        model: str = "gpt-4o-mini",
        max_steps: int = 10,
        verbose: bool = True,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_steps = max_steps
        self.verbose = verbose

        self.tools: dict[str, Tool] = {}
        if tools:
            for tool in tools:
                self.tools[tool.name] = tool

        self.conversation_history: list[dict[str, Any]] = []
        self.step_history: list[AgentStep] = []

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"  {message}")

    def _build_messages(self, user_query: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_query})
        return messages

    def _get_tool_definitions(self) -> list[dict[str, Any]]:
        return [tool.to_openai_tool() for tool in self.tools.values()]

    def _parse_tool_call_from_text(self, text: str) -> tuple[str, dict[str, Any]] | None:
        action_match = re.search(r"Action:\s*(\w+)", text)
        input_match = re.search(r"Action Input:\s*({.*?})", text, re.DOTALL)

        if not action_match:
            return None

        action = action_match.group(1)
        if action.lower() == "final_answer":
            return None

        action_input: dict[str, Any] = {}
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                raw = input_match.group(1).strip()
                action_input = {"input": raw}

        return action, action_input

    def run(self, query: str) -> str:
        """Run the agent on a user query.

        Args:
            query: The user's question or task.

        Returns:
            The agent's final answer as a string.
        """
        self.step_history.clear()
        self._log(f"\nUser: {query}")
        self._log("=" * 50)

        messages = self._build_messages(query)
        tool_defs = self._get_tool_definitions()

        for step_num in range(1, self.max_steps + 1):
            self._log(f"\nStep {step_num}:")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tool_defs if tool_defs else None,
                tool_choice="auto" if tool_defs else None,
                temperature=0,
            )

            assistant_message = response.choices[0].message

            if assistant_message.tool_calls:
                messages.append(assistant_message.model_dump())

                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)

                    self._log(f"  Action: {func_name}")
                    self._log(f"  Input: {func_args}")

                    if func_name in self.tools:
                        observation = self.tools[func_name].execute(**func_args)
                    else:
                        observation = f"Error: Unknown tool '{func_name}'"

                    self._log(f"  Observation: {observation[:200]}")

                    step = AgentStep(
                        thought="",
                        action=func_name,
                        action_input=func_args,
                        observation=observation,
                    )
                    self.step_history.append(step)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": observation,
                    })

            elif assistant_message.content:
                content = assistant_message.content

                if "Final Answer:" in content:
                    answer = content.split("Final Answer:", 1)[1].strip()
                    self._log(f"\n  Final Answer: {answer}")

                    step = AgentStep(
                        thought=content,
                        is_final=True,
                        final_answer=answer,
                    )
                    self.step_history.append(step)

                    self.conversation_history.append({"role": "user", "content": query})
                    self.conversation_history.append({"role": "assistant", "content": content})

                    return answer

                self._log(f"  Thought: {content[:200]}")
                messages.append({"role": "assistant", "content": content})

                parsed = self._parse_tool_call_from_text(content)
                if parsed:
                    action_name, action_input = parsed
                    if action_name in self.tools:
                        observation = self.tools[action_name].execute(**action_input)
                    else:
                        observation = f"Error: Unknown tool '{action_name}'"

                    self._log(f"  Action: {action_name}")
                    self._log(f"  Observation: {observation[:200]}")

                    messages.append({
                        "role": "user",
                        "content": f"Observation: {observation}",
                    })

                    step = AgentStep(
                        thought=content,
                        action=action_name,
                        action_input=action_input,
                        observation=observation,
                    )
                    self.step_history.append(step)
                else:
                    step = AgentStep(thought=content, is_final=True, final_answer=content)
                    self.step_history.append(step)

                    self.conversation_history.append({"role": "user", "content": query})
                    self.conversation_history.append({"role": "assistant", "content": content})

                    return content
            else:
                self._log("  Warning: Empty response from LLM")
                break

        self._log("\n  Max steps reached without final answer.")
        return "I was unable to determine a final answer within the step limit."


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the ReAct agent demo with example queries."""
    print("ReAct Agent Demo")
    print("=" * 60)

    tools = [
        CalculatorTool(),
        WikipediaLookupTool(),
        MathEvaluatorTool(),
    ]

    agent = ReActAgent(tools=tools, model="gpt-4o-mini", verbose=True)

    queries = [
        "What is the square root of 144 plus the cube root of 27?",
        "Tell me about Python programming language.",
        "If I have 15 items that cost $23.50 each, and I get a 15% discount, "
        "how much do I pay? Use the calculator to compute the total.",
    ]

    for query in queries:
        print("\n" + "=" * 60)
        answer = agent.run(query)
        print(f"\n>>> Final Answer: {answer}")

    print("\n" + "=" * 60)
    print("Agent Statistics:")
    print(f"  Total steps taken: {len(agent.step_history)}")
    print(f"  Tool calls made: {sum(1 for s in agent.step_history if s.action)}")
    print(f"  Conversation turns: {len(agent.conversation_history) // 2}")


if __name__ == "__main__":
    main()
