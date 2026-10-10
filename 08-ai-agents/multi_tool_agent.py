"""
Multi-Tool Agent with Tool Registry and Chaining

An agent that registers multiple tools, selects the appropriate tool based
on user queries, and chains tool outputs together.

Demonstrates:
- Tool registry pattern
- Tool selection based on query analysis
- Tool chaining (output of one tool feeds into the next)
- Sandboxed code execution
- Multi-step task orchestration

Requirements:
    pip install openai

Usage:
    export OPENAI_API_KEY="your-key"
    python multi_tool_agent.py
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------


class Tool(ABC):
    """Abstract base class for tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema for parameters."""

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Run the tool and return a string result."""

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


@dataclass
class ToolRegistry:
    """Central registry for all available tools."""

    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_definitions(self) -> list[dict[str, Any]]:
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def execute(self, name: str, **kwargs: Any) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Unknown tool '{name}'"
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return f"Error executing {name}: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Concrete Tools
# ---------------------------------------------------------------------------


class CalculatorTool(Tool):
    """Evaluate mathematical expressions."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluate a math expression. Supports +, -, *, /, **, sqrt, sin, cos."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"],
        }

    def execute(self, expression: str = "") -> str:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round, "min": min, "max": max})
        sanitized = re.sub(r"[^0-9+\-*/().,%s ]" % "".join(allowed.keys()), "", expression)
        try:
            return str(eval(sanitized, {"__builtins__": {}}, allowed))
        except Exception as e:
            return f"Error: {e}"


class TextAnalyzerTool(Tool):
    """Analyze text for statistics like word count, character count, etc."""

    @property
    def name(self) -> str:
        return "text_analyzer"

    @property
    def description(self) -> str:
        return "Analyze text and return statistics: word count, character count, sentence count, average word length."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyze"}
            },
            "required": ["text"],
        }

    def execute(self, text: str = "") -> str:
        if not text.strip():
            return json.dumps({"error": "Empty text"})

        words = text.split()
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        chars = len(text)
        avg_word_len = sum(len(w) for w in words) / len(words) if words else 0

        stats = {
            "word_count": len(words),
            "character_count": chars,
            "sentence_count": len(sentences),
            "average_word_length": round(avg_word_len, 2),
            "unique_words": len(set(w.lower() for w in words)),
        }
        return json.dumps(stats, indent=2)


class CodeExecutorTool(Tool):
    """Sandboxed Python code execution.

    Only allows safe AST operations. Rejects imports, file I/O, network calls.
    """

    SAFE_MODULES = {"math", "json", "datetime", "re", "itertools", "functools"}

    @property
    def name(self) -> str:
        return "code_executor"

    @property
    def description(self) -> str:
        return (
            "Execute Python code in a sandboxed environment. "
            "Only basic operations allowed: math, string operations, list comprehensions. "
            "No imports, file I/O, or network access."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"],
        }

    def _validate_code(self, code: str) -> str | None:
        """Check code for dangerous operations. Returns error message or None."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Syntax error: {e}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] not in self.SAFE_MODULES:
                        return f"Import of '{alias.name}' is not allowed"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] not in self.SAFE_MODULES:
                    return f"Import from '{node.module}' is not allowed"
            elif isinstance(node, (ast.Open, ast.Write)):
                return "File I/O is not allowed"
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in ("exec", "eval", "compile", "__import__"):
                    return f"Call to '{node.func.id}' is not allowed"

        return None

    def execute(self, code: str = "") -> str:
        error = self._validate_code(code)
        if error:
            return f"Security error: {error}"

        safe_globals: dict[str, Any] = {"__builtins__": {}}
        local_vars: dict[str, Any] = {}

        safe_builtins = {
            "len": len, "range": range, "print": print, "int": int,
            "float": float, "str": str, "bool": bool, "list": list,
            "dict": dict, "set": set, "tuple": tuple, "sorted": sorted,
            "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
            "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
            "isinstance": isinstance, "type": type, "hasattr": hasattr,
            "getattr": getattr, "True": True, "False": False, "None": None,
        }
        safe_globals["__builtins__"] = safe_builtins

        import math as _math
        safe_globals["math"] = _math

        try:
            exec(code, safe_globals, local_vars)
        except Exception as e:
            return f"Execution error: {type(e).__name__}: {e}"

        result_parts = []
        for key, value in local_vars.items():
            if not key.startswith("_"):
                result_parts.append(f"{key} = {value!r}")

        if not result_parts:
            output = safe_globals.get("_output", "")
            return str(output) if output else "Code executed successfully (no output variables)"

        return "\n".join(result_parts)


class FileReaderTool(Tool):
    """Read file contents (mock version for demo)."""

    MOCK_FILES: dict[str, str] = {
        "data.csv": "name,age,score\nAlice,30,95\nBob,25,87\nCharlie,35,92\nDiana,28,88",
        "report.txt": "Q3 2024 Report: Revenue increased 15% YoY. Customer satisfaction at 94%.",
        "config.json": '{"debug": false, "max_retries": 3, "timeout": 30}',
    }

    @property
    def name(self) -> str:
        return "file_reader"

    @property
    def description(self) -> str:
        return "Read the contents of a file. Available files: data.csv, report.txt, config.json"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Name of file to read"}
            },
            "required": ["filename"],
        }

    def execute(self, filename: str = "") -> str:
        if filename in self.MOCK_FILES:
            return self.MOCK_FILES[filename]
        return f"Error: File '{filename}' not found. Available: {list(self.MOCK_FILES.keys())}"


class WebSearchTool(Tool):
    """Mock web search tool."""

    MOCK_RESULTS: dict[str, str] = {
        "python": "Python 3.12 released Oct 2023. Key features: improved error messages, performance gains.",
        "weather": "Today: 72F partly cloudy. Tomorrow: 68F rain. Weekend: 75F sunny.",
        "news": "Tech stocks up 3% on AI optimism. New startup raises $50M Series B.",
        "recipe": "Classic pancake recipe: 1 cup flour, 1 egg, 1 cup milk, 2 tbsp sugar, 1 tsp baking powder.",
    }

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for current information on any topic."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        }

    def execute(self, query: str = "") -> str:
        query_lower = query.lower()
        for key, result in self.MOCK_RESULTS.items():
            if key in query_lower:
                return result
        return f"Search results for '{query}': [Mock] No specific results found."


# ---------------------------------------------------------------------------
# Multi-Tool Agent
# ---------------------------------------------------------------------------


class MultiToolAgent:
    """Agent that selects tools and chains their outputs.

    Uses OpenAI function calling to decide which tool to use and when.
    Supports multi-step workflows where tool outputs feed into subsequent calls.
    """

    SYSTEM_PROMPT = (
        "You are a helpful assistant with access to these tools:\n"
        "- calculator: evaluate math expressions\n"
        "- text_analyzer: analyze text statistics\n"
        "- code_executor: run sandboxed Python code\n"
        "- file_reader: read file contents\n"
        "- web_search: search the web\n\n"
        "When given a task, break it into steps and use the right tools. "
        "Chain tool outputs when one tool's result is needed by another. "
        "Always explain what you are doing and provide a clear final answer."
    )

    def __init__(
        self,
        registry: ToolRegistry,
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
        self.registry = registry
        self.model = model
        self.max_steps = max_steps
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"  {message}")

    def run(self, query: str) -> str:
        """Execute a multi-step task.

        Args:
            query: User's task description.

        Returns:
            Final answer after all tool steps are complete.
        """
        self._log(f"\nUser: {query}")
        self._log("=" * 50)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        tool_defs = self.registry.get_definitions()
        execution_log: list[dict[str, str]] = []

        for step in range(1, self.max_steps + 1):
            self._log(f"\nStep {step}:")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tool_defs,
                tool_choice="auto",
                temperature=0,
            )

            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append(msg.model_dump())

                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    self._log(f"  Tool: {name}({args})")

                    result = self.registry.execute(name, **args)
                    self._log(f"  Result: {result[:200]}")

                    execution_log.append({
                        "tool": name,
                        "input": json.dumps(args),
                        "output": result[:500],
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

            elif msg.content:
                self._log(f"  Response: {msg.content[:200]}")
                messages.append({"role": "assistant", "content": msg.content})

                if not msg.tool_calls:
                    return msg.content

        return "Task completed after maximum steps."


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def demo_registry() -> None:
    """Show the tool registry pattern."""
    print("=" * 60)
    print("TOOL REGISTRY DEMO")
    print("=" * 60)

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(TextAnalyzerTool())
    registry.register(CodeExecutorTool())
    registry.register(FileReaderTool())
    registry.register(WebSearchTool())

    print(f"\nRegistered tools: {registry.list_tools()}")

    print(f"\nCalculator: {registry.execute('calculator', expression='2**10')}")
    print(f"Text stats: {registry.execute('text_analyzer', text='Hello world. This is a test.')}")
    print(f"Code exec: {registry.execute('code_executor', code='result = sum(range(1, 101))')}")
    print(f"File read: {registry.execute('file_reader', filename='data.csv')}")
    print(f"Web search: {registry.execute('web_search', query='python news')}")


def demo_sandbox() -> None:
    """Demonstrate sandboxed code execution."""
    print("\n" + "=" * 60)
    print("SANDBOXED CODE EXECUTION")
    print("=" * 60)

    executor = CodeExecutorTool()

    safe_codes = [
        "result = [x**2 for x in range(10)]",
        "total = sum(1.0/i for i in range(1, 101))",
        "fib = [0, 1]\nfor i in range(8):\n    fib.append(fib[-1] + fib[-2])",
    ]

    dangerous_codes = [
        "import os; os.system('ls')",
        "open('file.txt').read()",
        "__import__('subprocess').call(['ls'])",
    ]

    print("\nSafe code execution:")
    for code in safe_codes:
        result = executor.execute(code=code)
        print(f"  Code: {code[:50]}...")
        print(f"  Result: {result}\n")

    print("Blocked dangerous code:")
    for code in dangerous_codes:
        result = executor.execute(code=code)
        print(f"  Code: {code}")
        print(f"  Result: {result}\n")


def demo_chaining() -> None:
    """Demonstrate tool chaining with the agent."""
    print("=" * 60)
    print("TOOL CHAINING DEMO")
    print("=" * 60)

    try:
        from openai import OpenAI
    except ImportError:
        print("Skipping — openai package not installed.")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping — OPENAI_API_KEY not set.")
        return

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(TextAnalyzerTool())
    registry.register(CodeExecutorTool())
    registry.register(FileReaderTool())

    agent = MultiToolAgent(registry=registry, verbose=True)

    tasks = [
        (
            "Read the file 'data.csv', analyze the text content to count words "
            "and characters, then calculate the average score from the data."
        ),
        (
            "Write Python code to calculate the first 20 Fibonacci numbers, "
            "then analyze the resulting text to count how many digits appear."
        ),
    ]

    for task in tasks:
        print("\n" + "=" * 50)
        answer = agent.run(task)
        print(f"\n>>> Final: {answer}")


def main() -> None:
    """Run all multi-tool agent examples."""
    print("Multi-Tool Agent Examples")
    print("=" * 60)

    demo_registry()
    demo_sandbox()
    demo_chaining()

    print("\n" + "=" * 60)
    print("All multi-tool agent examples completed.")


if __name__ == "__main__":
    main()
