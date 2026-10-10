#!/usr/bin/env python3
"""Code Review Agent — ReAct pattern with mock tooling."""

import argparse
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MAX_ITERATIONS = 5

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    name: str
    output: str
    success: bool = True


def tool_linter(code: str, language: str) -> ToolResult:
    """Mock linter — checks naming conventions, unused imports, style."""
    issues = []
    lines = code.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # Check for common Python issues
        if language == "python":
            if re.match(r"^def [A-Z]", stripped):
                issues.append(f"L{i}: Function '{stripped.split('(')[0].replace('def ', '')}' should use snake_case")
            if "import *" in stripped:
                issues.append(f"L{i}: Wildcard import detected")
            if stripped.startswith("# noqa"):
                continue
            if len(line) > 120:
                issues.append(f"L{i}: Line exceeds 120 characters ({len(line)} chars)")
            if "\t" in line and "    " in line:
                issues.append(f"L{i}: Mixed tabs and spaces")
        elif language in ("javascript", "typescript"):
            if "==" in stripped and "===" not in stripped:
                issues.append(f"L{i}: Use === instead of ==")
            if "var " in stripped:
                issues.append(f"L{i}: Prefer const/let over var")
            if "console.log" in stripped:
                issues.append(f"L{i}: Remove console.log in production")

    output = "No lint issues found." if not issues else "\n".join(issues)
    return ToolResult(name="linter", output=output, success=True)


def tool_complexity(code: str, language: str) -> ToolResult:
    """Analyze code complexity — function length, nesting depth, branching."""
    lines = code.split("\n")
    report = []
    total_lines = len(lines)
    report.append(f"Total lines: {total_lines}")

    # Find functions
    func_pattern = re.compile(r"^\s*(?:async\s+)?(?:function\s+(\w+)|(?:async\s+)?(?:def\s+(\w+)))")
    functions = []
    for i, line in enumerate(lines, 1):
        m = func_pattern.match(line)
        if m:
            name = m.group(1) or m.group(2)
            # Find end of function (next dedent or end of file)
            indent = len(line) - len(line.lstrip())
            end = total_lines
            for j in range(i, total_lines):
                if lines[j].strip() and (len(lines[j]) - len(lines[j].lstrip())) <= indent and j > i - 1:
                    end = j
                    break
            func_len = end - i + 1
            nesting = max(
                (line.count("    ") for line in lines[i - 1 : end]),
                default=0,
            )
            functions.append({"name": name, "start": i, "length": func_len, "nesting": nesting})

    if functions:
        report.append(f"Functions found: {len(functions)}")
        for f in functions:
            flag = " ⚠️" if f["length"] > 30 or f["nesting"] > 4 else ""
            report.append(
                f"  {f['name']}: L{f['start']}-{f['start']+f['length']-1} "
                f"({f['length']} lines, nesting depth {f['nesting']}){flag}"
            )
    else:
        report.append("No functions found.")

    # Branching
    branch_keywords = {"if", "elif", "else", "switch", "case", "for", "while", "catch"}
    branch_count = sum(
        1 for line in lines
        for kw in branch_keywords
        if re.match(rf"^\s*{kw}\b", line)
    )
    report.append(f"Branch points: {branch_count}")

    # Cyclomatic complexity estimate
    cc = branch_count + 1
    report.append(f"Estimated cyclomatic complexity: {cc} {'⚠️ HIGH' if cc > 10 else '✓ OK'}")

    return ToolResult(name="complexity", output="\n".join(report))


def tool_security(code: str, language: str) -> ToolResult:
    """Mock security scanner — detects common vulnerabilities."""
    findings = []
    lines = code.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # SQL injection
        if re.search(r"""execute\s*\(\s*['\"].*%s|\.format\(|f['\"]""", stripped):
            findings.append(f"L{i}: HIGH — Possible SQL injection (string formatting in query)")
        # Hardcoded secrets
        if re.search(r"(password|secret|api_key|token)\s*=\s*['\"][^'\"]{4,}", stripped, re.IGNORECASE):
            findings.append(f"L{i}: HIGH — Hardcoded secret detected")
        # Command injection
        if re.search(r"os\.system\(|subprocess\.call\(.*shell\s*=\s*True", stripped):
            findings.append(f"L{i}: HIGH — Command injection risk (shell=True or os.system)")
        # Eval/exec
        if re.match(r"^\s*(eval|exec)\s*\(", stripped):
            findings.append(f"L{i}: MEDIUM — Use of {stripped.split('(')[0].strip()}()")
        # Path traversal
        if "open(" in stripped and ".." in stripped:
            findings.append(f"L{i}: MEDIUM — Potential path traversal")
        # Insecure deserialization
        if "pickle.loads(" in stripped:
            findings.append(f"L{i}: HIGH — Insecure deserialization (pickle)")
        if "yaml.load(" in stripped and "Loader=" not in stripped:
            findings.append(f"L{i}: MEDIUM — Unsafe yaml.load()")

    output = "No security issues found." if not findings else "\n".join(findings)
    severity = "HIGH" if any("HIGH" in f for f in findings) else "MEDIUM" if findings else "LOW"
    return ToolResult(name="security", output=output, success=True)

# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------

TOOLS = {
    "linter": tool_linter,
    "complexity": tool_complexity,
    "security": tool_security,
}

TOOL_DESCRIPTIONS = """Available tools:
- linter(code, language): Run lint checks for style and conventions
- complexity(code, language): Analyze code complexity and function metrics
- security(code, language): Scan for security vulnerabilities
"""


def build_system_prompt() -> str:
    return textwrap.dedent(f"""\
        You are an expert code reviewer. You analyze code using a ReAct pattern:
        1. THINK about what aspect to analyze next
        2. ACTION: call a tool by responding with a JSON block
        3. OBSERVE the tool result
        4. Repeat until you have a complete review

        When calling a tool, respond with exactly this format:
        ```json
        {{"tool": "<tool_name>", "args": {{"code": "...", "language": "..."}}}}
        ```

        After all analysis is complete, produce a final structured report in Markdown with:
        - **Summary**: file count, language, overall assessment
        - **Issues**: table with Severity, Line, Description
        - **Suggestions**: actionable improvement list
        - **Positive Notes**: what was done well

        {TOOL_DESCRIPTIONS}
    """)


def call_llm(messages: list[dict]) -> str:
    if not client:
        console.print("[red]No OPENAI_API_KEY set.[/red]")
        sys.exit(1)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
    )
    return resp.choices[0].message.content


def parse_tool_call(text: str) -> dict | None:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try inline JSON
    match = re.search(r'\{"tool":\s*"[^"]+",\s*"args":\s*\{[^}]*\}\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def detect_language(code: str) -> str:
    if re.search(r"\bdef\s+\w+\s*\(", code) or "import " in code:
        return "python"
    if re.search(r"\bfunction\s+\w+|=>\s*\{", code):
        if re.search(r":\s*(string|number|boolean|any)\b", code):
            return "typescript"
        return "javascript"
    return "python"


def extract_code_from_diff(diff_text: str) -> str:
    """Extract actual code from a unified diff."""
    code_lines = []
    for line in diff_text.split("\n"):
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            code_lines.append(line[1:])
        elif line.startswith("-"):
            continue  # skip removed lines for review
        else:
            code_lines.append(line[1:] if line.startswith(" ") else line)
    return "\n".join(code_lines)


def run_agent(code: str, language: str | None = None) -> str:
    lang = language or detect_language(code)
    console.print(f"[dim]Detected language: {lang}[/dim]")

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": f"Review this {lang} code:\n\n```{lang}\n{code}\n```"},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        console.print(f"\n[bold cyan]Iteration {iteration}[/bold cyan]")
        response = call_llm(messages)
        console.print(Panel(response[:300] + ("..." if len(response) > 300 else ""), title="Agent Thought"))
        messages.append({"role": "assistant", "content": response})

        # Check for tool call
        tool_call = parse_tool_call(response)
        if tool_call and tool_call.get("tool") in TOOLS:
            tool_name = tool_call["tool"]
            args = tool_call.get("args", {})
            args.setdefault("code", code)
            args.setdefault("language", lang)
            console.print(f"[yellow]Calling tool: {tool_name}[/yellow]")
            result = TOOLS[tool_name](**args)
            observation = f"Tool '{result.name}' result:\n{result.output}"
            console.print(f"[green]{observation[:200]}[/green]")
            messages.append({"role": "user", "content": observation})
        else:
            # No tool call — agent is done
            console.print("[dim]Agent finished.[/dim]")
            return response

    # If we exhausted iterations, ask for final report
    messages.append({"role": "user", "content": "You've used all tool iterations. Please produce your final Markdown report now."})
    return call_llm(messages)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AI Code Review Agent")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--diff", type=str, help="Path to a unified diff file")
    group.add_argument("--file", type=str, help="Path to a source file")
    group.add_argument("--stdin", action="store_true", help="Read diff from stdin")
    group.add_argument("--code", type=str, help="Inline code string to review")
    parser.add_argument("--language", type=str, default=None, help="Override language detection")
    parser.add_argument("--output", type=str, default=None, help="Save report to file")

    args = parser.parse_args()

    if args.diff:
        code = Path(args.diff).read_text(encoding="utf-8", errors="replace")
        code = extract_code_from_diff(code)
        source = args.diff
    elif args.file:
        code = Path(args.file).read_text(encoding="utf-8", errors="replace")
        source = args.file
    elif args.stdin:
        code = sys.stdin.read()
        code = extract_code_from_diff(code)
        source = "stdin"
    elif args.code:
        code = args.code
        source = "inline"
    else:
        parser.print_help()
        return

    console.print(Panel(f"[bold]Code Review Agent[/bold]\nSource: {source}", border_style="blue"))
    report = run_agent(code, args.language)

    console.print("\n")
    console.print(Panel(Markdown(report), title="Review Report", border_style="green"))

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        console.print(f"\n[green]Report saved to {args.output}[/green]")


if __name__ == "__main__":
    main()
