"""
Module 07: Meta-Prompting Techniques
Automatic prompt improvement, optimization loops,
meta-system prompts, and constitutional prompting.

Usage:
    python meta_prompting.py
"""

import os
import json
import re
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def call_llm(
    messages: list[dict],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def call_llm_json(
    messages: list[dict],
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
) -> dict:
    raw = call_llm(messages, model=model, temperature=temperature, max_tokens=2048)
    cleaned = re.sub(r"```json\s*|\s*```", "", raw)
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# 1. Automatic Prompt Improvement
# ---------------------------------------------------------------------------

@dataclass
class PromptImprovementResult:
    """Result of prompt improvement."""
    original_prompt: str
    improved_prompt: str
    changes_made: list[str] = field(default_factory=list)


def automatic_prompt_improvement(
    prompt: str,
    task_description: str = "",
    rounds: int = 2,
    model: str = "gpt-4o-mini",
) -> PromptImprovementResult:
    """
    Give the LLM a prompt and ask it to improve itself.

    Args:
        prompt: The prompt to improve.
        task_description: Context about what the prompt is used for.
        rounds: Number of improvement iterations.
        model: LLM model name.

    Returns:
        PromptImprovementResult with improved prompt and changes.
    """
    current_prompt = prompt
    all_changes = []

    for i in range(rounds):
        context = f"Task context: {task_description}\n\n" if task_description else ""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert prompt engineer. Analyze and improve the given prompt. "
                    "Return JSON with:\n"
                    "- 'improved': the improved prompt\n"
                    "- 'changes': array of strings describing what you changed and why"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{context}Original prompt:\n---\n{current_prompt}\n---\n\n"
                    f"Improve this prompt. Make it clearer, more specific, and more effective."
                ),
            },
        ]
        try:
            result = call_llm_json(messages, model=model)
            current_prompt = result.get("improved", current_prompt)
            changes = result.get("changes", [])
            all_changes.extend([f"Round {i+1}: {c}" for c in changes])
        except (json.JSONDecodeError, Exception) as e:
            all_changes.append(f"Round {i+1}: Error during improvement - {e}")

    return PromptImprovementResult(
        original_prompt=prompt,
        improved_prompt=current_prompt,
        changes_made=all_changes,
    )


# ---------------------------------------------------------------------------
# 2. Prompt Optimization Loop
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Result from prompt optimization."""
    prompt: str
    score: float
    history: list[dict] = field(default_factory=list)


def evaluate_prompt(
    prompt: str,
    examples: list[dict],
    model: str = "gpt-4o-mini",
) -> float:
    """
    Evaluate a prompt against test examples.
    Each example has 'input' and 'expected_output'.
    Returns a score from 0.0 to 1.0.
    """
    correct = 0
    for ex in examples:
        messages = [{"role": "user", "content": f"{prompt}\n\nInput: {ex['input']}"}]
        response = call_llm(messages, model=model, temperature=0.0)
        # Simple substring match for evaluation
        if ex["expected_output"].lower() in response.lower():
            correct += 1

    return correct / len(examples) if examples else 0.0


def prompt_optimization_loop(
    task_description: str,
    examples: list[dict],
    initial_prompt: str = "",
    max_iterations: int = 3,
    model: str = "gpt-4o-mini",
) -> OptimizationResult:
    """
    Iteratively optimize a prompt by testing on examples and asking
    the LLM to improve it based on failures.

    Args:
        task_description: What the prompt should accomplish.
        examples: List of {"input": ..., "expected_output": ...} dicts.
        initial_prompt: Starting prompt (generated if empty).
        max_iterations: Number of optimization rounds.
        model: LLM model name.

    Returns:
        OptimizationResult with best prompt and score history.
    """
    # Generate initial prompt if not provided
    if not initial_prompt:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Write a prompt that accomplishes this task:\n{task_description}\n\n"
                    f"Return only the prompt text, nothing else."
                ),
            }
        ]
        current_prompt = call_llm(messages, model=model)
    else:
        current_prompt = initial_prompt

    best_prompt = current_prompt
    best_score = evaluate_prompt(current_prompt, examples, model=model)
    history = [{"prompt": current_prompt, "score": best_score, "iteration": 0}]

    print(f"  Initial prompt score: {best_score:.2%}")

    for i in range(max_iterations):
        # Identify failures
        failures = []
        for ex in examples:
            messages = [{"role": "user", "content": f"{current_prompt}\n\nInput: {ex['input']}"}]
            response = call_llm(messages, model=model, temperature=0.0)
            if ex["expected_output"].lower() not in response.lower():
                failures.append(
                    {"input": ex["input"], "expected": ex["expected_output"], "got": response}
                )

        if not failures:
            print(f"  All examples passed at iteration {i+1}")
            break

        # Ask LLM to improve based on failures
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a prompt optimization expert. Given a prompt that fails on "
                    "some examples, improve it to handle those cases. Return the improved "
                    "prompt only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Current prompt:\n---\n{current_prompt}\n---\n\n"
                    f"Failing examples:\n{json.dumps(failures, indent=2)}\n\n"
                    f"Improve the prompt to handle these cases correctly."
                ),
            },
        ]
        current_prompt = call_llm(messages, model=model)
        score = evaluate_prompt(current_prompt, examples, model=model)
        history.append({"prompt": current_prompt, "score": score, "iteration": i + 1})

        print(f"  Iteration {i+1}: score = {score:.2%}")

        if score > best_score:
            best_score = score
            best_prompt = current_prompt

    return OptimizationResult(prompt=best_prompt, score=best_score, history=history)


# ---------------------------------------------------------------------------
# 3. Meta-System Prompt
# ---------------------------------------------------------------------------

META_SYSTEM_PROMPT = """\
You are a meta-prompt engineer. Your job is to create high-quality
system prompts for other AI assistants.

When given a task description, generate a system prompt that:
1. Clearly defines the assistant's role and expertise
2. Sets specific behavioral guidelines and constraints
3. Provides formatting expectations
4. Includes examples of ideal behavior when helpful
5. Anticipates edge cases and addresses them

The system prompt should be:
- Specific enough to ensure consistent behavior
- Flexible enough to handle varied inputs
- Professional and clear in tone
- Self-contained (no external context needed)

Output the system prompt only, wrapped in <system_prompt> tags.
"""


def generate_system_prompt(
    task_description: str,
    constraints: list[str] | None = None,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Generate a system prompt for an AI assistant given a task description.

    Args:
        task_description: What the assistant should do.
        constraints: List of behavioral constraints or requirements.
        model: LLM model name.

    Returns:
        The generated system prompt.
    """
    constraint_text = ""
    if constraints:
        constraint_text = "Additional constraints:\n" + "\n".join(
            f"- {c}" for c in constraints
        )

    messages = [
        {"role": "system", "content": META_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Task: {task_description}\n\n{constraint_text}",
        },
    ]
    result = call_llm(messages, model=model)
    # Extract from tags if present
    match = re.search(r"<system_prompt>(.*?)</system_prompt>", result, re.DOTALL)
    return match.group(1).strip() if match else result.strip()


# ---------------------------------------------------------------------------
# 4. Constitutional Prompting (Self-Critique and Revise)
# ---------------------------------------------------------------------------

CONSTITUTION = [
    "Is the response helpful and informative?",
    "Is the response accurate and factually grounded?",
    "Is the response clear and well-structured?",
    "Does the response avoid harmful, biased, or misleading content?",
    "Is the response appropriate for the audience and context?",
    "Does the response directly address the user's question?",
]


@dataclass
class ConstitutionalResult:
    """Result from constitutional prompting."""
    final_response: str
    critiques: list[dict] = field(default_factory=list)
    revisions: int = 0


def constitutional_prompting(
    query: str,
    constitution: list[str] | None = None,
    max_revisions: int = 2,
    model: str = "gpt-4o-mini",
) -> ConstitutionalResult:
    """
    Generate a response, self-critique it against a constitution,
    and revise until the response passes all checks.

    Args:
        query: The user's question or task.
        constitution: List of criteria to evaluate against.
        max_revisions: Maximum number of critique-revise cycles.
        model: LLM model name.

    Returns:
        ConstitutionalResult with final response and critique history.
    """
    principles = constitution or CONSTITUTION
    critiques = []

    # Step 1: Generate initial response
    messages = [{"role": "user", "content": query}]
    response = call_llm(messages, model=model)

    for rev in range(max_revisions):
        # Step 2: Critique against each principle
        critique_prompt = [
            {
                "role": "user",
                "content": (
                    f"Original query: {query}\n\n"
                    f"Response to critique:\n{response}\n\n"
                    f"Evaluate the response against each of these principles:\n"
                    + "\n".join(f"{i+1}. {p}" for i, p in enumerate(principles))
                    + "\n\nReturn JSON with:\n"
                    '- "passing": boolean (true if all principles pass)\n'
                    '- "issues": array of objects with "principle" and "feedback"\n'
                ),
            }
        ]

        try:
            critique = call_llm_json(critique_prompt, model=model)
            passing = critique.get("passing", True)
            issues = critique.get("issues", [])
        except (json.JSONDecodeError, Exception):
            passing = True
            issues = []

        critiques.append({"revision": rev + 1, "passing": passing, "issues": issues})

        if passing:
            break

        # Step 3: Revise based on critique
        revision_prompt = [
            {
                "role": "user",
                "content": (
                    f"Original query: {query}\n\n"
                    f"Current response:\n{response}\n\n"
                    f"Critique:\n{json.dumps(issues, indent=2)}\n\n"
                    f"Revise the response to address the issues above. "
                    f"Output ONLY the revised response."
                ),
            }
        ]
        response = call_llm(revision_prompt, model=model)

    return ConstitutionalResult(
        final_response=response,
        critiques=critiques,
        revisions=min(len(critiques), max_revisions),
    )


# ---------------------------------------------------------------------------
# Main: Run all demos
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  META-PROMPTING TECHNIQUES DEMO")
    print("=" * 70)

    # ---- 1. Automatic Prompt Improvement ----
    print("\n\n--- 1. Automatic Prompt Improvement ---")
    original = "Summarize the text."
    result = automatic_prompt_improvement(
        prompt=original,
        task_description="Summarize news articles in 2-3 sentences",
        rounds=2,
    )
    print(f"Original:  {result.original_prompt}")
    print(f"Improved:  {result.improved_prompt}")
    print("Changes:")
    for c in result.changes_made:
        print(f"  - {c}")

    # ---- 2. Prompt Optimization Loop ----
    print("\n\n--- 2. Prompt Optimization Loop ---")
    examples = [
        {"input": "2+2", "expected_output": "4"},
        {"input": "10-3", "expected_output": "7"},
        {"input": "5*4", "expected_output": "20"},
    ]
    opt_result = prompt_optimization_loop(
        task_description="Solve basic arithmetic expressions",
        examples=examples,
        max_iterations=2,
    )
    print(f"Best prompt:\n{opt_result.prompt[:200]}...")
    print(f"Best score: {opt_result.score:.2%}")

    # ---- 3. Meta-System Prompt ----
    print("\n\n--- 3. Meta-System Prompt Generation ---")
    system_prompt = generate_system_prompt(
        task_description=(
            "A customer support agent for a SaaS company that helps users "
            "with billing, technical issues, and account management."
        ),
        constraints=[
            "Always be empathetic and professional",
            "Never share internal system details",
            "Escalate unresolved issues after 3 attempts",
        ],
    )
    print(f"Generated system prompt:\n{system_prompt[:300]}...")

    # ---- 4. Constitutional Prompting ----
    print("\n\n--- 4. Constitutional Prompting ---")
    const_result = constitutional_prompting(
        query="What are the pros and cons of remote work?",
        max_revisions=2,
    )
    print(f"Final response:\n{const_result.final_response[:300]}...")
    print(f"Revisions: {const_result.revisions}")
    print(f"Critique history:")
    for c in const_result.critiques:
        print(f"  Rev {c['revision']}: passing={c['passing']}, issues={len(c['issues'])}")

    print("\n" + "=" * 70)
    print("  All meta-prompting demos complete!")
    print("=" * 70)
