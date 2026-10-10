"""
Module 07: Advanced Prompting Techniques
Runnable implementations of Self-Consistency, Tree-of-Thought,
ReAct, Least-to-Most, and Skeleton-of-Thought.

Usage:
    python techniques.py
"""

import os
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    """Single helper for all LLM calls."""
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
    """Call LLM and parse JSON from response."""
    raw = call_llm(messages, model=model, temperature=temperature, max_tokens=2048)
    # Strip markdown fences if present
    cleaned = re.sub(r"```json\s*|\s*```", "", raw)
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# 1. Self-Consistency
# ---------------------------------------------------------------------------

@dataclass
class ConsistencyResult:
    """Result from self-consistency voting."""
    answer: str
    confidence: float
    votes: dict[str, int]
    raw_responses: list[str] = field(default_factory=list)


def self_consistency(
    problem: str,
    n_samples: int = 5,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> ConsistencyResult:
    """
    Generate N independent Chain-of-Thought completions and take a
    majority vote on the final answer.

    Args:
        problem: The question or task to solve.
        n_samples: Number of independent completions to generate.
        model: LLM model name.
        temperature: Sampling temperature (>0 for diversity).

    Returns:
        ConsistencyResult with the winning answer and confidence.
    """
    prompt = (
        "Solve this step by step. On the final line, write your answer "
        "in the format: ANSWER: <your answer>\n\n"
        f"Problem: {problem}"
    )

    def sample_one():
        return call_llm(
            [{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
        )

    # Generate samples in parallel
    raw_responses = []
    with ThreadPoolExecutor(max_workers=n_samples) as pool:
        futures = [pool.submit(sample_one) for _ in range(n_samples)]
        for f in as_completed(futures):
            raw_responses.append(f.result())

    # Extract answers via regex
    answers = []
    for resp in raw_responses:
        match = re.search(r"ANSWER:\s*(.+)", resp, re.IGNORECASE)
        if match:
            answers.append(match.group(1).strip())

    vote_counts = Counter(answers)
    if not vote_counts:
        return ConsistencyResult(
            answer="No answer found", confidence=0.0, votes={}, raw_responses=raw_responses
        )

    best_answer, count = vote_counts.most_common(1)[0]
    confidence = count / len(answers) if answers else 0.0

    return ConsistencyResult(
        answer=best_answer,
        confidence=confidence,
        votes=dict(vote_counts),
        raw_responses=raw_responses,
    )


# ---------------------------------------------------------------------------
# 2. Tree-of-Thought
# ---------------------------------------------------------------------------

@dataclass
class ToTNode:
    """A node in the Tree-of-Thought."""
    content: str
    score: float = 0.0
    children: list["ToTNode"] = field(default_factory=list)


def tree_of_thought(
    problem: str,
    branches_per_step: int = 3,
    max_depth: int = 3,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Implement a simplified Tree-of-Thought.

    At each depth the model generates candidate continuations, evaluates
    them, and keeps only the best one to proceed.

    Args:
        problem: The task to solve.
        branches_per_step: Number of candidate branches at each node.
        max_depth: How many reasoning steps to take.
        model: LLM model name.

    Returns:
        The final best reasoning chain.
    """
    root = ToTNode(content=problem)
    current_nodes = [root]

    for depth in range(max_depth):
        next_nodes = []
        for node in current_nodes:
            # Generate branches
            gen_prompt = [
                {
                    "role": "user",
                    "content": (
                        f"Problem: {problem}\n\n"
                        f"Current reasoning:\n{node.content}\n\n"
                        f"Generate {branches_per_step} distinct next reasoning steps. "
                        f"Return JSON array of objects with 'text' and 'reasoning_quality' (1-10).\n"
                        f'[{{"text": "...", "reasoning_quality": 7}}, ...]'
                    ),
                }
            ]
            try:
                branches = call_llm_json(gen_prompt, model=model)
                if not isinstance(branches, list):
                    branches = []
            except (json.JSONDecodeError, Exception):
                continue

            # Score branches and keep the best
            for b in branches:
                child = ToTNode(
                    content=b.get("text", ""),
                    score=b.get("reasoning_quality", 5),
                )
                node.children.append(child)
                next_nodes.append(child)

        if not next_nodes:
            break
        # Prune: keep top candidates
        current_nodes = sorted(next_nodes, key=lambda n: n.score, reverse=True)[
            :branches_per_step
        ]

    # Return the best leaf
    best = max(current_nodes, key=lambda n: n.score)
    return best.content


# ---------------------------------------------------------------------------
# 3. ReAct Pattern (Reasoning + Acting)
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    """A callable tool for the ReAct agent."""
    name: str
    description: str
    function: callable


@dataclass
class ReActStep:
    """A single step in a ReAct trace."""
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None


def build_tool_descriptions(tools: list[Tool]) -> str:
    lines = []
    for t in tools:
        lines.append(f"- {t.name}: {t.description}")
    return "\n".join(lines)


def react_agent(
    query: str,
    tools: list[Tool],
    max_steps: int = 5,
    model: str = "gpt-4o-mini",
) -> tuple[str, list[ReActStep]]:
    """
    Run a ReAct agent that reasons and acts in a loop.

    Args:
        query: The user's question.
        tools: List of Tool objects available to the agent.
        max_steps: Maximum reasoning+acting iterations.
        model: LLM model name.

    Returns:
        Tuple of (final_answer, list_of_steps).
    """
    tool_map = {t.name: t for t in tools}
    tool_desc = build_tool_descriptions(tools)

    system_prompt = (
        "You are a reasoning agent. Use the Thought/Action/Observation format.\n\n"
        "Available tools:\n"
        f"{tool_desc}\n\n"
        "Format for each step:\n"
        "Thought: <your reasoning>\n"
        "Action: <tool name>\n"
        "Action Input: <input to the tool>\n\n"
        "When you have the final answer, use:\n"
        "Thought: I now have enough information.\n"
        "Action: finish\n"
        "Action Input: <final answer>"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    steps: list[ReActStep] = []

    for _ in range(max_steps):
        raw = call_llm(messages, model=model, temperature=0.2)
        messages.append({"role": "assistant", "content": raw})

        step = parse_react_step(raw)
        steps.append(step)

        if step.action == "finish":
            return step.action_input or "", steps

        # Execute tool
        if step.action in tool_map:
            try:
                obs = tool_map[step.action].function(step.action_input or "")
            except Exception as e:
                obs = f"Error: {e}"
        else:
            obs = f"Unknown tool: {step.action}"

        step.observation = obs
        messages.append({"role": "user", "content": f"Observation: {obs}"})

    return "Maximum steps reached.", steps


def parse_react_step(text: str) -> ReActStep:
    """Parse Thought/Action/Action Input from raw LLM output."""
    thought = ""
    action = None
    action_input = None

    thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    action_match = re.search(r"Action:\s*(.+)", text)
    if action_match:
        action = action_match.group(1).strip()

    input_match = re.search(r"Action Input:\s*(.+)", text, re.DOTALL)
    if input_match:
        action_input = input_match.group(1).strip()

    return ReActStep(thought=thought, action=action, action_input=action_input)


# ---------------------------------------------------------------------------
# 4. Least-to-Most Prompting
# ---------------------------------------------------------------------------

def least_to_most(
    problem: str,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Decompose a problem into sub-problems ordered from easiest to
    hardest, then solve each sequentially with accumulated context.

    Args:
        problem: The complex problem to solve.
        model: LLM model name.

    Returns:
        The final answer.
    """
    # Step 1: Decompose
    decomp_prompt = [
        {
            "role": "user",
            "content": (
                f"Break this problem into a list of simpler sub-problems, "
                f"ordered from easiest to hardest. Return a JSON array of strings.\n\n"
                f"Problem: {problem}"
            ),
        }
    ]
    try:
        sub_problems = call_llm_json(decomp_prompt, model=model)
        if not isinstance(sub_problems, list):
            sub_problems = [problem]
    except (json.JSONDecodeError, Exception):
        sub_problems = [problem]

    print(f"  Decomposed into {len(sub_problems)} sub-problems:")
    for i, sp in enumerate(sub_problems, 1):
        print(f"    {i}. {sp}")

    # Step 2: Solve sequentially with accumulated context
    context = ""
    for i, sub in enumerate(sub_problems, 1):
        solve_prompt = [
            {
                "role": "user",
                "content": (
                    f"Previous context:\n{context}\n\n"
                    f"Sub-problem {i}/{len(sub_problems)}: {sub}\n\n"
                    f"Solve this sub-problem and provide your answer."
                ),
            }
        ]
        solution = call_llm(solve_prompt, model=model, temperature=0.3)
        context += f"\nSub-problem {i}: {sub}\nAnswer: {solution}\n"
        print(f"  Solved sub-problem {i}: {solution[:100]}...")

    return solution


# ---------------------------------------------------------------------------
# 5. Skeleton-of-Thought (Parallel Outline + Expand)
# ---------------------------------------------------------------------------

def skeleton_of_thought(
    topic: str,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Generate a skeleton outline first, then expand each point
    independently (potentially in parallel).

    Args:
        topic: The topic to write about.
        model: LLM model name.

    Returns:
        Fully expanded content.
    """
    # Step 1: Generate skeleton
    skeleton_prompt = [
        {
            "role": "user",
            "content": (
                f"Generate a numbered outline for a detailed response about: {topic}\n"
                f"Return JSON array of objects with 'title' and 'key_points' (array of strings).\n"
                f'[{{"title": "Section 1", "key_points": ["point a", "point b"]}}]'
            ),
        }
    ]
    try:
        skeleton = call_llm_json(skeleton_prompt, model=model)
        if not isinstance(skeleton, list):
            skeleton = [{"title": topic, "key_points": []}]
    except (json.JSONDecodeError, Exception):
        skeleton = [{"title": topic, "key_points": []}]

    print(f"  Generated skeleton with {len(skeleton)} sections:")
    for section in skeleton:
        title = section.get("title", "Untitled")
        points = section.get("key_points", [])
        print(f"    - {title} ({len(points)} points)")

    # Step 2: Expand each section
    def expand_section(section: dict) -> str:
        title = section.get("title", "Untitled")
        points = section.get("key_points", [])
        expand_prompt = [
            {
                "role": "user",
                "content": (
                    f"Expand the following outline section into a well-written "
                    f"2-3 paragraph response.\n\n"
                    f"Section: {title}\n"
                    f"Key points: {json.dumps(points)}"
                ),
            }
        ]
        return call_llm(expand_prompt, model=model, temperature=0.6)

    expansions = []
    # Parallel expansion
    with ThreadPoolExecutor(max_workers=len(skeleton)) as pool:
        futures = {pool.submit(expand_section, s): s for s in skeleton}
        for f in as_completed(futures):
            expansions.append(f.result())

    return "\n\n".join(expansions)


# ---------------------------------------------------------------------------
# Main: Run all demos
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  ADVANCED PROMPTING TECHNIQUES DEMO")
    print("=" * 70)

    # ---- Self-Consistency ----
    print("\n\n--- 1. Self-Consistency ---")
    problem = "A farmer has 15 chickens. Each chicken lays 2 eggs per day. How many eggs does the farmer collect in a week?"
    result = self_consistency(problem, n_samples=3)
    print(f"Problem: {problem}")
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Votes: {result.votes}")

    # ---- Tree-of-Thought ----
    print("\n\n--- 2. Tree-of-Thought ---")
    tot_problem = (
        "I have 3 shirts (red, blue, green) and 2 pants (black, white). "
        "How many outfits can I make, and list 3 of them?"
    )
    tot_result = tree_of_thought(tot_problem, branches_per_step=2, max_depth=2)
    print(f"Problem: {tot_problem}")
    print(f"Result:\n{tot_result}")

    # ---- ReAct ----
    print("\n\n--- 3. ReAct Agent ---")

    # Define simple tools
    def calculator(expr: str) -> str:
        try:
            return str(eval(expr))
        except Exception as e:
            return f"Error: {e}"

    def lookup(query: str) -> str:
        # Simulated knowledge lookup
        knowledge = {
            "speed of light": "299,792,458 m/s",
            "pi": "3.14159265359",
            "euler's number": "2.71828182846",
        }
        for k, v in knowledge.items():
            if k in query.lower():
                return v
        return f"No data found for: {query}"

    tools = [
        Tool(name="calculator", description="Evaluate a math expression. Input: a math expression string.", function=calculator),
        Tool(name="lookup", description="Look up a scientific constant or fact. Input: query string.", function=lookup),
    ]

    answer, steps = react_agent(
        "What is (speed of light in m/s) / (pi * 1000)?",
        tools=tools,
        max_steps=4,
    )
    print(f"Final answer: {answer}")
    print(f"Steps taken: {len(steps)}")
    for i, s in enumerate(steps, 1):
        print(f"  Step {i}: {s.thought[:80]}...")
        if s.action:
            print(f"    Action: {s.action}({s.action_input})")
        if s.observation:
            print(f"    Observation: {s.observation}")

    # ---- Least-to-Most ----
    print("\n\n--- 4. Least-to-Most ---")
    ltm_problem = (
        "If Alice is 3 years older than Bob, Bob is twice as old as Charlie, "
        "and Charlie is 5, how old is Alice?"
    )
    ltm_result = least_to_most(ltm_problem)
    print(f"Final answer: {ltm_result[:200]}")

    # ---- Skeleton-of-Thought ----
    print("\n\n--- 5. Skeleton-of-Thought ---")
    sot_result = skeleton_of_thought("benefits of learning to code")
    print(f"Expanded content:\n{sot_result[:500]}...")

    print("\n" + "=" * 70)
    print("  All demos complete!")
    print("=" * 70)
