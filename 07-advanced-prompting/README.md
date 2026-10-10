# Module 07: Advanced Prompting

## Why Advanced Prompting Matters

Basic zero-shot and few-shot prompting get you started, but real-world tasks demand structured reasoning strategies. Advanced prompting techniques give you:

- **Reliability**: Reducing variance across runs
- **Composability**: Breaking complex tasks into solvable sub-tasks
- **Interactivity**: Connecting LLMs to tools and external data
- **Self-improvement**: Letting the model critique and refine its own outputs

---

## Techniques Covered

### 1. Self-Consistency Prompting

Generate multiple Chain-of-Thought (CoT) completions for the same problem, then take a **majority vote** on the final answer. This reduces randomness and improves accuracy on reasoning-heavy tasks.

**Key idea**: One CoT path can be wrong. Many paths converging on the same answer is strong evidence.

```python
from collections import Counter
from openai import OpenAI

client = OpenAI()

def self_consistency(problem, n_samples=5):
    answers = []
    for _ in range(n_samples):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Think step by step, then give your final answer on its own line prefixed with 'Answer: '.\n\nProblem: {problem}"
            }],
            temperature=0.7,
        )
        text = response.choices[0].message.content
        answer_line = [l for l in text.splitlines() if l.strip().startswith("Answer:")]
        if answer_line:
            answers.append(answer_line[0].replace("Answer:", "").strip())
    # Majority vote
    if not answers:
        raise ValueError("No parseable answers were produced")
    return Counter(answers).most_common(1)[0][0]

result = self_consistency("If a train travels 60mph for 2.5 hours, how far does it go?")
print(result)  # "150 miles"
```

### 2. Tree-of-Thought (ToT)

Branch the reasoning process. At each step, the model generates multiple candidate continuations, evaluates them, and prunes weak branches. This explores a search tree instead of a single path.

```python
def tree_of_thought(problem, branches=3, depth=3):
    """Simple ToT: generate branches, evaluate, keep best."""
    candidates = [problem]
    for step in range(depth):
        new_candidates = []
        for cand in candidates:
            # Generate branch continuations
            branch_prompt = (
                f"Given this partial reasoning:\n{cand}\n\n"
                f"Generate {branches} distinct next steps to continue solving. "
                f"Label them Step A, Step B, Step C."
            )
            response = call_llm(branch_prompt)
            # Evaluate each branch
            eval_prompt = (
                f"Rate each step 1-10 for how likely it leads to a correct solution.\n{response}"
            )
            scores = call_llm(eval_prompt)
            best = extract_best_branch(response, scores)
            new_candidates.append(best)
        candidates = new_candidates
    return candidates[0]
```

### 3. ReAct Pattern (Reasoning + Acting)

Interleave **thought** (reasoning) and **action** (tool calls) in a loop. The model thinks about what to do, takes an action, observes the result, and repeats until it has an answer.

```
Thought: I need to find the population of France.
Action: search("population of France")
Observation: France has approximately 67.4 million people.
Thought: Now I can answer.
Action: finish("67.4 million")
```

```python
def react_loop(query, tools, max_steps=5):
    messages = [{"role": "user", "content": build_react_prompt(query, tools)}]
    for _ in range(max_steps):
        response = call_llm(messages)
        thought, action = parse_react_output(response)
        if action.type == "finish":
            return action.input
        result = execute_tool(action)
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Observation: {result}"})
    return "Max steps reached"
```

### 4. Least-to-Most Prompting

First **decompose** a problem into simpler sub-problems (ordered from least to most complex), then solve each sub-problem in sequence, passing previous answers as context.

```python
def least_to_most(problem):
    # Step 1: Decompose
    decomposition_prompt = (
        f"Break this problem into simpler sub-problems, "
        f"ordered from easiest to hardest:\n\n{problem}"
    )
    sub_problems = call_llm(decomposition_prompt)

    # Step 2: Solve sequentially
    context = ""
    for sub in sub_problems:
        solution = call_llm(
            f"Given context:\n{context}\n\nSolve: {sub}"
        )
        context += f"\n{sub} -> {solution}"
    return extract_final_answer(context)
```

### 5. Skeleton-of-Thought (SoT)

Generate a **skeleton/outline** first, then expand each point in parallel. This produces faster, more structured, and better-organized responses.

```python
def skeleton_of_thought(topic):
    # Step 1: Generate skeleton
    skeleton_prompt = (
        f"Give a numbered outline skeleton for a detailed response about: {topic}\n"
        f"Only the skeleton points, no details."
    )
    skeleton = call_llm(skeleton_prompt)
    points = parse_skeleton(skeleton)

    # Step 2: Expand each point (can be parallelized)
    expansions = []
    for point in points:
        expansion = call_llm(f"Expand on this point in 2-3 sentences: {point}")
        expansions.append(expansion)

    return "\n\n".join(expansions)
```

### 6. Automatic Prompt Optimization (APE)

Use the LLM itself to generate and refine prompts. Start with a task, generate candidate prompts, evaluate them on examples, and iterate.

```python
def ape_optimize(task_description, examples, iterations=3):
    # Generate initial prompt candidates
    candidates = call_llm(
        f"Generate 5 different prompts for this task:\n{task_description}"
    )
    # Evaluate on examples
    best_prompt = evaluate_candidates(candidates, examples)
    # Iterate
    for _ in range(iterations):
        improved = call_llm(
            f"This prompt got score {best_prompt.score}:\n{best_prompt.text}\n"
            f"Suggest 3 improvements to make it more accurate."
        )
        best_prompt = evaluate_candidates(improved, examples)
    return best_prompt.text
```

### 7. Prompt Chaining

Break complex workflows into sequential steps, where each step's output feeds into the next. Add validation, error handling, and branching between steps.

```python
chain = PromptChain([
    ChainStep("extract", extract_entities),
    ChainStep("transform", transform_entities),
    ChainStep("validate", validate_results),
    ChainStep("format", format_output),
])
result = chain.run(input_text)
```

---

## Decision Matrix: When to Use Each Technique

| Technique | Best For | Complexity | Latency | Accuracy Gain |
|---|---|---|---|---|
| **Self-Consistency** | Math, logic, classification | Low | High (N calls) | High |
| **Tree-of-Thought** | Planning, creative problems | High | High | Medium-High |
| **ReAct** | Tool-augmented tasks, QA | Medium | Medium | High |
| **Least-to-Most** | Multi-step reasoning, curriculum | Medium | Medium | High |
| **Skeleton-of-Thought** | Long-form writing, presentations | Low | Low-Medium | Medium |
| **APE** | Prompt engineering at scale | Medium | High | High |
| **Prompt Chaining** | Workflows, pipelines | Medium | Medium | High |

---

## Module Structure

| File | Description |
|---|---|
| `README.md` | This module guide |
| `techniques.py` | Runnable demos of all advanced techniques |
| `meta_prompting.py` | Automatic prompt improvement and meta-prompting |
| `prompt_chaining.py` | Multi-step prompt chain framework |
| `exercises.md` | Practice exercises with solutions |

---

## Quick Start

```bash
cd 07-advanced-prompting
pip install openai python-dotenv

# Set your API key
echo "OPENAI_API_KEY=sk-..." > .env

# Run the technique demos
python techniques.py

# Explore meta-prompting
python meta_prompting.py

# Run prompt chaining examples
python prompt_chaining.py
```

---

## Visual Deep Dive

Continue to **[Module 7 Deep Dive](deep-dive.md)** for decision diagrams, worked reasoning patterns, failure modes, and techniques for measuring prompt improvements.

## What's Next?

In **Module 08: AI Agents & Tool Use**, you'll learn how to connect model decisions to controlled tools, manage state, and bound an agent's autonomy.
