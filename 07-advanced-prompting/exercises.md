# Module 07: Advanced Prompting — Exercises

Complete these exercises to solidify your understanding of advanced prompting techniques.

---

## Exercise 1: Self-Consistency for Math Problems

**Objective**: Implement a self-consistency voting system for arithmetic word problems.

**Task**: Write a function `self_consistency_math(problem, n_samples=5)` that:
1. Generates `n_samples` Chain-of-Thought completions for a math problem
2. Extracts the numeric answer from each completion
3. Returns the majority-vote answer and confidence percentage
4. Handle ties by returning the lowest-confidence result with a warning

**Test cases**:
- "A store has 24 apples. They sell 1/3 in the morning and 1/2 of the remainder in the afternoon. How many are left?"
- "If you double a number and add 7, you get 23. What is the number?"
- "A rectangular room is 12m by 8m. What is the area?"

**Starter code**:
```python
def self_consistency_math(problem: str, n_samples: int = 5) -> dict:
    """
    Returns:
        {
            "answer": float,
            "confidence": float,  # 0.0 to 1.0
            "n_valid_samples": int,
            "all_answers": list[float]
        }
    """
    # Your code here
    pass
```

<details>
<summary>Click for Solution</summary>

```python
import openai
from collections import Counter

def self_consistency_math(problem: str, n_samples: int = 5) -> dict:
    client = openai.OpenAI()
    answers = []

    prompt = (
        "Solve this math problem step by step.\n"
        "On the LAST line, write ONLY the numeric answer in the format: ANSWER: <number>\n\n"
        f"Problem: {problem}"
    )

    for _ in range(n_samples):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )
        text = resp.choices[0].message.content
        
        import re
        match = re.search(r"ANSWER:\s*([-\d.]+)", text)
        if match:
            try:
                answers.append(float(match.group(1)))
            except ValueError:
                continue
    
    if not answers:
        return {"answer": None, "confidence": 0.0, "n_valid_samples": 0, "all_answers": []}
    
    # Round for grouping (handle floating point)
    rounded = [round(a, 2) for a in answers]
    counts = Counter(rounded)
    best_answer, best_count = counts.most_common(1)[0]
    
    return {
        "answer": best_answer,
        "confidence": best_count / len(rounded),
        "n_valid_samples": len(answers),
        "all_answers": answers,
    }
```

</details>

---

## Exercise 2: Tree-of-Thought for Planning

**Objective**: Build a Tree-of-Thought planner that explores multiple strategies.

**Task**: Write a function `tot_planner(goal, constraints, branches=3, depth=2)` that:
1. At each level, generates `branches` different plans
2. Evaluates each plan using an LLM scoring prompt (1-10)
3. Keeps only the top half of candidates at each level
4. Returns the best final plan with its score

**Example**:
```python
plan = tot_planner(
    goal="Organize a 50-person team retreat within 2 weeks and $5000 budget",
    constraints=["Must be within 100 miles", "Outdoor activities preferred", "Need catering"]
)
```

**Starter code**:
```python
def tot_planner(goal: str, constraints: list[str], branches: int = 3, depth: int = 2) -> dict:
    """
    Returns:
        {
            "plan": str,
            "score": float,
            "planning_path": list[dict]  # decisions at each level
        }
    """
    # Your code here
    pass
```

<details>
<summary>Click for Solution</summary>

```python
import openai
import json
import re

def tot_planner(goal: str, constraints: list[str], branches: int = 3, depth: int = 2) -> dict:
    client = openai.OpenAI()
    
    constraints_text = "\n".join(f"- {c}" for c in constraints)
    planning_path = []
    
    # Root prompt
    candidates = [
        f"Goal: {goal}\nConstraints:\n{constraints_text}"
    ]
    
    for level in range(depth):
        new_candidates = []
        
        for candidate in candidates:
            # Generate branches
            gen_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": (
                    f"Given this planning context:\n{candidate}\n\n"
                    f"Generate {branches} distinct next steps or decisions. "
                    f"Return JSON array of objects with 'step' (string) and 'rationale' (string)."
                )}],
                temperature=0.8,
            )
            raw = gen_resp.choices[0].message.content
            cleaned = re.sub(r"```json\s*|\s*```", "", raw)
            
            try:
                steps = json.loads(cleaned)
            except json.JSONDecodeError:
                steps = [{"step": raw, "rationale": "direct response"}]
            
            # Evaluate each step
            for step in steps:
                eval_resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": (
                        f"Rate this plan step 1-10 for feasibility and effectiveness.\n"
                        f"Goal: {goal}\nConstraints: {constraints_text}\n"
                        f"Step: {step.get('step', '')}\n"
                        f"Return JSON: {{\"score\": <number>, \"reasoning\": \"<string>\"}}"
                    )}],
                    temperature=0.2,
                )
                eval_raw = eval_resp.choices[0].message.content
                eval_cleaned = re.sub(r"```json\s*|\s*```", "", eval_raw)
                
                try:
                    evaluation = json.loads(eval_cleaned)
                    score = evaluation.get("score", 5)
                except json.JSONDecodeError:
                    score = 5
                
                new_candidates.append({
                    "content": candidate + "\n" + step.get("step", ""),
                    "score": score,
                    "step": step.get("step", ""),
                })
        
        # Keep top half
        new_candidates.sort(key=lambda x: x["score"], reverse=True)
        keep = max(1, branches)
        candidates = [c["content"] for c in new_candidates[:keep]]
        planning_path.append(new_candidates[:keep])
    
    best = new_candidates[0] if new_candidates else {"content": "", "score": 0}
    return {
        "plan": best["content"],
        "score": best["score"],
        "planning_path": planning_path,
    }
```

</details>

---

## Exercise 3: ReAct Agent with Calculator

**Objective**: Build a ReAct agent that can reason and use a calculator tool.

**Task**: Implement a full ReAct loop that:
1. Parses Thought/Action/Action Input from LLM output
2. Executes the calculator tool
3. Passes observations back to the LLM
4. Terminates when Action is "finish"
5. Returns the full trace of steps

**Tool definition**:
```python
def calculator(expression: str) -> str:
    """Safely evaluate a math expression."""
    import ast
    import math
    import operator

    binary_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            value = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in binary_ops:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponent magnitude must be at most 10")
            value = binary_ops[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = evaluate(node.operand)
            value = operand if isinstance(node.op, ast.UAdd) else -operand
        else:
            raise ValueError("Only numeric literals and basic arithmetic are allowed")

        if not math.isfinite(value) or abs(value) > 1e12:
            raise ValueError("Result is outside the allowed numeric range")
        return value

    try:
        if len(expression) > 100:
            raise ValueError("Expression is too long")
        tree = ast.parse(expression, mode='eval')
        return str(evaluate(tree))
    except Exception as e:
        return f"Error: {e}"
```

**Starter code**:
```python
def react_calculator(query: str, max_steps: int = 5) -> dict:
    """
    Returns:
        {
            "answer": str,
            "trace": list[dict]  # [{"thought": ..., "action": ..., "observation": ...}]
        }
    """
    # Your code here
    pass
```

<details>
<summary>Click for Solution</summary>

```python
import openai
import re
import ast
import math
import operator

def calculator(expression: str) -> str:
    try:
        if len(expression) > 100:
            raise ValueError("Expression is too long")
        tree = ast.parse(expression, mode='eval')
        binary_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }

        def evaluate(node):
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                value = node.value
            elif isinstance(node, ast.BinOp) and type(node.op) in binary_ops:
                left, right = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Pow) and abs(right) > 10:
                    raise ValueError("Exponent magnitude must be at most 10")
                value = binary_ops[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                operand = evaluate(node.operand)
                value = operand if isinstance(node.op, ast.UAdd) else -operand
            else:
                raise ValueError("Only numeric literals and basic arithmetic are allowed")
            if not math.isfinite(value) or abs(value) > 1e12:
                raise ValueError("Result is outside the allowed numeric range")
            return value

        return str(evaluate(tree))
    except Exception as e:
        return f"Error: {e}"

def react_calculator(query: str, max_steps: int = 5) -> dict:
    client = openai.OpenAI()
    trace = []
    
    system = (
        "You are a reasoning agent. Use this format:\n"
        "Thought: <reasoning>\n"
        "Action: calculator\n"
        "Action Input: <math expression>\n\n"
        "Or when done:\n"
        "Thought: I now have the answer.\n"
        "Action: finish\n"
        "Action Input: <final answer>"
    )
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]
    
    for _ in range(max_steps):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
        )
        raw = resp.choices[0].message.content
        messages.append({"role": "assistant", "content": raw})
        
        # Parse
        thought = ""
        action = None
        action_input = None
        
        t = re.search(r"Thought:\s*(.+?)(?=Action:|$)", raw, re.DOTALL)
        if t:
            thought = t.group(1).strip()
        
        a = re.search(r"Action:\s*(.+)", raw)
        if a:
            action = a.group(1).strip()
        
        ai = re.search(r"Action Input:\s*(.+)", raw, re.DOTALL)
        if ai:
            action_input = ai.group(1).strip()
        
        if action == "finish":
            trace.append({"thought": thought, "action": "finish", "observation": action_input})
            return {"answer": action_input or "", "trace": trace}
        
        # Execute tool
        observation = calculator(action_input or "0")
        trace.append({"thought": thought, "action": action or "", "observation": observation})
        messages.append({"role": "user", "content": f"Observation: {observation}"})
    
    return {"answer": "Max steps reached", "trace": trace}
```

</details>

---

## Exercise 4: Prompt Chain for Content Generation

**Objective**: Design a prompt chain that generates a blog post from a topic.

**Task**: Create a `PromptChain` with these steps:
1. `brainstorm` — Generate 5 key points about the topic
2. `outline` — Create a structured outline from the key points
3. `draft` — Write the full post from the outline
4. `edit` — Review and polish the draft

Each step should:
- Have a proper `prompt_template` function
- Include a `parser` (use `parse_json_array` or `parse_json_object`)
- Validate that it produced meaningful output

**Test it**:
```python
chain = build_blog_chain()
result = chain.run({"topic": "Getting started with prompt engineering"})
print(result.outputs.get("edit", "")[:500])
```

<details>
<summary>Click for Solution</summary>

```python
import openai
import re
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Reuse ChainStep, PromptChain, parse_json_array, parse_json_object
# from prompt_chaining.py

def build_blog_chain():
    def brainstorm_prompt(ctx):
        topic = ctx.get("topic", "")
        return [{"role": "user", "content": (
            f"Generate 5 key points about: {topic}\n"
            f"Return JSON array of objects with 'point' and 'why_important'."
        )}]
    
    def outline_prompt(ctx):
        points = ctx.get("brainstorm", [])
        return [{"role": "user", "content": (
            f"Create a blog post outline from these key points:\n"
            f"{json.dumps(points, indent=2)}\n\n"
            f"Return JSON with 'title', 'intro_hooks' (array), "
            f"'sections' (array of objects with 'heading' and 'bullets' array)."
        )}]
    
    def draft_prompt(ctx):
        outline = ctx.get("outline", {})
        return [{"role": "user", "content": (
            f"Write a complete blog post from this outline.\n"
            f"Title: {outline.get('title', '')}\n"
            f"Outline: {json.dumps(outline, indent=2)}\n\n"
            f"Write 400-600 words with clear sections."
        )}]
    
    def edit_prompt(ctx):
        draft = ctx.get("draft", "")
        return [{"role": "user", "content": (
            f"Edit and polish this blog post draft.\n"
            f"Fix grammar, improve flow, strengthen the conclusion.\n\n"
            f"Draft:\n{draft}\n\n"
            f"Output the final edited version."
        )}]
    
    def validate_points(output):
        if isinstance(output, list) and len(output) >= 3:
            return True, ""
        return False, "Need at least 3 key points"
    
    def validate_outline(output):
        if isinstance(output, dict) and "sections" in output:
            return True, ""
        return False, "Outline must have sections"
    
    from prompt_chaining import ChainStep, PromptChain, parse_json_array, parse_json_object
    
    return PromptChain(
        steps=[
            ChainStep("brainstorm", brainstorm_prompt, parse_json_array, validate_points),
            ChainStep("outline", outline_prompt, parse_json_object, validate_outline),
            ChainStep("draft", draft_prompt),
            ChainStep("edit", edit_prompt),
        ],
        fail_fast=True,
    )
```

</details>

---

## Exercise 5: Meta-Prompt That Generates Prompts

**Objective**: Write a meta-prompt that generates high-quality prompts for arbitrary tasks.

**Task**: Create a function `meta_prompt_generator(task_description)` that:
1. Analyzes the task description
2. Generates a detailed system prompt that handles edge cases
3. Includes 2-3 few-shot examples in the generated prompt
4. Returns the prompt with a self-evaluation score

**Requirements**:
- The generated prompt must be production-ready (not just instructions)
- Include error handling guidance in the generated prompt
- The meta-prompt should output JSON with `prompt` and `evaluation` fields

**Starter code**:
```python
def meta_prompt_generator(task_description: str) -> dict:
    """
    Returns:
        {
            "prompt": str,       # The generated production-ready prompt
            "evaluation": {
                "clarity": int,      # 1-10
                "completeness": int, # 1-10
                "edge_cases": int,   # 1-10
            },
            "usage_example": str  # How to use the generated prompt
        }
    """
    # Your code here
    pass
```

<details>
<summary>Click for Solution</summary>

```python
import openai
import json
import re

def meta_prompt_generator(task_description: str) -> dict:
    client = openai.OpenAI()
    
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "You are an expert prompt engineer. Generate production-ready prompts "
                "that include:\n"
                "1. Clear role definition\n"
                "2. Specific instructions with constraints\n"
                "3. 2-3 few-shot examples\n"
                "4. Error handling guidance\n"
                "5. Output format specification\n\n"
                "Return JSON with: 'prompt', 'evaluation' (object with clarity, completeness, "
                "edge_cases scores 1-10), and 'usage_example'."
            )},
            {"role": "user", "content": f"Task: {task_description}"},
        ],
        temperature=0.7,
    )
    
    raw = resp.choices[0].message.content
    cleaned = re.sub(r"```json\s*|\s*```", "", raw)
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "prompt": raw,
            "evaluation": {"clarity": 5, "completeness": 5, "edge_cases": 5},
            "usage_example": f"Use the prompt with: {task_description}",
        }
```

</details>

---

## Exercise 6: Compare Techniques on the Same Problem

**Objective**: Run the same problem through multiple techniques and compare results.

**Task**: Write a comparison function that:
1. Takes a problem and runs it through:
   - Zero-shot (baseline)
   - Chain-of-Thought (single)
   - Self-Consistency (5 samples)
   - Least-to-Most
2. Compares outputs and identifies where techniques agree/disagree
3. Returns a structured comparison report

**Test problem**: "If a car travels at 45 mph for 3.5 hours, then at 60 mph for 2 hours, what is the average speed for the entire trip?"

**Expected output**:
```python
{
    "problem": "...",
    "techniques": {
        "zero_shot": {"answer": ..., "correct": bool},
        "cot": {"answer": ..., "correct": bool},
        "self_consistency": {"answer": ..., "confidence": ..., "correct": bool},
        "least_to_most": {"answer": ..., "correct": bool},
    },
    "consensus": bool,
    "correct_answer": float,
}
```

<details>
<summary>Click for Solution</summary>

```python
import openai
import re
from collections import Counter

def compare_techniques(problem: str) -> dict:
    client = openai.OpenAI()
    correct_answer = 50.0  # (45*3.5 + 60*2) / 5.5 = 307.5/5.5 ≈ 55.9
    
    # Actually let me recalculate: total distance = 45*3.5 + 60*2 = 157.5 + 120 = 277.5
    # total time = 3.5 + 2 = 5.5
    # average = 277.5 / 5.5 = 50.45 mph
    correct_answer = 277.5 / 5.5  # ~50.45
    
    def extract_number(text):
        # Try to find a number that looks like speed
        matches = re.findall(r"(\d+\.?\d*)\s*mph", text, re.IGNORECASE)
        if matches:
            return float(matches[-1])
        # Fallback: find any number
        matches = re.findall(r"(\d+\.?\d*)", text)
        return float(matches[-1]) if matches else None
    
    def is_close(a, b, tolerance=1.0):
        return abs(a - b) <= tolerance
    
    results = {}
    
    # 1. Zero-shot
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"What is the average speed? {problem}"}],
        temperature=0.0,
    )
    zero_shot_ans = extract_number(resp.choices[0].message.content)
    results["zero_shot"] = {
        "answer": zero_shot_ans,
        "correct": is_close(zero_shot_ans, correct_answer) if zero_shot_ans else False,
    }
    
    # 2. Chain-of-Thought
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": (
            f"Solve step by step. {problem}\n\n"
            f"Show your work, then state: ANSWER: <number> mph"
        )}],
        temperature=0.0,
    )
    cot_text = resp.choices[0].message.content
    cot_ans = extract_number(cot_text)
    results["cot"] = {
        "answer": cot_ans,
        "correct": is_close(cot_ans, correct_answer) if cot_ans else False,
    }
    
    # 3. Self-Consistency (5 samples)
    sc_answers = []
    for _ in range(5):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": (
                f"Think step by step, then answer: ANSWER: <number> mph\n\n{problem}"
            )}],
            temperature=0.7,
        )
        ans = extract_number(resp.choices[0].message.content)
        if ans is not None:
            sc_answers.append(round(ans, 1))
    
    if sc_answers:
        vote = Counter(sc_answers).most_common(1)[0][0]
        confidence = Counter(sc_answers).most_common(1)[0][1] / len(sc_answers)
    else:
        vote, confidence = None, 0.0
    
    results["self_consistency"] = {
        "answer": vote,
        "confidence": confidence,
        "correct": is_close(vote, correct_answer) if vote else False,
    }
    
    # 4. Least-to-Most
    decomp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": (
            f"Break into sub-problems: {problem}\n"
            f"Return JSON array of sub-problems."
        )}],
        temperature=0.3,
    )
    # Solve sequentially
    solve_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": (
            f"Solve this step by step:\n{problem}\n\n"
            f"1. Find total distance\n2. Find total time\n3. Calculate average speed\n\n"
            f"ANSWER: <number> mph"
        )}],
        temperature=0.0,
    )
    ltm_ans = extract_number(solve_resp.choices[0].message.content)
    results["least_to_most"] = {
        "answer": ltm_ans,
        "correct": is_close(ltm_ans, correct_answer) if ltm_ans else False,
    }
    
    consensus = sum(1 for r in results.values() if r.get("correct")) >= 3
    
    return {
        "problem": problem,
        "correct_answer": round(correct_answer, 2),
        "techniques": results,
        "consensus": consensus,
        "techniques_correct": sum(1 for r in results.values() if r.get("correct")),
    }
```

</details>

---

## Bonus Challenge

Implement a **prompt chain** that automatically:
1. Detects which technique to use based on problem type
2. Runs that technique
3. Validates the result
4. Falls back to a different technique if validation fails

This combines Exercise 4's chaining with Exercises 1-3's techniques.
