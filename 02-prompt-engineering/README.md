# Module 2 — Prompt Engineering 🟡

> **Level:** Beginner → Intermediate | **Time:** ~2 hours | **Code:** Optional

---

## What is Prompt Engineering?

**Prompt Engineering** is the art of crafting inputs to LLMs to get the best possible outputs.

Same model. Different prompt. Wildly different results.

```
❌ Bad prompt:   "Summarize this."
✅ Good prompt:  "Summarize the following article in 3 bullet points 
                 for a non-technical audience. Focus on key takeaways."
```

---

## The Anatomy of a Good Prompt

A well-structured prompt has these components:

```
┌─────────────────────────────────────────────┐
│  ROLE      → Who the model should act as    │
│  CONTEXT   → Background information         │
│  TASK      → What you want done             │
│  FORMAT    → How to structure the output    │
│  EXAMPLES  → Show, don't just tell          │
│  CONSTRAINTS → What to avoid / word limits  │
└─────────────────────────────────────────────┘
```

Not every prompt needs all 6 — but using more gives better results.

---

## Technique 1: Zero-Shot Prompting

Just ask the question directly. No examples provided.

```
Prompt:
"Classify the sentiment of this review as Positive, Negative, or Neutral:
'The product works fine but the packaging was damaged.'"

Output:
"Neutral"
```

**When to use:** Simple tasks, quick tests.

---

## Technique 2: Few-Shot Prompting

Provide examples before your actual question. This "teaches" the model the pattern.

```
Prompt:
"Classify sentiment. Examples:

Review: 'Amazing product, love it!' → Positive
Review: 'Terrible, broke after one day.' → Negative
Review: 'It arrived on time.' → Neutral

Now classify:
Review: 'Good quality but a bit expensive.'"

Output:
"Neutral"
```

**When to use:** Custom classification, specific output formats, edge cases.

---

## Technique 3: Chain-of-Thought (CoT) Prompting

Ask the model to **think step-by-step** before giving an answer.

```
❌ Without CoT:
"If a train travels 60 mph for 2.5 hours, how far does it go?"
→ "150 miles" (sometimes wrong on harder problems)

✅ With CoT:
"If a train travels 60 mph for 2.5 hours, how far does it go?
Think step by step."

→ "Step 1: Distance = Speed × Time
   Step 2: Distance = 60 × 2.5
   Step 3: Distance = 150 miles
   Answer: 150 miles" ✅
```

**When to use:** Math, logic, multi-step reasoning tasks.

---

## Technique 4: Role Prompting (System Prompts)

Tell the model WHO it is. This shapes its entire behavior.

```python
system_prompt = """
You are a senior Python developer with 10 years of experience.
You write clean, well-commented code following PEP 8 standards.
When reviewing code, you always explain WHY something is wrong, not just that it is.
"""

user_message = "Review this code: [code here]"
```

**When to use:** Chatbots, specialized assistants, consistent tone/persona.

---

## Technique 5: Prompt Templates

Reusable prompts with variables — like a function for your AI.

```python
def summarize_prompt(text: str, audience: str, bullets: int) -> str:
    return f"""
    Summarize the following text in {bullets} bullet points.
    Target audience: {audience}
    Be concise and focus on actionable insights.
    
    Text:
    {text}
    """

# Usage
prompt = summarize_prompt(
    text=article,
    audience="business executives",
    bullets=5
)
```

---

## Common Prompt Mistakes (And Fixes)

### ❌ Mistake 1: Being Vague
```
Bad:  "Write something about AI."
Good: "Write a 200-word LinkedIn post about how AI is changing 
       software development, with a professional but conversational tone."
```

### ❌ Mistake 2: Not Specifying Format
```
Bad:  "Give me ideas for my app."
Good: "Give me 5 app feature ideas as a numbered list. 
       For each: name, one-sentence description, and difficulty (Easy/Medium/Hard)."
```

### ❌ Mistake 3: Asking Multiple Questions at Once
```
Bad:  "Explain neural networks, tell me about transformers, 
       and also how does GPT work?"

Good: Ask one clear question at a time.
```

### ❌ Mistake 4: Negative Instructions Only
```
Bad:  "Don't make it too long and don't use jargon."
Good: "Keep it under 100 words. Use simple, everyday language."
```

---

## Advanced: Meta-Prompting

Ask the model to **write a better prompt for you**.

```
"I want to ask an LLM to help me write cold outreach emails for B2B sales.
Write me the best possible system prompt for this use case."
```

---

## The Golden Rules

1. **Be specific** — vague in, vague out
2. **Show examples** — few-shot beats zero-shot for complex tasks
3. **Assign a role** — gives the model a persona and context
4. **Specify format** — tell it exactly how to structure output
5. **Iterate** — prompt engineering is a loop, not a one-shot

---

## 📁 Files in This Module

| File | Description |
|---|---|
| [`examples.py`](./examples.py) | All techniques above as runnable Python |
| [`templates.py`](./templates.py) | Reusable prompt template library |
| [`exercises.md`](./exercises.md) | Practice exercises with solutions |

---

## ✅ What's Next?

👉 **[Module 3 — LLM APIs](../03-llm-apis/README.md)**
