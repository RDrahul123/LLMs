# Module 3: Prompt Engineering

> **Level:** Beginner-Intermediate | **Time:** ~1.5 hours | **Prerequisites:** Modules 1-2 | **Code:** examples.py, templates.py, exercises.md

---

## Why Prompt Engineering?

The same LLM can produce vastly different results based on *how* you ask. Prompt engineering is the skill of crafting inputs that reliably get the outputs you want.

---

## What is Prompt Engineering?

Prompt engineering is the practice of designing inputs to an LLM to guide it toward producing desired outputs. It combines:

- **Clarity** — being explicit about what you want
- **Structure** — organizing information logically
- **Context** — providing the right background
- **Examples** — showing, not just telling

---

## Anatomy of a Good Prompt

A well-crafted prompt contains these elements:

```
┌─────────────────────────────────────────┐
│  ROLE: Who should the LLM be?           │
│  "You are a senior Python developer"    │
├─────────────────────────────────────────┤
│  CONTEXT: What's the situation?          │
│  "I'm building a REST API for..."       │
├─────────────────────────────────────────┤
│  TASK: What should the LLM do?          │
│  "Review this code for security..."     │
├─────────────────────────────────────────┤
│  FORMAT: How should it respond?         │
│  "Return a JSON with..."               │
├─────────────────────────────────────────┤
│  EXAMPLES: Show what you want           │
│  "Input: ... → Output: ..."            │
├─────────────────────────────────────────┤
│  CONSTRAINTS: What to avoid             │
│  "Do not include..."                    │
└─────────────────────────────────────────┘
```

---

## Prompting Techniques

### Zero-Shot Prompting

Ask the model to perform a task with no examples.

```
Classify the sentiment of this review as positive, negative, or neutral:
"This movie was absolutely wonderful!"
```

### Few-Shot Prompting

Provide examples to guide the model.

```
Classify the sentiment:

Review: "I loved it!" → Positive
Review: "Terrible waste of time." → Negative
Review: "It was okay, nothing special." → Neutral

Review: "This movie was absolutely wonderful!" →
```

### Chain-of-Thought (CoT) Prompting

Ask the model to show its reasoning step by step.

```
Solve this step by step:

Q: A store sells shirts for $25 each. If you buy 3 or more, you get 20% off.
   How much do 5 shirts cost?

A: Let me think step by step:
   1. Price per shirt: $25
   2. Number of shirts: 5
   3. Since 5 >= 3, the discount applies
   4. Discount: 20% of $25 = $5 off per shirt
   5. Discounted price: $25 - $5 = $20 per shirt
   6. Total: 5 × $20 = $100

   The answer is $100.
```

---

## Role Prompting & System Prompts

### Role Prompting

Assigning a role to the LLM changes its behavior significantly.

```
You are an expert technical writer with 15 years of experience.
You write clear, concise documentation for APIs.
Use markdown formatting with code examples.

Explain what a webhook is to a junior developer.
```

### System Prompts

System prompts set persistent behavior for multi-turn conversations:

```python
system_prompt = """
You are a helpful coding assistant. Rules:
1. Always provide working code examples
2. Explain complex concepts in simple terms
3. When reviewing code, focus on: readability, performance, security
4. If you're unsure, say so rather than guessing
"""
```

---

## Prompt Templates & Reusability

### Using templates.py

```python
from templates import PromptTemplate

template = PromptTemplate(
    system="You are a {role} expert.",
    user="Explain {topic} for a {audience} audience."
)

prompt = template.format(role="Python", topic="decorators", audience="beginner")
```

### Common Patterns

**Classification Template:**
```
Classify the following text into one of these categories: {categories}

Text: {text}

Category:
```

**Extraction Template:**
```
Extract the following information from the text:
- {field_1}
- {field_2}
- {field_3}

Text: {text}

Return as JSON:
```

---

## Common Mistakes & Fixes

| Mistake | Example | Fix |
|---------|---------|-----|
| Vague instructions | "Write about AI" | "Write a 200-word summary of recent advances in LLMs" |
| No format specified | "Tell me about dogs" | "List 5 facts about dogs as bullet points" |
| Too many tasks | "Summarize, translate, and critique this" | Do one at a time |
| No examples | "Classify sentiment" | Add 2-3 examples first |
| Assuming knowledge | "Make it like the other one" | Provide the full context |
| Ignoring length limits | "Write a novel" | "Write 150-200 words" |

---

## Try It Yourself

```bash
python 02-prompt-engineering/examples.py
```

---

## Exercises

Complete the 5 exercises in [exercises.md](exercises.md) to practice what you've learned.

---

## What's Next?

You've learned how to talk to LLMs effectively. Now let's learn how to build with them programmatically.

**→ [Module 4: Working with LLM APIs](../03-llm-apis/README.md)**

Learn to call OpenAI, Claude, and other LLM APIs from Python.

---

## Further Reading

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering)
- [Google Gemini Prompt Design Guide](https://ai.google.dev/gemini-api/docs/prompting-intro)
- [DAIR.ai Prompt Engineering Guide](https://www.promptingguide.ai/)

## Visual Deep Dive

Continue to **[Module 3 Deep Dive](deep-dive.md)** for prompt structure, schema validation, safe treatment of untrusted text, and a measurable iteration workflow.
