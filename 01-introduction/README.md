# Module 1 — Introduction to LLMs 🟢

> **Level:** Beginner | **Time:** ~1 hour | **Code:** None required

---

## What is a Large Language Model?

A **Large Language Model (LLM)** is an AI model trained on massive amounts of text data to understand and generate human language.

Think of it like this:
- You read millions of books → you become very good at writing and reasoning
- An LLM reads billions of web pages, books, and code → it learns to predict and generate text

**Examples of LLMs you've probably heard of:**
| Model | Made By | Known For |
|---|---|---|
| GPT-4o | OpenAI | General purpose, very capable |
| Claude 3.5 Sonnet | Anthropic | Safe, long context, great reasoning |
| Gemini 1.5 Pro | Google | Multimodal, huge context window |
| LLaMA 3 | Meta | Open source, run locally |
| Mistral | Mistral AI | Fast, efficient, open source |

---

## How Does an LLM Work? (Simplified)

### Step 1: Tokenization
LLMs don't read words — they read **tokens** (chunks of characters).

```
"Hello world" → ["Hello", " world"]  = 2 tokens
"Unbelievable" → ["Un", "bel", "iev", "able"] = 4 tokens
```

> 💡 Rule of thumb: 1 token ≈ 0.75 words. GPT-4 can handle ~128,000 tokens at once.

---

### Step 2: Training
The model is trained to **predict the next token** given all previous tokens.

```
Input:  "The cat sat on the"
Predict: "mat" (or "floor", "chair", etc.)
```

This is done billions of times on massive text datasets until the model gets really good at it.

---

### Step 3: The Transformer Architecture
LLMs use a **Transformer** — a neural network architecture with a key mechanism called **attention**.

Attention lets the model focus on relevant parts of the input:

```
"The bank can guarantee deposits will eventually cover 
 future tuition costs because it has a good reserve."

When predicting what "bank" means → attention focuses on:
"deposits", "reserve" → financial bank ✅ (not river bank)
```

---

## Key Concepts You Must Know

### 🌡️ Temperature
Controls how creative/random the output is.

```
Temperature = 0.0  → Very deterministic, same answer every time
Temperature = 0.7  → Balanced (recommended for most use cases)
Temperature = 1.5  → Very creative/random, can go off-track
```

### 📏 Context Window
The maximum amount of text an LLM can "see" at once (input + output).

```
GPT-3.5      →  16,000 tokens  (~12,000 words)
GPT-4o       → 128,000 tokens  (~96,000 words)
Claude 3.5   → 200,000 tokens  (~150,000 words)
Gemini 1.5   → 1,000,000 tokens (!)
```

### 🎭 Hallucination
When an LLM confidently states something **false**.

```
❌ "The Eiffel Tower was built in 1756."  ← Wrong (it's 1889), said confidently
```

**Why it happens:** LLMs generate plausible-sounding text, not verified facts.  
**How to handle it:** Always verify important facts, use RAG for grounding.

### 🔢 Parameters
The "size" of a model. More parameters = more capacity to learn.

```
GPT-2        →   1.5 Billion parameters
LLaMA 3 8B   →   8 Billion parameters
GPT-4        → ~1.8 Trillion parameters (estimated)
```

---

## LLM Families Explained

### Closed-Source (API Only)
You access these via API — you can't download or modify them.
- OpenAI (GPT-4, GPT-4o)
- Anthropic (Claude 3.5)
- Google (Gemini)

**Pros:** State-of-the-art quality, no hardware needed  
**Cons:** Costs money, data sent to third party

### Open-Source (Run Locally)
You can download and run these on your own machine.
- Meta LLaMA 3
- Mistral / Mixtral
- Google Gemma

**Pros:** Free, private, customizable  
**Cons:** Needs GPU, slightly lower quality than top closed models

---

## What Can LLMs Do?

| Task | Example |
|---|---|
| **Text Generation** | Write articles, emails, stories |
| **Summarization** | TL;DR of long documents |
| **Translation** | English → French |
| **Code Generation** | Write Python functions |
| **Q&A** | Answer questions about your data |
| **Classification** | Spam detection, sentiment analysis |
| **Extraction** | Pull names/dates from documents |
| **Reasoning** | Solve math problems, logic puzzles |

---

## What LLMs Can NOT Do (Yet)

- ❌ Browse the internet in real-time (unless given tools)
- ❌ Remember past conversations (unless you store them)
- ❌ Perform reliable arithmetic on large numbers
- ❌ Access your files or databases (unless connected via RAG/tools)
- ❌ Guarantee factual accuracy

---

## Summary

| Concept | One-liner |
|---|---|
| LLM | AI trained to predict/generate text |
| Token | Chunk of text the model processes |
| Temperature | Controls randomness of output |
| Context Window | Max text the model can see at once |
| Hallucination | Model confidently states false info |
| Parameters | Model size/capacity |

---

## ✅ What's Next?

Now that you understand what LLMs are, let's learn how to **talk to them effectively**.

👉 **[Module 2 — Prompt Engineering](../02-prompt-engineering/README.md)**
