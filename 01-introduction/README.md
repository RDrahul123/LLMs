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

LLMs are built on a neural network architecture called the **Transformer**, introduced by Google in the 2017 paper *"Attention Is All You Need"*. Every major LLM today — GPT, Claude, Gemini, LLaMA — is a Transformer.

Here's the full picture of what happens inside:

```
YOUR PROMPT (raw text)
        │
        ▼
┌───────────────────┐
│   TOKENIZATION    │  "Hello world" → [15339, 1917]
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│    EMBEDDINGS     │  Each token → a vector of numbers
│                   │  [15339] → [0.24, -0.81, 0.55, ...]
└────────┬──────────┘
         │
         ▼
┌─────────────────────────────────────┐
│         TRANSFORMER BLOCKS          │  (repeated N times)
│                                     │
│  ┌─────────────────────────────┐    │
│  │   MULTI-HEAD ATTENTION      │    │
│  │  "what should I focus on?"  │    │
│  └──────────────┬──────────────┘    │
│                 │                   │
│  ┌──────────────▼──────────────┐    │
│  │   FEED-FORWARD NETWORK      │    │
│  │  "process what I focused on"│    │
│  └─────────────────────────────┘    │
│                                     │
└────────┬────────────────────────────┘
         │
         ▼
┌───────────────────┐
│   OUTPUT HEAD     │  Scores for every possible next token
└────────┬──────────┘
         │
         ▼
    NEXT TOKEN
  ("mat", "chair", ...)
```

---

## 🔍 The Transformer — Piece by Piece

### 1. Embeddings — Turning Words into Numbers

Before any computation, each token is converted into a **vector** — a list of numbers that captures its meaning.

```
"king"   → [ 0.81,  0.45, -0.23,  0.67, ...]
"queen"  → [ 0.79,  0.42, -0.20,  0.69, ...]  ← very similar!
"banana" → [-0.12, -0.55,  0.91, -0.30, ...]  ← very different
```

Words with similar meanings end up with similar vectors. This is how the model understands that "dog" and "puppy" are related, even though they look nothing alike as text.

---

### 2. Positional Encoding — Knowing Word Order

Transformers process all tokens **simultaneously** (not one by one), so they need a way to know word order. **Positional encoding** adds a position signal to each embedding:

```
Token 1: "The"   → embedding + position_1_signal
Token 2: "cat"   → embedding + position_2_signal
Token 3: "sat"   → embedding + position_3_signal
```

Without this, "dog bites man" and "man bites dog" would look identical to the model.

---

### 3. Self-Attention — The Core Mechanism ⭐

This is the heart of the Transformer. **Self-attention** lets every token look at every other token and decide: *"how much should I focus on you to understand my own meaning?"*

```
Sentence: "The animal didn't cross the street because it was too tired."

What does "it" refer to?
                                         ┌──────────────────┐
                                         │  Attention Scores │
                                         │  for "it"         │
Token        → Attention Weight          │                   │
─────────────────────────────            │  animal  ████ 0.7 │
"The"        → 0.01  (barely)            │  street  ██   0.2 │
"animal"     → 0.70  ← HIGH             │  tired   █    0.1 │
"didn't"     → 0.02                     └──────────────────┘
"cross"      → 0.03
"street"     → 0.20  ← some
"it"         → 0.04  (itself)
```

The model correctly learns that "it" = "animal" by attending strongly to it.

**How it works (conceptually):**

Every token creates three vectors:
```
Q (Query)  → "What am I looking for?"
K (Key)    → "What do I contain?"
V (Value)  → "What information do I pass along?"

Attention Score = Q · K  (dot product = how well they match)
Output = weighted sum of all V vectors
```

---

### 4. Multi-Head Attention — Multiple Perspectives

Instead of running attention once, the Transformer runs it **multiple times in parallel** (e.g., 12 or 32 "heads"), each learning to focus on different relationships:

```
Head 1  → focuses on grammatical structure ("subject → verb")
Head 2  → focuses on coreferences ("it" → "animal")
Head 3  → focuses on semantic similarity ("fast" ↔ "quick")
Head 4  → focuses on positional proximity
...
Head N  → ...

All heads combined → richer, fuller understanding
```

---

### 5. Feed-Forward Network — Processing the Attended Info

After attention, each token passes through a **feed-forward neural network** independently. This is where the model does most of its "reasoning" — transforming the attended context into a useful representation.

```
attended_token_vector
        │
        ▼
  Linear Layer (expand)
        │
        ▼
  ReLU Activation (non-linearity)
        │
        ▼
  Linear Layer (compress)
        │
        ▼
  processed_token_vector
```

---

### 6. Stacking Blocks — Depth = Understanding

A single attention + feed-forward block is called a **Transformer layer**. LLMs stack many of these:

```
GPT-2 Small  →  12 layers
GPT-3        →  96 layers
GPT-4        → ~120 layers (estimated)
LLaMA 3 70B  →  80 layers
```

Each layer builds on the previous one — early layers learn syntax, middle layers learn semantics, deep layers learn complex reasoning and world knowledge.

---

### 7. The Output — Predicting the Next Token

After all transformer layers, the final representation is passed to an **output head** that produces a probability score for every token in the vocabulary (~50,000 tokens for GPT):

```
Vocabulary scores after "The cat sat on the":

"mat"    → 18.4%  ← highest
"floor"  → 12.1%
"chair"  →  9.3%
"roof"   →  3.2%
"moon"   →  0.1%
...
```

The model picks the next token (influenced by temperature), appends it to the input, and repeats the whole process — one token at a time.

---

### Full Flow — End to End

```
Input: "Translate to French: Hello"
         │
         ▼
    Tokenize → [41819, 284, 4141, 25, 18435]
         │
         ▼
    Embed each token into vectors
         │
         ▼
    Add positional encodings
         │
         ▼
    ┌──────────────────────┐
    │  Transformer Block 1 │ ← learns basic patterns
    │  Transformer Block 2 │ ← learns grammar
    │  ...                 │
    │  Transformer Block N │ ← learns task intent
    └──────────────────────┘
         │
         ▼
    Output scores → "Bonjour" (highest probability)
         │
         ▼ (repeat with "Bonjour" appended)
    Output scores → "!" 
         │
         ▼
Final output: "Bonjour !"
```

---

### Why "Attention Is All You Need"?

Before Transformers (2017), models processed text **sequentially** — word by word, left to right (RNNs, LSTMs). This was slow and struggled with long-range dependencies.

Transformers process **all tokens simultaneously** and can attend to any token regardless of distance. This made them:
- ✅ Much faster to train (parallelizable on GPUs)
- ✅ Better at long-range dependencies
- ✅ Scalable to billions of parameters

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
