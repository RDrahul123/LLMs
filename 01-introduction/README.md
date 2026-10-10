# Module 1: Introduction to Large Language Models

> **Level:** Beginner | **Time:** ~1 hour | **Prerequisites:** None | **Code Required:** No

---

## What You'll Learn

- What an LLM is and how it works at a high level
- How LLMs are trained (tokenization, next-token prediction)
- The Transformer architecture at a glance
- Key concepts every practitioner must know
- What LLMs can and cannot do

![LLM lifecycle from model training to a validated application response](../assets/diagrams/llm-lifecycle.svg)

---

## What is an LLM?

A **Large Language Model (LLM)** is a neural network trained on vast amounts of text data to understand and generate human language. At its core, an LLM is a function that takes a sequence of tokens (words or sub-words) and predicts the next token in the sequence.

```
Input:  "The capital of France is"
Output: "Paris"
```

LLMs are "large" both in terms of **parameters** (billions of weights) and **training data** (trillions of tokens from books, websites, code, and more).

---

## How Training Works

### Step 1: Tokenization

Before an LLM can process text, it must convert raw text into numbers. This is done through **tokenization** — breaking text into tokens.

```
"Hello, world!" → ["Hello", ",", " world", "!"]
"unbelievable"  → ["un", "believ", "able"]
```

Modern tokenizers (BPE, SentencePiece) learn the most efficient way to split text during training.

### Step 2: Next-Token Prediction

The core training objective is deceptively simple: **predict the next token**.

```
Given:  "The cat sat on the"
Target: "mat"
```

The model sees billions of such examples and learns patterns — grammar, facts, reasoning, even coding.

### Step 3: Self-Supervised Learning

No human labels the data. The model learns by predicting the next word in sentences it has never seen before. This is why internet text is so valuable — it's essentially free supervision.

### Step 4: Fine-Tuning & Alignment

After pre-training, models are further refined with:
- **Supervised Fine-Tuning (SFT):** Trained on instruction-response pairs
- **RLHF / DPO:** Aligned to human preferences using reinforcement learning

---

## Transformer Architecture Overview

The **Transformer** (Vaswani et al., 2017) is the architecture behind all modern LLMs. Here's the flow:

```
┌─────────────────────────────────────────────┐
│              Input Text                      │
│         "The cat sat on"                     │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│          Tokenization & Embedding           │
│    Convert tokens → dense vectors            │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│        Positional Encoding                   │
│    Add position information to embeddings    │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│       Transformer Block (×N layers)          │
│  ┌─────────────────────────────────────┐    │
│  │       Multi-Head Self-Attention      │    │
│  │   Each token attends to all others   │    │
│  └──────────────────┬──────────────────┘    │
│                     ▼                        │
│  ┌─────────────────────────────────────┐    │
│  │       Feed-Forward Network           │    │
│  │   Transform each token independently │    │
│  └──────────────────┬──────────────────┘    │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│          Output Probabilities               │
│    Next token: "mat" (p=0.87)               │
│                "floor" (p=0.04)             │
│                "chair" (p=0.02)             │
└─────────────────────────────────────────────┘
```

---

## Famous Models

| Model family / example | Creator | Access pattern | Notes |
|-----------------------|---------|----------------|-------|
| GPT family | OpenAI | Hosted API | General-purpose models; capabilities and model IDs vary by release |
| Claude family | Anthropic | Hosted API | General-purpose and multimodal models; consult current context and feature documentation |
| Gemini family | Google | Hosted API | Multimodal model family; available capabilities depend on model/version |
| Llama family | Meta and ecosystem | Open-weight releases | Can be run or adapted under the license terms for the specific release |
| Mistral family | Mistral AI | Hosted API and some open-weight releases | Check the specific model card and license |

Model names, context limits, pricing, and access change frequently. Parameter counts for proprietary models are often undisclosed; do not infer capability from an unverified estimate. Check the provider's current documentation before selecting a model.

---

## Key Concepts

### Temperature

Controls the randomness of outputs.

| Temperature | Behavior |
|-------------|----------|
| 0.0 | Deterministic, always picks the most likely token |
| 0.7 | Balanced — good for most tasks |
| 1.0 | More creative, higher diversity |
| 1.5+ | Very random, may produce incoherent output |

```
Prompt: "The sky is"
Temp 0:   "blue"       (always)
Temp 0.7: "blue"       (usually)
Temp 1.2: "beautiful"  (sometimes)
```

### Context Window

The maximum number of tokens a particular model/version can process at once. The limit usually includes the serialized instructions, history, user input, tool results, and generated output. Limits change by model and API; check the current model documentation and leave room for the response.

### Hallucinations

When an LLM generates plausible-sounding but **factually incorrect** information. This happens because the model optimizes for fluency, not truthfulness.

```
Q: "What year did Lincoln give the Gettysburg Address?"
A: "Lincoln delivered the Gettysburg Address in 1853."  ← HALLUCINATION
   (Correct answer: 1863)
```

### Parameters

The weights of the neural network. More parameters generally means more capacity to store knowledge and reason, but also more compute required.

- **1B parameters** = about 4 GB for weights at FP32, before runtime overhead
- **70B parameters** = about 280 GB for weights at FP32, before runtime overhead

At FP16/BF16, raw weight storage is roughly half that amount. Actual inference also needs KV-cache and runtime memory; quantization can reduce weight storage with quality/performance trade-offs.

---

## Open vs Closed Source

| Aspect | Open-weight model | Hosted proprietary model |
|--------|------------|---------------|
| **Examples** | Some Llama, Mistral, and Phi releases | Hosted offerings from OpenAI, Anthropic, and Google |
| **Access** | Download/use under the specific model license; local or hosted | Commonly accessed through a provider API |
| **Customization** | Can offer more deployment/control options, subject to license and hardware | Usually configured through APIs; some providers offer additional adaptation features |
| **Cost** | Compute, storage, and operations | Provider usage charges and service costs |
| **Privacy** | Can be run in a controlled environment; deployment determines data flow | Requests are processed under provider terms and settings |
| **Quality** | Varies by model and task | Varies by model and task |

“Open weights” does not necessarily mean open-source under every definition. Review the license, acceptable-use terms, and data handling for the exact model and hosting setup.

---

## What LLMs CAN Do

- Answer questions and explain concepts
- Write and debug code
- Summarize long documents
- Translate between languages
- Creative writing (stories, poems, scripts)
- Analyze data and generate insights
- Follow multi-step instructions
- Engage in multi-turn conversation

## What LLMs CANNOT Do

- Access real-time information (unless given tools)
- Perform precise mathematical calculations reliably
- Guarantee factual accuracy (hallucinations)
- Remember past conversations (stateless by default)
- Learn from conversations without retraining
- Replace domain-specific tools (databases, calculators, etc.)
- Truly "understand" — they're pattern matchers

---

## What's Next?

Now that you understand what LLMs are, it's time to peek under the hood.

**→ [Module 2: Transformer Architecture](../02-transformer-architecture/README.md)**

Learn how self-attention works, build a mini Transformer from scratch, and understand why this architecture revolutionized AI.

---

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — The original Transformer paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar's visual guide
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) — Build GPT from scratch
- [Lilian Weng's Blog](https://lilianweng.github.io/) — Excellent AI research summaries

## Visual Deep Dive

For the expanded walkthrough of training stages, token budgets, decoding, parameter memory, diagrams, and knowledge checks, continue to **[Module 1 Deep Dive](deep-dive.md)**.
