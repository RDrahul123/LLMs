# Module 2 — Transformer Architecture 🟡

> **Level:** Beginner → Intermediate | **Time:** ~2 hours | **Code:** Python (optional)

---

## Why Learn This?

Every modern LLM — GPT-4, Claude, Gemini, LLaMA — is built on the **Transformer** architecture introduced in the 2017 paper *"Attention Is All You Need"*.

Understanding it helps you:
- Intuitively debug why LLMs fail
- Make better decisions about context windows, chunking, and fine-tuning
- Read research papers without getting lost

You don't need to build one from scratch. You need to understand how the pieces fit.

---

## The Big Picture

Before the Transformer, models processed text **sequentially** — one word at a time (RNNs, LSTMs). This was slow and struggled with long-range dependencies.

The Transformer processes **all tokens in parallel** and uses **attention** to relate any token to any other token — regardless of distance.

```
Input Text
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                    TRANSFORMER                      │
│                                                     │
│  ┌─────────────┐           ┌─────────────────────┐  │
│  │   ENCODER   │           │       DECODER       │  │
│  │             │  Context  │                     │  │
│  │ Understands │ ────────► │ Generates output    │  │
│  │ the input   │           │ token by token      │  │
│  └─────────────┘           └─────────────────────┘  │
└─────────────────────────────────────────────────────┘
    │
    ▼
Output Text
```

> 💡 LLMs like GPT are **decoder-only** transformers. BERT is **encoder-only**. The original translation model used both.

---

## Step 1: Tokenization & Embeddings

### Tokenization
Text is split into **tokens** before the model sees it.

```
"Hello, world!" → ["Hello", ",", " world", "!"]
Token IDs      → [15496, 11, 995, 0]
```

Each token gets a unique integer ID from the model's **vocabulary** (~50,000–100,000 tokens).

### Token Embeddings
Each token ID is mapped to a **dense vector** (a list of floats) that captures meaning.

```
"king"   → [0.82, -0.31, 0.14, 0.95, ...]   # 768 or 4096 floats
"queen"  → [0.79, -0.28, 0.11, 0.91, ...]   # very similar!
"banana" → [-0.12, 0.67, -0.44, 0.03, ...]  # very different
```

These vectors are **learned during training** — similar meaning = similar vector direction.

### Positional Encoding
Since the Transformer processes tokens in parallel (not sequentially), it needs to know **where each token is** in the sequence.

Positional encodings are added to embeddings:

```
final_input = token_embedding + positional_encoding
```

Original transformers used sine/cosine functions. Modern LLMs use **Rotary Position Embeddings (RoPE)** — which also handle very long sequences better.

---

## Step 2: The Attention Mechanism ⭐

This is the heart of the Transformer. Attention lets every token **look at** and **gather information from** every other token.

### The Intuition

Consider this sentence:
```
"The trophy didn't fit in the suitcase because it was too big."
```

What does **"it"** refer to? The trophy or the suitcase?

Humans resolve this by attending to context clues ("too big" → the thing that doesn't fit → the trophy). Attention does the same mathematically.

---

### Queries, Keys, and Values (Q, K, V)

For each token, the model creates three vectors:

| Vector | Role | Analogy |
|---|---|---|
| **Query (Q)** | "What am I looking for?" | A search query |
| **Key (K)** | "What do I contain?" | A document tag/label |
| **Value (V)** | "What information do I carry?" | The document content |

```python
# For each token embedding x:
Q = x @ W_Q   # project to query space
K = x @ W_K   # project to key space
V = x @ W_V   # project to value space
```

### Attention Score Calculation

```
                         Q · Kᵀ
Attention(Q, K, V) = softmax(───────) · V
                           √d_k
```

Breaking it down:

**1. Compute similarity scores (Q · Kᵀ)**
```
Each query token "asks": how relevant is each key token to me?
Result: a score matrix (seq_len × seq_len)

         "The"  "cat"  "sat"  "on"
"The"  [  8.1,   2.3,   0.1,  0.4 ]
"cat"  [  1.2,   9.4,   1.1,  0.2 ]
"sat"  [  0.3,   7.2,   8.8,  1.5 ]
"on"   [  0.1,   0.4,   2.1,  7.6 ]
```

**2. Scale by √d_k**
Prevents scores from getting too large (which would make softmax outputs very sharp/saturated).

**3. Apply softmax**
Converts raw scores → probabilities that sum to 1.

```
"cat" attends to:
  "The" → 0.08
  "cat" → 0.84   ← mostly attends to itself
  "sat" → 0.06
  "on"  → 0.02
```

**4. Weighted sum of Values**
```
output for "cat" = 0.08 × V("The") + 0.84 × V("cat") + 0.06 × V("sat") + ...
```

The result is a new, context-enriched representation of each token.

---

### Multi-Head Attention

Instead of one attention operation, the Transformer runs **multiple attention "heads" in parallel**, each learning different relationship types.

```
┌─────────────────────────────────────────────┐
│              MULTI-HEAD ATTENTION           │
│                                             │
│  Head 1: subject-verb relationships         │
│  Head 2: pronoun-antecedent resolution      │
│  Head 3: adjective-noun agreement           │
│  Head 4: long-range dependencies            │
│  ...                                        │
│  Head N: (learned automatically)            │
│                                             │
│  → Concatenate all heads → Linear layer     │
└─────────────────────────────────────────────┘
```

```python
# Simplified multi-head attention
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, channels

        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        attended = weights @ V

        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(attended)
```

---

## Step 3: Feed-Forward Network

After attention, each token passes through a **small neural network independently**:

```
FFN(x) = ReLU(x · W₁ + b₁) · W₂ + b₂
```

- Typical size: 4× the model dimension
  - GPT-2 (768 dim) → FFN has 3072 neurons
  - LLaMA 3 (4096 dim) → FFN has 16384 neurons

This is where **most of the model's factual knowledge** is stored. (Attention figures out *which* information to retrieve; FFN stores the actual knowledge.)

---

## Step 4: Layer Normalization & Residual Connections

### Residual Connections
Every sub-layer adds its input back to its output:

```
output = LayerNorm(x + Attention(x))
output = LayerNorm(x + FFN(x))
```

This solves the **vanishing gradient problem** and allows very deep networks to train.

### Layer Normalization
Normalizes activations to have mean=0, std=1 within each token's representation. Keeps training stable.

---

## Putting It All Together: One Transformer Block

```
Input Tokens
     │
     ▼
[Token Embeddings + Positional Encoding]
     │
     ▼ ┌──────────────────────────────────┐
     │  │       TRANSFORMER BLOCK         │  × N layers
     │  │                                 │
     │  │  ┌─────────────────────────┐    │
     │  │  │   Multi-Head Attention  │    │
     │  │  └──────────┬──────────────┘    │
     │  │             │ + residual        │
     │  │  ┌──────────▼──────────────┐    │
     │  │  │     Layer Norm          │    │
     │  │  └──────────┬──────────────┘    │
     │  │             │                   │
     │  │  ┌──────────▼──────────────┐    │
     │  │  │   Feed-Forward Network  │    │
     │  │  └──────────┬──────────────┘    │
     │  │             │ + residual        │
     │  │  ┌──────────▼──────────────┐    │
     │  │  │     Layer Norm          │    │
     │  │  └─────────────────────────┘    │
     │  └──────────────────────────────────┘
     │
     ▼
[Output Layer → Vocabulary Logits]
     │
     ▼
Next Token Prediction
```

---

## Model Size = Depth × Width

| Model | Layers (N) | Dimension (d) | Heads | Parameters |
|---|---|---|---|---|
| GPT-2 Small | 12 | 768 | 12 | 117M |
| GPT-2 XL | 48 | 1600 | 25 | 1.5B |
| LLaMA 3 8B | 32 | 4096 | 32 | 8B |
| LLaMA 3 70B | 80 | 8192 | 64 | 70B |
| GPT-4 (est.) | ~120 | ~12288 | ~96 | ~1.8T |

---

## Causal Masking (Decoder-Only Models)

LLMs like GPT use **causal masking** — when predicting token N, they can only attend to tokens 1 through N-1. Future tokens are masked.

```
         "The"  "cat"  "sat"
"The"  [  ✅,    ❌,    ❌  ]   ← can only see itself
"cat"  [  ✅,    ✅,    ❌  ]   ← can see "The" and itself
"sat"  [  ✅,    ✅,    ✅  ]   ← can see all previous
```

This is implemented by adding `-inf` to masked positions before softmax (so they become 0 after softmax).

---

## Key Innovations in Modern LLMs

| Innovation | What It Solves | Used In |
|---|---|---|
| **RoPE** (Rotary Position Embeddings) | Better long-context handling | LLaMA, Mistral |
| **GQA** (Grouped Query Attention) | Faster inference, less memory | LLaMA 3, Gemma |
| **SwiGLU** (activation function) | Better FFN performance | LLaMA, PaLM |
| **Flash Attention** | Memory-efficient attention | Most modern LLMs |
| **KV Cache** | Faster generation (reuse past keys/values) | All inference |

---

## Summary

| Component | What It Does |
|---|---|
| **Tokenization** | Converts text → integer IDs |
| **Embeddings** | Maps token IDs → semantic vectors |
| **Positional Encoding** | Tells model where each token is |
| **Multi-Head Attention** | Each token gathers context from all others |
| **Feed-Forward Network** | Stores knowledge, transforms representations |
| **Residual + LayerNorm** | Enables stable deep training |
| **Causal Mask** | Prevents peeking at future tokens |
| **N stacked blocks** | Progressively refines representations |

---

## 📁 Files in This Module

| File | Description |
|---|---|
| [`attention_demo.py`](./attention_demo.py) | Visualize attention weights on real text |
| [`mini_transformer.py`](./mini_transformer.py) | A tiny GPT built from scratch (~150 lines) |

---

## 📖 Further Reading

- [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) — the original 2017 paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — best visual explainer ever made
- [Andrej Karpathy — Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) — build one from scratch in video form

---

## ✅ What's Next?

👉 **[Module 3 — Prompt Engineering](../03-prompt-engineering/README.md)**
