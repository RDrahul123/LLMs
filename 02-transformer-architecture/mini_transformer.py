"""
Module 2: Mini GPT — Transformer from Scratch
================================================
A tiny character-level GPT that you can train on any text.
Based on Andrej Karpathy's nanoGPT (simplified for learning).

Usage:
    python mini_transformer.py --train         # train on shakespeare.txt
    python mini_transformer.py --generate      # generate text after training

Requirements:
    pip install torch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────

BATCH_SIZE    = 32      # sequences per training step
BLOCK_SIZE    = 64      # max context length (tokens)
MAX_ITERS     = 3000    # training steps
EVAL_INTERVAL = 300     # how often to evaluate
LR            = 3e-4    # learning rate
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
N_EMBED       = 64      # embedding dimension
N_HEADS       = 4       # attention heads
N_LAYERS      = 4       # transformer blocks
DROPOUT       = 0.1


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────

SAMPLE_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heartache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream.
"""


def prepare_data(text: str):
    """Encode text as character-level tokens."""
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, vocab_size, encode, decode


def get_batch(data, batch_size=BATCH_SIZE, block_size=BLOCK_SIZE):
    """Sample a random batch of sequences."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ─────────────────────────────────────────────
# Model Components
# ─────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, n_embed, n_heads, block_size, dropout):
        super().__init__()
        assert n_embed % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads

        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal mask: upper triangle = -inf (can't attend to future)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape

        # Compute Q, K, V all at once
        q, k, v = self.qkv(x).split(C, dim=2)

        # Reshape for multi-head: (B, heads, T, head_dim)
        def reshape(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale          # (B, heads, T, T)
        attn = attn.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum of values
        out = attn @ v                                      # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One transformer block: attention + feed-forward with residuals."""

    def __init__(self, n_embed, n_heads, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffn = FeedForward(n_embed, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual connection
        x = x + self.ffn(self.ln2(x))    # residual connection
        return x


class MiniGPT(nn.Module):
    """A tiny GPT-style language model."""

    def __init__(self, vocab_size, n_embed=N_EMBED, n_heads=N_HEADS,
                 n_layers=N_LAYERS, block_size=BLOCK_SIZE, dropout=DROPOUT):
        super().__init__()
        self.block_size = block_size

        self.transformer = nn.ModuleDict({
            "token_emb": nn.Embedding(vocab_size, n_embed),
            "pos_emb":   nn.Embedding(block_size, n_embed),
            "drop":      nn.Dropout(dropout),
            "blocks":    nn.ModuleList([
                TransformerBlock(n_embed, n_heads, block_size, dropout)
                for _ in range(n_layers)
            ]),
            "ln_f":      nn.LayerNorm(n_embed),
        })
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"🤖 MiniGPT initialized: {n_params:,} parameters")

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size

        # Token + positional embeddings
        tok_emb = self.transformer["token_emb"](idx)
        pos = torch.arange(T, device=DEVICE)
        pos_emb = self.transformer["pos_emb"](pos)
        x = self.transformer["drop"](tok_emb + pos_emb)

        # Pass through transformer blocks
        for block in self.transformer["blocks"]:
            x = block(x)

        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = 50):
        """Generate tokens autoregressively."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-K sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters=50):
    model.eval()
    losses = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        split_losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data)
            _, loss = model(x, y)
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)
    model.train()
    return losses


def train(text: str = SAMPLE_TEXT):
    print(f"🖥️  Device: {DEVICE}")
    print(f"📚 Text length: {len(text):,} characters\n")

    train_data, val_data, vocab_size, encode, decode = prepare_data(text)
    print(f"📖 Vocabulary size: {vocab_size} unique characters")

    model = MiniGPT(vocab_size=vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print(f"\n🚀 Training for {MAX_ITERS} steps...\n")

    for step in range(MAX_ITERS):
        # Evaluate periodically
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(f"Step {step:4d} | train loss: {losses['train']:.3f} | val loss: {losses['val']:.3f}")

        x, y = get_batch(train_data)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"\n✅ Training complete!")

    # Save model and metadata
    torch.save({
        "model_state": model.state_dict(),
        "vocab_size": vocab_size,
        "encode": encode,
        "decode": decode,
    }, "mini_gpt.pt")
    print("💾 Model saved to mini_gpt.pt")

    # Generate a sample
    print("\n📝 Generated sample:")
    start = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    generated_ids = model.generate(start, max_new_tokens=200, temperature=0.8)
    print(decode(generated_ids[0].tolist()))


def generate_text(prompt: str = "\n", max_tokens: int = 300, temperature: float = 0.8):
    checkpoint = torch.load("mini_gpt.pt", map_location=DEVICE)
    vocab_size = checkpoint["vocab_size"]
    encode = checkpoint["encode"]
    decode = checkpoint["decode"]

    model = MiniGPT(vocab_size=vocab_size).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
    generated = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature)
    print(decode(generated[0].tolist()))


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini GPT Transformer")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--generate", action="store_true", help="Generate text from saved model")
    parser.add_argument("--prompt", default="\n", help="Prompt for generation")
    parser.add_argument("--tokens", type=int, default=300, help="Tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature (0.1=focused, 1.5=creative)")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.generate:
        generate_text(prompt=args.prompt, max_tokens=args.tokens, temperature=args.temp)
    else:
        print("Usage:")
        print("  python mini_transformer.py --train")
        print("  python mini_transformer.py --generate --prompt 'To be'")
