"""
Mini GPT — Build a GPT from Scratch
====================================
A minimal GPT implementation in ~150 lines of PyTorch.
Trains on Shakespeare text and generates new passages.

Install: pip install torch

Usage:
    python mini_transformer.py --train
    python mini_transformer.py --generate
    python mini_transformer.py --train --generate
"""

import argparse
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


SHAKESPEARE_TEXT = """The first thing we do, let's kill all the lawyers.
Now is the winter of our discontent made glorious summer by this sun of York.
All the world's a stage, and all the men and women merely players.
To be or not to be, that is the question.
Whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune.
Good night, good night! Parting is such sweet sorrow.
What's in a name? That which we call a rose by any other name would smell as sweet.
All that glitters is not gold; often have you heard that told.
The lady doth protest too much, methinks.
We are such stuff as dreams are made on, and our little life is rounded with a sleep.
If music be the food of love, play on.
Once more unto the breach, dear friends, once more.
Now is the time of our discontent.
This royal throne of kings, this sceptred isle.
This blessed plot, this earth, this realm, this England.
""" * 10


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, block_size=256):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, block_size) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.block_size :]
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def prepare_data(text, block_size=256):
    """Character-level tokenization."""
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, vocab_size, encode, decode


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


def train_model(model, train_data, val_data, device, block_size=256, max_iters=500, eval_interval=50):
    """Train the model."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("Training MiniGPT...\n")
    for step in range(max_iters):
        if step % eval_interval == 0:
            model.eval()
            losses = {"train": 0.0, "val": 0.0}
            for split in ["train", "val"]:
                data = train_data if split == "train" else val_data
                for _ in range(10):
                    x, y = get_batch(data, block_size, batch_size=32)
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y)
                    losses[split] += loss.item()
                losses[split] /= 10
            print(f"  step {step:4d} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")

            # Generate a sample
            if step % (eval_interval * 2) == 0:
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                generated = model.generate(context, max_new_tokens=100, temperature=0.8)
                print(f"  Sample: {decode(generated[0].tolist())[:120]}...\n")

        model.train()
        x, y = get_batch(train_data, block_size, batch_size=32)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Training complete!")


def generate_text(model, decode, device, prompt=None, max_tokens=300):
    """Generate text from the trained model."""
    model.eval()
    if prompt:
        encode = lambda s: [model.tok_emb.weight.device and 0]  # simplified
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    generated = model.generate(context, max_new_tokens=max_tokens, temperature=0.8, top_k=40)
    text = decode(generated[0].tolist())
    print(f"\n{'='*50}")
    print(" GENERATED TEXT")
    print(f"{'='*50}\n")
    print(text)
    print(f"\n{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Mini GPT from scratch")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--generate", action="store_true", help="Generate text")
    parser.add_argument("--iters", type=int, default=500, help="Training iterations")
    parser.add_argument("--save", type=str, default="mini_gpt.pt", help="Model save path")
    args = parser.parse_args()

    if not args.train and not args.generate:
        args.train = True
        args.generate = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    block_size = 256
    train_data, val_data, vocab_size, encode, decode = prepare_data(SHAKESPEARE_TEXT, block_size)
    print(f"Vocabulary size: {vocab_size} characters")

    model = MiniGPT(vocab_size, d_model=128, n_heads=4, n_layers=4, block_size=block_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    if args.train:
        train_model(model, train_data, val_data, device, block_size, max_iters=args.iters)
        torch.save(model.state_dict(), args.save)
        print(f"Model saved to {args.save}")

    if args.generate:
        if os.path.exists(args.save) and not args.train:
            model.load_state_dict(torch.load(args.save, map_location=device))
            print(f"Loaded model from {args.save}")

        generate_text(model, decode, device)


if __name__ == "__main__":
    main()
