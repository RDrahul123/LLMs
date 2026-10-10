"""
Attention Visualization Demo
============================
Uses BERT to visualize attention weights across layers and heads.

Install: pip install transformers torch bertviz matplotlib

Usage:
    python attention_demo.py
    python attention_demo.py --text "The cat sat on the mat because it was tired"
"""

import argparse
import math
import sys

import torch
from transformers import BertTokenizer, BertModel


def get_attention_weights(text: str, model_name: str = "bert-base-uncased"):
    """Extract attention weights from BERT for the given text."""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_attentions=True)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs)

    attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)
    return attentions, tokens


def print_attention_heatmap(attention, tokens, layer=None, head=None):
    """Print a text-based heatmap of attention weights using block characters."""
    blocks = " ░▒▓█"

    if layer is not None and head is not None:
        attn = attention[layer][0, head].numpy()
        print(f"\n--- Layer {layer}, Head {head} ---")
        _print_matrix(attn, tokens, blocks)
    else:
        for l in range(len(attention)):
            for h in range(attention[l].shape[1]):
                attn = attention[l][0, h].numpy()
                print(f"\n--- Layer {l}, Head {h} ---")
                _print_matrix(attn, tokens, blocks)


def _print_matrix(attn, tokens, blocks):
    """Print a single attention matrix as a text heatmap."""
    n = len(tokens)
    max_token_len = max(len(t) for t in tokens)

    # Header
    header = " " * (max_token_len + 1)
    for j in range(n):
        header += f"{j:>3}"
    print(header)
    print(" " * (max_token_len + 1) + "-" * (n * 3))

    for i in range(n):
        row = f"{tokens[i]:>{max_token_len}} |"
        for j in range(n):
            val = attn[i][j]
            block_idx = min(int(val * (len(blocks) - 1)), len(blocks) - 1)
            row += f" {blocks[block_idx]} "
        print(row)


def print_top_attentions(attention, tokens, top_k: int = 5):
    """Print the strongest attention connections across all layers and heads."""
    connections = []

    for l in range(len(attention)):
        for h in range(attention[l].shape[1]):
            attn = attention[l][0, h].numpy()
            n = attn.shape[0]
            for i in range(n):
                for j in range(n):
                    if i != j:
                        connections.append((attn[i][j], l, h, tokens[i], tokens[j]))

    connections.sort(reverse=True)

    print(f"\n{'='*60}")
    print(f" TOP {top_k} ATTENTION CONNECTIONS")
    print(f"{'='*60}")
    print(f"{'Score':>6}  {'Layer':>5}  {'Head':>4}  {'From':<15} {'To':<15}")
    print(f"{'-'*60}")

    for score, layer, head, src, dst in connections[:top_k]:
        print(f"{score:6.4f}  {layer:5d}  {head:4d}  {src:<15} {dst:<15}")


def analyze_text(text: str, model_name: str = "bert-base-uncased"):
    """Full analysis: heatmap, top attentions, and entropy."""
    print(f"Analyzing: \"{text}\"\n")
    print("Loading BERT... (this may take a moment on first run)\n")

    attentions, tokens = get_attention_weights(text, model_name)

    # Show tokenization
    print(f"Tokens ({len(tokens)}): {tokens}\n")

    # Show selected attention heatmaps (layer 0 head 0, layer 6 head 3, last layer head 0)
    print_attention_heatmap(attentions, tokens, layer=0, head=0)
    if attentions[0].shape[1] > 3:
        print_attention_heatmap(attentions, tokens, layer=min(6, len(attentions) - 1), head=3)
    print_attention_heatmap(attentions, tokens, layer=len(attentions) - 1, head=0)

    # Top attention connections
    print_top_attentions(attentions, tokens, top_k=10)

    # Entropy analysis
    print_entropy_analysis(attentions, tokens)


def print_entropy_analysis(attention, tokens):
    """Compute and display attention entropy across layers."""
    print(f"\n{'='*60}")
    print(" ATTENTION ENTROPY BY LAYER")
    print(" (Low = focused, High = spread out)")
    print(f"{'='*60}\n")

    n_layers = len(attention)
    n_heads = attention[0].shape[1]

    for l in range(n_layers):
        entropies = []
        for h in range(n_heads):
            attn = attention[l][0, h].numpy()
            # Average entropy across all source tokens
            avg_entropy = 0.0
            n = attn.shape[0]
            for i in range(n):
                row = attn[i]
                # Add epsilon to avoid log(0)
                row = row + 1e-10
                avg_entropy += -sum(row[j] * math.log2(row[j]) for j in range(len(row)))
            avg_entropy /= n
            entropies.append(avg_entropy)

        bar_width = 40
        avg_e = sum(entropies) / len(entropies)
        filled = int(avg_e / math.log2(len(tokens)) * bar_width) if len(tokens) > 1 else 0
        filled = min(filled, bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        print(f"  Layer {l:2d}: [{bar}] avg={avg_e:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize BERT attention weights")
    parser.add_argument(
        "--text",
        type=str,
        default="The cat sat on the mat because it was tired",
        help="Text to analyze",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace model name",
    )
    args = parser.parse_args()

    try:
        analyze_text(args.text, args.model)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        print("Make sure you have: pip install transformers torch", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
