"""
Module 2: Attention Weight Visualization
==========================================
Visualizes how attention flows between tokens using a HuggingFace model.

Usage:
    python attention_demo.py
    python attention_demo.py --text "The cat sat on the mat"

Requirements:
    pip install transformers torch bertviz
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModel


def get_attention_weights(text: str, model_name: str = "bert-base-uncased"):
    """
    Extract attention weights from a transformer model.
    Returns tokens and attention matrices for all layers/heads.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.attentions: tuple of (num_layers,) each [batch, heads, seq, seq]
    return tokens, outputs.attentions


def print_attention_heatmap(tokens: list, attention_layer, head: int = 0):
    """Print a text-based attention heatmap for a specific head."""
    attn = attention_layer[0, head].numpy()  # [seq_len, seq_len]
    seq_len = len(tokens)

    # Header row
    col_width = 8
    header = " " * 12 + "".join(t[:col_width].ljust(col_width) for t in tokens)
    print(header)
    print("-" * len(header))

    for i, row_token in enumerate(tokens):
        row_label = row_token[:10].ljust(12)
        row_values = ""
        for j in range(seq_len):
            score = attn[i, j]
            # Use block characters to show intensity
            if score > 0.5:
                block = "█"
            elif score > 0.3:
                block = "▓"
            elif score > 0.15:
                block = "▒"
            elif score > 0.05:
                block = "░"
            else:
                block = "·"
            row_values += (block + f"{score:.2f}").ljust(col_width)
        print(row_label + row_values)


def print_top_attentions(tokens: list, attention_layer, head: int = 0, top_k: int = 5):
    """Print the top-K attention pairs for a given head."""
    attn = attention_layer[0, head].numpy()
    seq_len = len(tokens)

    pairs = []
    for i in range(seq_len):
        for j in range(seq_len):
            if i != j:  # skip self-attention
                pairs.append((tokens[i], tokens[j], attn[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"\nTop {top_k} attention pairs (Head {head}):")
    print(f"  {'FROM':<12} → {'TO':<12}  WEIGHT")
    print(f"  {'-'*12}   {'-'*12}  ------")
    for from_tok, to_tok, weight in pairs[:top_k]:
        bar = "█" * int(weight * 20)
        print(f"  {from_tok:<12} → {to_tok:<12}  {weight:.3f} {bar}")


def analyze_text(text: str):
    """Run full attention analysis on input text."""
    print(f"\n📝 Text: '{text}'")
    print("=" * 60)
    print("Loading BERT model (first run downloads ~400MB)...")

    tokens, attentions = get_attention_weights(text)

    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]

    print(f"\n✅ Model loaded!")
    print(f"   Tokens ({len(tokens)}): {tokens}")
    print(f"   Layers: {num_layers}")
    print(f"   Heads per layer: {num_heads}")

    # Show attention for layer 0, head 0
    print(f"\n{'='*60}")
    print(f"ATTENTION HEATMAP — Layer 1, Head 1")
    print(f"{'='*60}")
    print_attention_heatmap(tokens, attentions[0], head=0)

    # Show top attention pairs across a few heads
    print(f"\n{'='*60}")
    print(f"TOP ATTENTION PAIRS — Layer 1")
    print(f"{'='*60}")
    for head in range(min(3, num_heads)):
        print_top_attentions(tokens, attentions[0], head=head, top_k=3)

    # Show how attention evolves across layers
    print(f"\n{'='*60}")
    print("ATTENTION COMPLEXITY ACROSS LAYERS")
    print("(entropy — higher = more distributed attention)")
    print(f"{'='*60}")
    import numpy as np
    for layer_idx in range(num_layers):
        attn = attentions[layer_idx][0, 0].numpy()
        # Compute entropy of attention distribution
        entropy = -np.sum(attn * np.log(attn + 1e-9), axis=-1).mean()
        bar = "█" * int(entropy * 5)
        print(f"  Layer {layer_idx + 1:2d}: {bar} ({entropy:.2f})")

    print("\n💡 Tip: Use BertViz for interactive attention visualization:")
    print("   from bertviz import head_view")
    print("   head_view(attentions, tokens)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize transformer attention weights")
    parser.add_argument(
        "--text",
        default="The trophy didn't fit in the suitcase because it was too big.",
        help="Text to analyze"
    )
    args = parser.parse_args()

    analyze_text(args.text)
