"""
Embeddings visualization and analysis demo.

Generates embeddings for sample texts, computes cosine similarity,
performs dimensionality reduction with t-SNE, and demonstrates clustering.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai not installed. Run: pip install openai")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Error: numpy not installed. Run: pip install numpy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except ImportError:
    plt = None
    print("Warning: matplotlib not installed. Plotting disabled. Run: pip install matplotlib")

try:
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
except ImportError:
    KMeans = None
    TSNE = None
    print("Warning: scikit-learn not installed. Clustering/TSNE disabled. Run: pip install scikit-learn")


SAMPLE_TEXTS = [
    # Technology cluster
    "Python is a versatile programming language used for web development and data science.",
    "Machine learning algorithms can identify patterns in large datasets.",
    "Cloud computing provides scalable infrastructure for modern applications.",
    "Natural language processing enables computers to understand human language.",
    "Deep learning neural networks have revolutionized computer vision.",
    # Nature cluster
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "The Amazon rainforest is home to millions of species of plants and animals.",
    "Ocean currents play a crucial role in regulating global climate patterns.",
    "Migration patterns of birds change with seasonal temperature variations.",
    "Coral reefs are among the most biodiverse ecosystems on Earth.",
    # Food cluster
    "Sourdough bread requires a fermented starter culture for its distinctive flavor.",
    "Italian cuisine is known for its pasta, pizza, and regional specialties.",
    "Sushi preparation involves carefully selecting fresh fish and vinegared rice.",
    "French pastries require precise temperature control and quality ingredients.",
    "Thai food balances sweet, sour, salty, and spicy flavors in harmony.",
]


def get_embeddings(
    client: OpenAI,
    texts: list[str],
    model: str = "text-embedding-3-small",
) -> np.ndarray:
    """Generate embeddings for a list of texts.

    Args:
        client: OpenAI client instance.
        texts: List of strings to embed.
        model: Embedding model name.

    Returns:
        numpy array of shape (n_texts, embedding_dim).
    """
    print(f"Generating embeddings for {len(texts)} texts...")
    response = client.embeddings.create(model=model, input=texts)
    embeddings = np.array([item.embedding for item in response.data])
    print(f"Embedding dimension: {embeddings.shape[1]}")
    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity score between -1 and 1.
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.

    Args:
        embeddings: Array of shape (n, dim).

    Returns:
        Similarity matrix of shape (n, n).
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    return np.dot(normalized, normalized.T)


def show_similarity_analysis(texts: list[str], embeddings: np.ndarray) -> None:
    """Display similarity analysis between texts."""
    print("\n" + "=" * 70)
    print("SIMILARITY ANALYSIS")
    print("=" * 70)

    sim_matrix = compute_similarity_matrix(embeddings)

    # Find most similar pairs (excluding self)
    n = len(texts)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, sim_matrix[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    print("\nMost similar pairs:")
    print("-" * 70)
    for i, j, score in pairs[:5]:
        print(f"  {score:.4f} | {texts[i][:50]}...")
        print(f"         | {texts[j][:50]}...")
        print()

    print("\nMost different pairs:")
    print("-" * 70)
    for i, j, score in pairs[-5:]:
        print(f"  {score:.4f} | {texts[i][:50]}...")
        print(f"         | {texts[j][:50]}...")
        print()


def show_within_cluster_similarity(texts: list[str], embeddings: np.ndarray) -> None:
    """Show average similarity within predefined clusters."""
    print("\n" + "=" * 70)
    print("CLUSTER SIMILARITY")
    print("=" * 70)

    sim_matrix = compute_similarity_matrix(embeddings)

    clusters = {
        "Technology": list(range(0, 5)),
        "Nature": list(range(5, 10)),
        "Food": list(range(10, 15)),
    }

    for cluster_name, indices in clusters.items():
        similarities = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                similarities.append(sim_matrix[indices[i], indices[j]])
        avg_sim = np.mean(similarities) if similarities else 0
        print(f"\n{cluster_name} cluster:")
        print(f"  Average intra-cluster similarity: {avg_sim:.4f}")
        for idx in indices:
            print(f"    - {texts[idx][:60]}...")

    # Cross-cluster similarities
    print("\nCross-cluster similarities:")
    cluster_names = list(clusters.keys())
    for i in range(len(cluster_names)):
        for j in range(i + 1, len(cluster_names)):
            name_a, name_b = cluster_names[i], cluster_names[j]
            indices_a, indices_b = clusters[name_a], clusters[name_b]
            similarities = []
            for idx_a in indices_a:
                for idx_b in indices_b:
                    similarities.append(sim_matrix[idx_a, idx_b])
            avg_sim = np.mean(similarities) if similarities else 0
            print(f"  {name_a} <-> {name_b}: {avg_sim:.4f}")


def reduce_dimensions_tsne(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: float = 5.0,
) -> np.ndarray:
    """Reduce embedding dimensions using t-SNE.

    Args:
        embeddings: High-dimensional embeddings.
        n_components: Target dimensions (default 2 for plotting).
        perplexity: t-SNE perplexity parameter.

    Returns:
        Reduced-dimension array.
    """
    if TSNE is None:
        print("Error: scikit-learn not installed")
        return embeddings[:, :n_components]

    n_samples = embeddings.shape[0]
    actual_perplexity = min(perplexity, n_samples - 1)

    tsne = TSNE(
        n_components=n_components,
        perplexity=actual_perplexity,
        random_state=42,
        n_iter=1000,
    )
    reduced = tsne.fit_transform(embeddings)
    return reduced


def plot_embeddings(
    embeddings: np.ndarray,
    texts: list[str],
    labels: list[str],
    title: str = "Embedding Visualization",
    output_file: str = "embeddings_plot.png",
) -> None:
    """Plot 2D embedding visualization.

    Args:
        embeddings: 2D reduced embeddings.
        texts: Original text strings (used for annotations).
        labels: Cluster labels for coloring.
        title: Plot title.
        output_file: Output file path.
    """
    if plt is None:
        print("Error: matplotlib not installed")
        return

    unique_labels = sorted(set(labels))
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=[colors[i]],
            label=label,
            s=100,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

    # Add text annotations
    for i, text in enumerate(texts):
        short_text = text[:30] + "..." if len(text) > 30 else text
        ax.annotate(
            short_text,
            (embeddings[i, 0], embeddings[i, 1]),
            fontsize=7,
            alpha=0.8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_title(title, fontsize=14)
    ax.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to: {output_file}")


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform K-means clustering on embeddings.

    Args:
        embeddings: Embedding vectors.
        n_clusters: Number of clusters.

    Returns:
        Tuple of (labels, centroids).
    """
    if KMeans is None:
        print("Error: scikit-learn not installed")
        return np.zeros(len(embeddings), dtype=int), np.zeros((n_clusters, embeddings.shape[1]))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans.cluster_centers_


def run_demo() -> None:
    """Run the complete embeddings demo."""
    print("=" * 70)
    print("Embeddings Demo - Visualization and Analysis")
    print("=" * 70)

    # Setup client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()

    # Generate embeddings
    embeddings = get_embeddings(client, SAMPLE_TEXTS)

    # Similarity analysis
    show_similarity_analysis(SAMPLE_TEXTS, embeddings)
    show_within_cluster_similarity(SAMPLE_TEXTS, embeddings)

    # Dimensionality reduction
    print("\nPerforming t-SNE dimensionality reduction...")
    reduced = reduce_dimensions_tsne(embeddings, n_components=2)

    # Clustering
    print("Performing K-means clustering...")
    true_labels = (
        ["Technology"] * 5 + ["Nature"] * 5 + ["Food"] * 5
    )
    cluster_labels, centroids = cluster_embeddings(embeddings, n_clusters=3)

    # Map cluster labels to human-readable names
    cluster_name_map = {}
    for cluster_id in range(3):
        mask = np.array(cluster_labels) == cluster_id
        true_in_cluster = [l for l, m in zip(true_labels, mask) if m]
        if true_in_cluster:
            from collections import Counter
            most_common = Counter(true_in_cluster).most_common(1)[0][0]
            cluster_name_map[cluster_id] = most_common
        else:
            cluster_name_map[cluster_id] = f"Cluster {cluster_id}"

    readable_cluster_labels = [cluster_name_map[c] for c in cluster_labels]

    # Plot with true labels
    plot_embeddings(
        reduced,
        SAMPLE_TEXTS,
        true_labels,
        title="Embeddings - True Categories",
        output_file="embeddings_true_categories.png",
    )

    # Plot with discovered clusters
    plot_embeddings(
        reduced,
        SAMPLE_TEXTS,
        readable_cluster_labels,
        title="Embeddings - K-Means Discovered Clusters",
        output_file="embeddings_discovered_clusters.png",
    )

    print("\n" + "=" * 70)
    print("Demo complete! Check the generated PNG files for visualizations.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
