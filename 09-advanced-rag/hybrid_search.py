"""
Hybrid Search: BM25 + Dense Vector Search

Combines lexical (BM25) and semantic (dense) retrieval for better results.
Includes adaptive weighting based on query characteristics.
"""

import re
import math
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from collections import Counter


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    text: str
    score: float
    method: str
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. BM25 Implementation
# ---------------------------------------------------------------------------

class BM25:
    """Okapi BM25 implementation from scratch."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[str] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.tf: List[Counter] = []
        self.df: Counter = Counter()
        self.n_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenization."""
        return re.findall(r'\b\w+\b', text.lower())

    def fit(self, documents: List[str]):
        """Build BM25 index."""
        self.documents = documents
        self.n_docs = len(documents)
        self.tf = []
        self.df = Counter()

        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            self.tf.append(Counter(tokens))
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.df[token] += 1

        self.avg_doc_length = sum(self.doc_lengths) / self.n_docs if self.n_docs > 0 else 0

    def score(self, query: str, doc_idx: int) -> float:
        """Compute BM25 score for a query against a single document."""
        query_tokens = self._tokenize(query)
        score = 0.0

        for token in query_tokens:
            if token not in self.df:
                continue

            tf = self.tf[doc_idx].get(token, 0)
            df = self.df[token]
            doc_len = self.doc_lengths[doc_idx]

            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
            )
            score += idf * tf_norm

        return score

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search and return top-k results."""
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            results.append(SearchResult(
                text=self.documents[idx],
                score=score,
                method="bm25",
            ))
        return results


# ---------------------------------------------------------------------------
# 2. Dense Vector Search (ChromaDB)
# ---------------------------------------------------------------------------

class DenseSearcher:
    """Dense retrieval using ChromaDB + sentence-transformers."""

    def __init__(self):
        import chromadb
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name="hybrid_dense",
            metadata={"hnsw:space": "cosine"},
        )
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                return None
        return self._model

    def add_documents(self, documents: List[str]):
        """Add documents to the dense index."""
        model = self._get_model()
        ids = [f"doc_{i}" for i in range(len(documents))]

        if model:
            embeddings = model.encode(documents).tolist()
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=ids,
            )
        else:
            self.collection.add(documents=documents, ids=ids)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search using dense retrieval."""
        model = self._get_model()
        kwargs = {"n_results": top_k}

        if model:
            kwargs["query_embeddings"] = [model.encode([query])[0].tolist()]
        else:
            kwargs["query_texts"] = [query]

        results = self.collection.query(**kwargs)

        search_results = []
        for i in range(len(results["documents"][0])):
            score = 1 - results["distances"][0][i] if results.get("distances") else 0.0
            search_results.append(SearchResult(
                text=results["documents"][0][i],
                score=score,
                method="dense",
            ))
        return search_results


# ---------------------------------------------------------------------------
# 3. Hybrid Search with Weighted Scoring
# ---------------------------------------------------------------------------

class HybridSearcher:
    """Combines BM25 and dense retrieval with weighted scoring."""

    def __init__(self, bm25_weight: float = 0.4, dense_weight: float = 0.6):
        self.bm25 = BM25()
        self.dense = DenseSearcher()
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def add_documents(self, documents: List[str]):
        """Add documents to both indices."""
        self.bm25.fit(documents)
        self.dense.add_documents(documents)

    def _normalize_scores(self, results: List[SearchResult]) -> Dict[str, float]:
        """Normalize scores to [0, 1] range."""
        if not results:
            return {}
        scores = [r.score for r in results]
        min_s, max_s = min(scores), max(scores)
        range_s = max_s - min_s if max_s > min_s else 1.0
        return {r.text[:100]: (r.score - min_s) / range_s for r in results}

    def _adaptive_weights(self, query: str) -> Tuple[float, float]:
        """
        Adjust BM25/dense weights based on query characteristics.

        - Queries with many technical terms → favor BM25
        - Queries with abstract concepts → favor dense
        - Short queries → favor dense
        - Long queries → favor BM25
        """
        words = query.split()
        word_count = len(words)

        # Count technical/specific terms (heuristic: words with numbers, camelCase, or > 8 chars)
        technical_count = sum(
            1 for w in words
            if any(c.isdigit() for c in w)
            or len(w) > 8
            or w[0].isupper()
        )
        technical_ratio = technical_count / max(word_count, 1)

        # Short queries favor semantic understanding
        if word_count <= 4:
            return 0.3, 0.7

        # Technical queries favor exact matching
        if technical_ratio > 0.5:
            return 0.7, 0.3

        # Default: balanced with slight dense preference
        return self.bm25_weight, self.dense_weight

    def search(self, query: str, top_k: int = 10, adaptive: bool = True) -> List[SearchResult]:
        """Hybrid search combining BM25 and dense retrieval."""
        bm25_weight, dense_weight = self._adaptive_weights(query) if adaptive else (self.bm25_weight, self.dense_weight)

        bm25_results = self.bm25.search(query, top_k=top_k)
        dense_results = self.dense.search(query, top_k=top_k)

        bm25_scores = self._normalize_scores(bm25_results)
        dense_scores = self._normalize_scores(dense_results)

        combined: Dict[str, SearchResult] = {}

        for result in bm25_results:
            key = result.text[:100]
            normalized = bm25_scores.get(key, 0)
            combined[key] = SearchResult(
                text=result.text,
                score=bm25_weight * normalized,
                method="hybrid",
                metadata={"bm25_score": result.score, "bm25_norm": normalized},
            )

        for result in dense_results:
            key = result.text[:100]
            normalized = dense_scores.get(key, 0)
            if key in combined:
                combined[key].score += dense_weight * normalized
                combined[key].metadata["dense_score"] = result.score
                combined[key].metadata["dense_norm"] = normalized
            else:
                combined[key] = SearchResult(
                    text=result.text,
                    score=dense_weight * normalized,
                    method="hybrid",
                    metadata={"dense_score": result.score, "dense_norm": normalized},
                )

        ranked = sorted(combined.values(), key=lambda r: r.score, reverse=True)[:top_k]
        return ranked


# ---------------------------------------------------------------------------
# 4. Performance Comparison
# ---------------------------------------------------------------------------

def compare_search_methods(documents: List[str], query: str, top_k: int = 5):
    """Compare BM25, dense, and hybrid search side by side."""
    print(f"\nQuery: {query}")
    print("=" * 70)

    # BM25
    bm25 = BM25()
    bm25.fit(documents)
    bm25_results = bm25.search(query, top_k)
    print(f"\n[BM25] Top {top_k}:")
    for i, r in enumerate(bm25_results):
        print(f"  {i+1}. (score={r.score:.3f}) {r.text[:80]}...")

    # Dense
    dense = DenseSearcher()
    dense.add_documents(documents)
    dense_results = dense.search(query, top_k)
    print(f"\n[Dense] Top {top_k}:")
    for i, r in enumerate(dense_results):
        print(f"  {i+1}. (score={r.score:.3f}) {r.text[:80]}...")

    # Hybrid
    hybrid = HybridSearcher()
    hybrid.add_documents(documents)
    hybrid_results = hybrid.search(query, top_k)
    print(f"\n[Hybrid] Top {top_k}:")
    for i, r in enumerate(hybrid_results):
        bm25_s = r.metadata.get("bm25_score", 0)
        dense_s = r.metadata.get("dense_score", 0)
        print(f"  {i+1}. (score={r.score:.3f}, bm25={bm25_s:.3f}, dense={dense_s:.3f}) {r.text[:80]}...")

    # Agreement analysis
    bm25_texts = {r.text[:60] for r in bm25_results}
    dense_texts = {r.text[:60] for r in dense_results}
    hybrid_texts = {r.text[:60] for r in hybrid_results}

    bm25_only = bm25_texts - dense_texts
    dense_only = dense_texts - bm25_texts
    both = bm25_texts & dense_texts

    print(f"\n[Analysis]")
    print(f"  Found by both BM25 and Dense: {len(both)}")
    print(f"  BM25 only: {len(bm25_only)}")
    print(f"  Dense only: {len(dense_only)}")
    print(f"  Hybrid combines both: {len(hybrid_texts)}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS = [
    "Python is a high-level programming language created by Guido van Rossum in 1991.",
    "Machine learning algorithms build models based on training data to make predictions.",
    "Deep learning is a subset of machine learning using neural networks with many layers.",
    "Natural language processing enables computers to understand and generate human language.",
    "Computer vision gives machines the ability to interpret and understand visual information.",
    "Reinforcement learning trains agents to make decisions by rewarding desired behaviors.",
    "The Transformer architecture revolutionized NLP with self-attention mechanisms.",
    "BERT is a bidirectional transformer pre-trained on large text corpora for NLP tasks.",
    "GPT models are autoregressive language models trained to predict the next token.",
    "Transfer learning allows reusing pre-trained models for new downstream tasks.",
    "Docker containers package applications with their dependencies for consistent deployment.",
    "Kubernetes orchestrates containerized applications across clusters of machines.",
    "REST APIs use HTTP methods for communication between distributed systems.",
    "GraphQL provides a query language for APIs with flexible data fetching.",
    "PostgreSQL is an advanced open-source relational database management system.",
    "Redis is an in-memory data structure store used as a database and cache.",
    "Neural networks consist of layers of interconnected nodes that process information.",
    "Convolutional neural networks are particularly effective for image processing tasks.",
    "Recurrent neural networks process sequential data like text and time series.",
    "Generative adversarial networks learn to generate realistic synthetic data.",
]


def demo():
    """Run the hybrid search demo."""
    print("=" * 70)
    print("HYBRID SEARCH DEMO: BM25 + Dense Retrieval")
    print("=" * 70)

    queries = [
        "What is the Transformer architecture?",
        "Docker container deployment",
        "How do neural networks learn from data?",
        "GPT-3 model parameters",
    ]

    for query in queries:
        compare_search_methods(SAMPLE_DOCUMENTS, query, top_k=5)


if __name__ == "__main__":
    demo()
