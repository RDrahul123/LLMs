"""
Response Caching — Exact match + semantic caching for LLM responses.

Implements:
  - Exact match cache (hash of messages)
  - Semantic cache (embedding similarity)
  - TTL-based cache expiration
  - Cache statistics (hit rate, size)
  - Redis-style interface (in-memory for demo)
  - Cache invalidation strategies
  - Demo showing cache performance improvement

Run: python caching.py
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache Entry
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: float
    ttl_seconds: float
    access_count: int = 0
    last_accessed: float = 0.0

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


# ---------------------------------------------------------------------------
# Cache Statistics
# ---------------------------------------------------------------------------


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    def record_hit(self):
        self.hits += 1
        self.total_requests += 1

    def record_miss(self):
        self.misses += 1
        self.total_requests += 1

    def record_eviction(self):
        self.evictions += 1

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "total_requests": self.total_requests,
            "hit_rate": f"{self.hit_rate:.2%}",
        }


# ---------------------------------------------------------------------------
# Exact Match Cache
# ---------------------------------------------------------------------------


class ExactMatchCache:
    """
    Hash-based cache: identical requests always return cached responses.
    Fast, deterministic, no false positives.
    """

    def __init__(self, max_size: int = 1000, default_ttl: float = 3600.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: dict[str, CacheEntry] = {}
        self.stats = CacheStats()
        self._lock = Lock()

    @staticmethod
    def make_key(model: str, messages: list[dict], **kwargs) -> str:
        """Generate deterministic cache key from request parameters."""
        content = json.dumps(
            {"model": model, "messages": messages, **kwargs},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self.cache.get(key)
            if entry is None:
                self.stats.record_miss()
                return None
            if entry.is_expired:
                del self.cache[key]
                self.stats.record_miss()
                return None
            entry.access_count += 1
            entry.last_accessed = time.time()
            self.stats.record_hit()
            return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None):
        with self._lock:
            if len(self.cache) >= self.max_size:
                self._evict()
            self.cache[key] = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                ttl_seconds=ttl or self.default_ttl,
                last_accessed=time.time(),
            )

    def invalidate(self, key: str) -> bool:
        with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def invalidate_pattern(self, prefix: str) -> int:
        """Invalidate all keys starting with prefix."""
        with self._lock:
            keys_to_remove = [k for k in self.cache if k.startswith(prefix)]
            for k in keys_to_remove:
                del self.cache[k]
            return len(keys_to_remove)

    def clear(self):
        with self._lock:
            self.cache.clear()

    def _evict(self):
        """Evict least recently accessed expired entries, then oldest."""
        now = time.time()
        expired = [
            (k, v) for k, v in self.cache.items()
            if v.is_expired
        ]
        for k, _ in expired:
            del self.cache[k]
            self.stats.record_eviction()

        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache, key=lambda k: self.cache[k].last_accessed)
            del self.cache[oldest_key]
            self.stats.record_eviction()

    def size(self) -> int:
        return len(self.cache)

    def stats_dict(self) -> dict:
        return {
            **self.stats.to_dict(),
            "size": self.size(),
            "max_size": self.max_size,
        }


# ---------------------------------------------------------------------------
# Semantic Cache (Cosine Similarity)
# ---------------------------------------------------------------------------


class SemanticCache:
    """
    Embedding-based cache: finds semantically similar prompts.
    Uses cosine similarity between prompt embeddings.

    Pros: Catches paraphrased requests.
    Cons: Slower than exact match, may have false positives.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        max_size: int = 500,
        default_ttl: float = 3600.0,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.entries: list[tuple[list[float], Any, float]] = []  # (embedding, value, created_at)
        self.stats = CacheStats()
        self._lock = Lock()

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get(self, query_embedding: list[float]) -> Any | None:
        with self._lock:
            if not self.entries:
                self.stats.record_miss()
                return None

            best_score = -1.0
            best_idx = -1
            now = time.time()

            for i, (emb, value, created_at) in enumerate(self.entries):
                if now - created_at > self.default_ttl:
                    continue
                score = self.cosine_similarity(query_embedding, emb)
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx >= 0 and best_score >= self.similarity_threshold:
                self.stats.record_hit()
                return self.entries[best_idx][1]

            self.stats.record_miss()
            return None

    def set(self, embedding: list[float], value: Any, ttl: float | None = None):
        with self._lock:
            if len(self.entries) >= self.max_size:
                self.entries.pop(0)  # Remove oldest
                self.stats.record_eviction()
            self.entries.append((embedding, value, time.time()))

    def size(self) -> int:
        return len(self.entries)

    def stats_dict(self) -> dict:
        return {
            **self.stats.to_dict(),
            "size": self.size(),
            "max_size": self.max_size,
            "similarity_threshold": self.similarity_threshold,
        }


# ---------------------------------------------------------------------------
# Combined Cache (Exact First, Semantic Second)
# ---------------------------------------------------------------------------


class LLMCache:
    """
    Two-tier cache:
      1. Exact match (fast, O(1) lookup)
      2. Semantic (slower, catches paraphrases)
    """

    def __init__(
        self,
        exact_max_size: int = 1000,
        semantic_max_size: int = 500,
        semantic_threshold: float = 0.95,
        default_ttl: float = 3600.0,
    ):
        self.exact = ExactMatchCache(exact_max_size, default_ttl)
        self.semantic = SemanticCache(semantic_threshold, semantic_max_size, default_ttl)

    def get(
        self,
        model: str,
        messages: list[dict],
        query_embedding: list[float] | None = None,
        **kwargs,
    ) -> Any | None:
        key = self.exact.make_key(model, messages, **kwargs)
        result = self.exact.get(key)
        if result is not None:
            return result

        if query_embedding is not None:
            result = self.semantic.get(query_embedding)
            if result is not None:
                self.exact.set(key, result)
                return result

        return None

    def set(
        self,
        model: str,
        messages: list[dict],
        value: Any,
        embedding: list[float] | None = None,
        ttl: float | None = None,
        **kwargs,
    ):
        key = self.exact.make_key(model, messages, **kwargs)
        self.exact.set(key, value, ttl)
        if embedding is not None:
            self.semantic.set(embedding, value, ttl)

    def invalidate(self, model: str, messages: list[dict], **kwargs) -> bool:
        key = self.exact.make_key(model, messages, **kwargs)
        return self.exact.invalidate(key)

    def clear(self):
        self.exact.clear()
        self.semantic.entries.clear()

    def stats_dict(self) -> dict:
        return {
            "exact": self.exact.stats_dict(),
            "semantic": self.semantic.stats_dict(),
        }


# ---------------------------------------------------------------------------
# Invalidation Strategies
# ---------------------------------------------------------------------------


class InvalidationStrategy:
    """Strategies for invalidating cached LLM responses."""

    @staticmethod
    def by_model(cache: LLMCache, model: str):
        """Invalidate all cached responses for a specific model."""
        keys_to_remove = [
            k for k in cache.exact.cache
            if cache.exact.cache[k].value.get("model") == model
        ]
        for k in keys_to_remove:
            cache.exact.invalidate(k)
        return len(keys_to_remove)

    @staticmethod
    def by_age(cache: LLMCache, max_age_seconds: float):
        """Invalidate entries older than max_age_seconds."""
        now = time.time()
        keys_to_remove = [
            k for k, v in cache.exact.cache.items()
            if now - v.created_at > max_age_seconds
        ]
        for k in keys_to_remove:
            cache.exact.invalidate(k)
        return len(keys_to_remove)

    @staticmethod
    def by_access_count(cache: LLMCache, min_access_count: int):
        """Invalidate rarely accessed entries."""
        keys_to_remove = [
            k for k, v in cache.exact.cache.items()
            if v.access_count < min_access_count and v.age_seconds > 300
        ]
        for k in keys_to_remove:
            cache.exact.invalidate(k)
        return len(keys_to_remove)

    @staticmethod
    def all(cache: LLMCache):
        """Nuclear option: clear everything."""
        cache.clear()
        return 0


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def simulate_llm_call(messages: list[dict], model: str = "gpt-4o") -> dict:
    """Simulate an LLM call with latency."""
    time.sleep(0.1)  # Simulate API latency
    return {
        "model": model,
        "response": f"Response to: {messages[-1]['content'][:50]}",
        "tokens": 100,
    }


def demo_exact_cache():
    print("\n=== Exact Match Cache ===")
    cache = ExactMatchCache(max_size=100, default_ttl=60.0)

    messages = [{"role": "user", "content": "What is Python?"}]

    # First call — miss
    key = cache.make_key("gpt-4o", messages)
    result = cache.get(key)
    if result is None:
        print("  Cache MISS — calling LLM")
        result = simulate_llm_call(messages)
        cache.set(key, result)
    else:
        print(f"  Cache HIT: {result}")

    # Second call — hit
    result = cache.get(key)
    if result is not None:
        print(f"  Cache HIT: {result}")
    else:
        print("  Cache MISS")

    print(f"  Stats: {cache.stats_dict()}")


def demo_semantic_cache():
    print("\n=== Semantic Cache ===")
    cache = SemanticCache(similarity_threshold=0.9, max_size=100)

    # Store embedding for "What is Python?"
    embedding1 = [0.8, 0.1, 0.3, 0.9, 0.2]
    value = {"response": "Python is a programming language."}
    cache.set(embedding1, value)

    # Query with similar embedding (paraphrase)
    query_similar = [0.79, 0.11, 0.28, 0.91, 0.19]
    result = cache.get(query_similar)
    if result:
        print(f"  Semantic HIT: {result}")
    else:
        print("  Semantic MISS")

    # Query with different embedding
    query_different = [0.1, 0.9, 0.1, 0.1, 0.9]
    result = cache.get(query_different)
    if result:
        print(f"  Semantic HIT: {result}")
    else:
        print("  Semantic MISS (different topic)")

    print(f"  Stats: {cache.stats_dict()}")


def demo_combined_cache():
    print("\n=== Combined Cache (Performance Demo) ===")
    cache = LLMCache(default_ttl=300.0)

    test_messages = [
        [{"role": "user", "content": "Explain machine learning"}],
        [{"role": "user", "content": "What is deep learning?"}],
        [{"role": "user", "content": "How does neural networks work?"}],
        [{"role": "user", "content": "Explain machine learning"}],  # Duplicate
    ]

    total_time = 0
    for i, msgs in enumerate(test_messages):
        key = cache.exact.make_key("gpt-4o", msgs)
        start = time.time()
        result = cache.get("gpt-4o", msgs)
        elapsed = (time.time() - start) * 1000

        if result is None:
            result = simulate_llm_call(msgs)
            cache.set("gpt-4o", msgs, result)
            status = "MISS"
        else:
            status = "HIT"

        total_time += elapsed
        print(f"  Request {i + 1}: {status} ({elapsed:.2f}ms)")

    print(f"\n  Final cache stats: {json.dumps(cache.stats_dict(), indent=2)}")
    print(f"  Total time: {total_time:.2f}ms")


def demo_invalidation():
    print("\n=== Cache Invalidation ===")
    cache = LLMCache(default_ttl=300.0)

    # Populate cache
    for i in range(5):
        msgs = [{"role": "user", "content": f"Question {i}"}]
        cache.set("gpt-4o", msgs, {"response": f"Answer {i}"})

    print(f"  Before: {cache.exact.size()} entries")

    # Invalidate by age (invalidate everything older than 0 seconds = all)
    removed = InvalidationStrategy.by_age(cache, max_age_seconds=0)
    print(f"  After age invalidation: {cache.exact.size()} entries (removed {removed})")


if __name__ == "__main__":
    demo_exact_cache()
    demo_semantic_cache()
    demo_combined_cache()
    demo_invalidation()
    print("\nAll cache demos complete.")
