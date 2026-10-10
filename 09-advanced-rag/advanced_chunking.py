"""
Advanced Chunking Strategies for RAG

Compares multiple chunking approaches with quality metrics.
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict = field(default_factory=dict)


@dataclass
class ChunkingResult:
    chunks: List[Chunk]
    strategy: str
    stats: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. Fixed-size chunking (baseline)
# ---------------------------------------------------------------------------

def fixed_size_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    strategy_name: str = "fixed_size",
) -> ChunkingResult:
    """Split text into fixed-size chunks with optional overlap."""
    chunks = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                index=idx,
                start_char=start,
                end_char=end,
                metadata={"strategy": strategy_name},
            ))
            idx += 1
        start += chunk_size - overlap

    return ChunkingResult(
        chunks=chunks,
        strategy=strategy_name,
        stats=_compute_stats(chunks),
    )


# ---------------------------------------------------------------------------
# 2. Sentence-aware chunking
# ---------------------------------------------------------------------------

def sentence_chunking(
    text: str,
    max_chunk_size: int = 500,
    strategy_name: str = "sentence_aware",
) -> ChunkingResult:
    """Split on sentence boundaries, merging sentences up to max_chunk_size."""
    sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_pattern.split(text)

    chunks = []
    current_chunk: List[str] = []
    current_size = 0
    idx = 0
    char_pos = 0

    for sentence in sentences:
        sent_len = len(sentence)
        if current_size + sent_len > max_chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            chunks.append(Chunk(
                text=chunk_text,
                index=idx,
                start_char=char_pos,
                end_char=char_pos + len(chunk_text),
                metadata={"strategy": strategy_name},
            ))
            idx += 1
            char_pos += len(chunk_text) + 1
            current_chunk = []
            current_size = 0
        current_chunk.append(sentence)
        current_size += sent_len

    if current_chunk:
        chunk_text = " ".join(current_chunk).strip()
        chunks.append(Chunk(
            text=chunk_text,
            index=idx,
            start_char=char_pos,
            end_char=char_pos + len(chunk_text),
            metadata={"strategy": strategy_name},
        ))

    return ChunkingResult(
        chunks=chunks,
        strategy=strategy_name,
        stats=_compute_stats(chunks),
    )


# ---------------------------------------------------------------------------
# 3. Paragraph-aware chunking
# ---------------------------------------------------------------------------

def paragraph_chunking(
    text: str,
    max_chunk_size: int = 1000,
    strategy_name: str = "paragraph_aware",
) -> ChunkingResult:
    """Split on double-newlines (paragraph boundaries)."""
    paragraphs = re.split(r'\n\s*\n', text)

    chunks = []
    current_group: List[str] = []
    current_size = 0
    idx = 0
    char_pos = 0

    for para in paragraphs:
        para_len = len(para)
        if current_size + para_len > max_chunk_size and current_group:
            chunk_text = "\n\n".join(current_group).strip()
            chunks.append(Chunk(
                text=chunk_text,
                index=idx,
                start_char=char_pos,
                end_char=char_pos + len(chunk_text),
                metadata={"strategy": strategy_name},
            ))
            idx += 1
            char_pos += len(chunk_text) + 2
            current_group = []
            current_size = 0
        current_group.append(para)
        current_size += para_len

    if current_group:
        chunk_text = "\n\n".join(current_group).strip()
        chunks.append(Chunk(
            text=chunk_text,
            index=idx,
            start_char=char_pos,
            end_char=char_pos + len(chunk_text),
            metadata={"strategy": strategy_name},
        ))

    return ChunkingResult(
        chunks=chunks,
        strategy=strategy_name,
        stats=_compute_stats(chunks),
    )


# ---------------------------------------------------------------------------
# 4. Semantic chunking (embedding-based topic detection)
# ---------------------------------------------------------------------------

def semantic_chunking(
    text: str,
    max_chunk_size: int = 1000,
    similarity_threshold: float = 0.5,
    strategy_name: str = "semantic",
) -> ChunkingResult:
    """
    Split text where embedding similarity between consecutive sentences drops
    below a threshold (indicating a topic change).

    Falls back to fixed-size splitting if sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("[semantic_chunking] sentence-transformers not installed, falling back to sentence chunking")
        result = sentence_chunking(text, max_chunk_size, strategy_name)
        return result

    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_pattern.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return ChunkingResult(
            chunks=[Chunk(text=text, index=0, start_char=0, end_char=len(text))],
            strategy=strategy_name,
            stats=_compute_stats([]),
        )

    embeddings = model.encode(sentences)

    chunks: List[Chunk] = []
    current_group: List[str] = [sentences[0]]
    idx = 0
    char_pos = 0

    for i in range(1, len(sentences)):
        cos_sim = float(np.dot(embeddings[i], embeddings[i - 1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i - 1]) + 1e-10
        ))
        group_text = " ".join(current_group)
        topic_shift = cos_sim < similarity_threshold
        too_long = len(group_text) + len(sentences[i]) > max_chunk_size

        if topic_shift or too_long:
            chunk_text = " ".join(current_group).strip()
            chunks.append(Chunk(
                text=chunk_text,
                index=idx,
                start_char=char_pos,
                end_char=char_pos + len(chunk_text),
                metadata={"strategy": strategy_name, "avg_similarity": round(cos_sim, 3)},
            ))
            idx += 1
            char_pos += len(chunk_text) + 1
            current_group = [sentences[i]]
        else:
            current_group.append(sentences[i])

    if current_group:
        chunk_text = " ".join(current_group).strip()
        chunks.append(Chunk(
            text=chunk_text,
            index=idx,
            start_char=char_pos,
            end_char=char_pos + len(chunk_text),
            metadata={"strategy": strategy_name},
        ))

    return ChunkingResult(
        chunks=chunks,
        strategy=strategy_name,
        stats=_compute_stats(chunks),
    )


# ---------------------------------------------------------------------------
# 5. Recursive chunking (try large separators first)
# ---------------------------------------------------------------------------

def recursive_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    strategy_name: str = "recursive",
) -> ChunkingResult:
    """
    Recursively try splitting on "\n\n", "\n", ". ", then characters.
    Falls back to smaller separators when chunks are still too large.
    """
    separators = ["\n\n", "\n", ". ", " "]

    def _split_recursive(text: str, seps: List[str]) -> List[str]:
        if len(text) <= chunk_size or not seps:
            return [text] if text.strip() else []

        sep = seps[0]
        parts = text.split(sep)
        result = []
        for part in parts:
            if len(part) <= chunk_size:
                result.append(part)
            else:
                result.extend(_split_recursive(part, seps[1:]))
        return result

    raw_chunks = _split_recursive(text, separators)

    chunks = []
    idx = 0
    char_pos = 0
    for raw in raw_chunks:
        chunk_text = raw.strip()
        if not chunk_text:
            continue
        chunks.append(Chunk(
            text=chunk_text,
            index=idx,
            start_char=char_pos,
            end_char=char_pos + len(chunk_text),
            metadata={"strategy": strategy_name},
        ))
        idx += 1
        char_pos += len(chunk_text) + 1

    return ChunkingResult(
        chunks=chunks,
        strategy=strategy_name,
        stats=_compute_stats(chunks),
    )


# ---------------------------------------------------------------------------
# 6. Document-aware chunking (respects headers / sections)
# ---------------------------------------------------------------------------

def document_aware_chunking(
    text: str,
    max_chunk_size: int = 1500,
    strategy_name: str = "document_aware",
) -> ChunkingResult:
    """
    Split on Markdown/structured document markers:
    headings (#, ##, ###), horizontal rules (---), and blank lines.
    """
    section_pattern = re.compile(
        r'(?=^#{1,3}\s)|(?=^---\s*$)',
        re.MULTILINE,
    )
    sections = section_pattern.split(text)
    sections = [s.strip() for s in sections if s.strip()]

    chunks: List[Chunk] = []
    current_group: List[str] = []
    current_size = 0
    idx = 0
    char_pos = 0

    for section in sections:
        sec_len = len(section)
        if current_size + sec_len > max_chunk_size and current_group:
            chunk_text = "\n\n".join(current_group).strip()
            chunks.append(Chunk(
                text=chunk_text,
                index=idx,
                start_char=char_pos,
                end_char=char_pos + len(chunk_text),
                metadata={"strategy": strategy_name},
            ))
            idx += 1
            char_pos += len(chunk_text) + 2
            current_group = []
            current_size = 0
        current_group.append(section)
        current_size += sec_len

    if current_group:
        chunk_text = "\n\n".join(current_group).strip()
        chunks.append(Chunk(
            text=chunk_text,
            index=idx,
            start_char=char_pos,
            end_char=char_pos + len(chunk_text),
            metadata={"strategy": strategy_name},
        ))

    return ChunkingResult(
        chunks=chunks,
        strategy=strategy_name,
        stats=_compute_stats(chunks),
    )


# ---------------------------------------------------------------------------
# 7. Agentic chunking (LLM decides boundaries)
# ---------------------------------------------------------------------------

def agentic_chunking(
    text: str,
    chunk_size: int = 800,
    strategy_name: str = "agentic",
) -> ChunkingResult:
    """
    Uses an LLM to identify natural chunk boundaries in text.
    Requires an OpenAI-compatible API. Falls back to sentence chunking otherwise.
    """
    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception:
        print("[agentic_chunking] OpenAI client not available, falling back to sentence chunking")
        return sentence_chunking(text, chunk_size, strategy_name)

    prompt = f"""Split the following text into coherent chunks. Each chunk should be a self-contained section.

Return ONLY a JSON array of strings (the chunks), nothing else.

Text:
{text[:8000]}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        import json
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        chunk_texts = json.loads(content)
    except Exception as e:
        print(f"[agentic_chunking] LLM call failed ({e}), falling back to sentence chunking")
        return sentence_chunking(text, chunk_size, strategy_name)

    chunks = []
    char_pos = 0
    for idx, ct in enumerate(chunk_texts):
        ct = ct.strip()
        if not ct:
            continue
        chunks.append(Chunk(
            text=ct,
            index=idx,
            start_char=char_pos,
            end_char=char_pos + len(ct),
            metadata={"strategy": strategy_name},
        ))
        char_pos += len(ct) + 1

    return ChunkingResult(
        chunks=chunks,
        strategy=strategy_name,
        stats=_compute_stats(chunks),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_stats(chunks: List[Chunk]) -> Dict:
    """Compute chunk quality metrics."""
    if not chunks:
        return {"count": 0, "avg_size": 0, "min_size": 0, "max_size": 0, "std_dev": 0}

    sizes = [len(c.text) for c in chunks]
    avg = sum(sizes) / len(sizes)
    variance = sum((s - avg) ** 2 for s in sizes) / len(sizes) if len(sizes) > 1 else 0

    return {
        "count": len(chunks),
        "avg_size": round(avg, 1),
        "min_size": min(sizes),
        "max_size": max(sizes),
        "std_dev": round(variance ** 0.5, 1),
    }


# ---------------------------------------------------------------------------
# Comparison demo
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENT = """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future.

## Supervised Learning

Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples. In supervised learning, each example is a pair consisting of an input object and a desired output value.

Common algorithms include linear regression, logistic regression, decision trees, random forests, and support vector machines. These algorithms have been applied successfully to problems in computer vision, speech recognition, and natural language processing.

## Unsupervised Learning

Unsupervised learning is a type of machine learning that looks for previously undetected patterns in a data set with no pre-existing labels and with a minimum of human supervision. In contrast to supervised learning that usually makes use of human-labeled data, unsupervised learning, also known as self-organization, allows modeling probability densities over inputs.

Clustering is the most common unsupervised learning technique. It involves grouping data points such that objects in the same group are more similar to other objects in the same group than those in other groups. Popular clustering algorithms include K-means, hierarchical clustering, and DBSCAN.

## Reinforcement Learning

Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize the notion of cumulative reward. It differs from supervised learning in that labeled input-output pairs need not be presented, and sub-optimal actions need not be explicitly corrected.

Key concepts include exploration vs exploitation, the discount factor, and the bellman equation. Applications include game playing (AlphaGo), robotics, and autonomous driving.

## Deep Learning

Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures have been applied to computer vision, speech recognition, natural language processing, machine translation, and drug design.

Convolutional neural networks (CNNs) are particularly effective for image processing tasks. Recurrent neural networks (RNNs) and Transformers excel at sequential data processing. The Transformer architecture, introduced in 2017, has revolutionized natural language processing and led to models like BERT and GPT.

## Conclusion

Machine learning continues to evolve rapidly, with new architectures and techniques emerging regularly. Understanding the fundamental approaches — supervised, unsupervised, reinforcement, and deep learning — provides a solid foundation for tackling real-world problems.
"""


def print_chunking_comparison():
    """Run all chunking strategies and compare results."""
    strategies = [
        ("Fixed-size (500, overlap=50)", fixed_size_chunking, {"chunk_size": 500, "overlap": 50}),
        ("Sentence-aware", sentence_chunking, {"max_chunk_size": 500}),
        ("Paragraph-aware", paragraph_chunking, {"max_chunk_size": 800}),
        ("Recursive", recursive_chunking, {"chunk_size": 500, "overlap": 50}),
        ("Document-aware", document_aware_chunking, {"max_chunk_size": 800}),
    ]

    print("=" * 70)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 70)
    print(f"Document length: {len(SAMPLE_DOCUMENT)} chars")
    print()

    results: List[Tuple[str, ChunkingResult]] = []

    for name, func, kwargs in strategies:
        result = func(SAMPLE_DOCUMENT, **kwargs)
        results.append((name, result))
        stats = result.stats
        print(f"--- {name} ---")
        print(f"  Chunks: {stats['count']}")
        print(f"  Avg size: {stats['avg_size']} chars")
        print(f"  Min/Max: {stats['min_size']} / {stats['max_size']}")
        print(f"  Std dev: {stats['std_dev']}")
        print()

    # Print first chunk from each strategy as sample
    print("=" * 70)
    print("SAMPLE CHUNKS (first from each strategy)")
    print("=" * 70)
    for name, result in results:
        if result.chunks:
            sample = result.chunks[0].text[:200]
            print(f"\n--- {name} ---")
            print(f"  '{sample}...'")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Strategy':<25} {'Chunks':>7} {'Avg Size':>10} {'Std Dev':>10}")
    print("-" * 55)
    for name, result in results:
        s = result.stats
        print(f"{name:<25} {s['count']:>7} {s['avg_size']:>10} {s['std_dev']:>10}")


if __name__ == "__main__":
    print_chunking_comparison()
