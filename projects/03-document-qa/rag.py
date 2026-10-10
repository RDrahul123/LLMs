"""Document parsing, overlapping chunking, and BM25 retrieval for ContextIQ."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    chunk_index: int


def extract_text(filename: str, content: bytes) -> str:
    """Extract UTF-8 text or PDF page text from an uploaded file."""
    suffix = filename.lower().rsplit(".", maxsplit=1)[-1]
    if suffix in {"txt", "md", "markdown"}:
        text = content.decode("utf-8", errors="replace")
    elif suffix == "pdf":
        try:
            import pymupdf
        except ImportError as exc:  # pragma: no cover - dependency is declared in requirements
            raise RuntimeError("PDF support requires PyMuPDF. Install the project requirements.") from exc
        try:
            with pymupdf.open(stream=content, filetype="pdf") as document:
                text = "\n".join(page.get_text("text") for page in document)
        except Exception as exc:
            raise ValueError("The PDF could not be read. Check that it is a valid, text-based PDF.") from exc
    else:
        raise ValueError(f"Unsupported file type: .{suffix}")
    return re.sub(r"\s+", " ", text).strip()


def chunk_document(text: str, source: str, chunk_size: int = 900, overlap: int = 150) -> list[Chunk]:
    """Split text into readable, overlapping character windows."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1")
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    chunks: list[Chunk] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        excerpt = normalized[start:end].strip()
        if excerpt:
            chunks.append(Chunk(text=excerpt, source=source, chunk_index=len(chunks)))
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def retrieve(query: str, chunks: list[Chunk], top_k: int = 4, k1: float = 1.5, b: float = 0.75) -> list[tuple[Chunk, float]]:
    """Return the top BM25-ranked chunks as (chunk, relevance score) pairs."""
    query_terms = _tokens(query)
    if not query_terms or not chunks or top_k <= 0:
        return []

    tokenized = [_tokens(chunk.text) for chunk in chunks]
    doc_frequencies: Counter[str] = Counter()
    for tokens in tokenized:
        doc_frequencies.update(set(tokens))
    average_length = sum(map(len, tokenized)) / len(tokenized)
    if average_length == 0:
        return []

    query_frequency = Counter(query_terms)
    scored: list[tuple[Chunk, float, int]] = []
    for position, (chunk, tokens) in enumerate(zip(chunks, tokenized)):
        term_frequency = Counter(tokens)
        score = 0.0
        for term, query_count in query_frequency.items():
            frequency = term_frequency[term]
            if not frequency:
                continue
            document_frequency = doc_frequencies[term]
            inverse_frequency = math.log1p(
                (len(chunks) - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            length_norm = k1 * (1 - b + b * len(tokens) / average_length)
            score += inverse_frequency * (frequency * (k1 + 1)) / (frequency + length_norm) * query_count
        if score > 0:
            scored.append((chunk, score, position))
    scored.sort(key=lambda item: (-item[1], item[2]))
    return [(chunk, score) for chunk, score, _ in scored[:top_k]]


def unique_sources(chunks: list[Chunk]) -> list[str]:
    """List document names in their original order, without duplicates."""
    return list(dict.fromkeys(chunk.source for chunk in chunks))
