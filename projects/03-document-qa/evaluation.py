"""Run a small, offline retrieval benchmark (Recall@K and MRR)."""

from __future__ import annotations

import json
from pathlib import Path

from rag import chunk_document, retrieve


ROOT = Path(__file__).parent
TOP_K = 3


def main() -> None:
    chunks = []
    for path in sorted((ROOT / "sample_docs").glob("*.md")):
        chunks.extend(chunk_document(path.read_text(encoding="utf-8"), path.name))
    cases = json.loads((ROOT / "evaluation_set.json").read_text(encoding="utf-8"))

    reciprocal_ranks = []
    hits = 0
    print(f"{'Question':<68} {'Expected':<26} Rank")
    print("-" * 102)
    for case in cases:
        results = retrieve(case["question"], chunks, top_k=TOP_K)
        ranks = [rank for rank, (chunk, _) in enumerate(results, start=1) if chunk.source == case["expected_source"]]
        rank = min(ranks) if ranks else None
        hits += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        short_question = case["question"][:65]
        print(f"{short_question:<68} {case['expected_source']:<26} {rank or 'miss'}")

    recall_at_k = hits / len(cases) if cases else 0.0
    mean_reciprocal_rank = sum(reciprocal_ranks) / len(reciprocal_ranks) if cases else 0.0
    print(f"\nRecall@{TOP_K}: {recall_at_k:.1%} ({hits}/{len(cases)})")
    print(f"MRR:        {mean_reciprocal_rank:.3f}")


if __name__ == "__main__":
    main()
