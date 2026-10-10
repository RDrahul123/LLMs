# Module 09: Advanced RAG

![A retrieval-augmented generation pipeline from indexing through answer verification](../assets/diagrams/rag-pipeline.svg)

Basic RAG (Retrieve → Generate) fails on real-world tasks: vague queries return irrelevant chunks, long documents lose context, and hallucinations go unchecked. This module covers production-grade techniques that fix these gaps.

## Why Basic RAG Isn't Enough

| Problem | Basic RAG Failure | Advanced Fix |
|---|---|---|
| Vague queries | Retrieves generic chunks | HyDE / Multi-query transformation |
| Long documents | Loses cross-section context | Graph RAG / Document-aware chunking |
| Irrelevant retrievals | LLM sees noise | Reranking / Hybrid search |
| Hallucination | No faithfulness check | Self-reflecting RAG / RAGAS eval |
| Tables/Images | Missed entirely | Multimodal RAG |
| Complex reasoning | Single retrieval insufficient | Agentic RAG (iterative) |

## Advanced Chunking Strategies

Chunking determines what the retriever sees. Bad chunks = bad answers.

- **Fixed-size with overlap** — baseline, splits at character/token boundaries
- **Sentence-aware** — splits on sentence boundaries, preserves meaning
- **Paragraph-aware** — respects natural document flow
- **Semantic chunking** — uses embeddings to detect topic shifts, creates coherent chunks
- **Recursive chunking** — tries large chunks first, falls back to smaller ones
- **Document-aware** — respects headers, sections, lists (Markdown/HTML aware)
- **Agentic chunking** — LLM decides chunk boundaries for maximum coherence

See `advanced_chunking.py` for implementations and comparison.

## Reranking

Initial retrieval (BM25, dense) uses fast but shallow matching. Reranking applies deep cross-encoder scoring to re-order results.

- **Cross-encoders** — `sentence-transformers` cross-encoder models score query-document pairs jointly
- **Cohere Rerank** — API-based reranking, strong performance
- **BGE reranker** — open-source alternative from BAAI
- **Reciprocal Rank Fusion** — combine rankings from multiple retrievers

Pipeline: `retrieve top-K → rerank → keep top-N → generate`

See `reranking.py` for full implementation.

## Hybrid Search (BM25 + Dense)

| Method | Strengths | Weaknesses |
|---|---|---|
| BM25 (sparse) | Exact keyword matching, fast | No semantic understanding |
| Dense (vector) | Semantic similarity | Misses exact terms |
| Hybrid | Best of both worlds | Slightly higher latency |

Combines BM25 lexical scores with dense vector similarity using weighted fusion. Adaptive weighting adjusts based on query characteristics.

See `hybrid_search.py` for implementation.

## Graph RAG

Builds a knowledge graph from documents, then retrieves relevant subgraphs for context.

1. **Extract** — entities and relationships from text (NER + relation extraction)
2. **Build** — knowledge graph with NetworkX
3. **Retrieve** — find relevant subgraph for query
4. **Combine** — merge graph context with vector retrieval results

Best for multi-hop reasoning and questions requiring relationship traversal.

See `graph_rag.py` for full pipeline.

## Agentic RAG

Self-reflecting, iterative retrieval that doesn't give up on the first attempt.

1. Retrieve context
2. LLM evaluates: "Is this sufficient to answer?"
3. If no → rewrite query, retrieve again
4. If yes → generate answer with confidence score
5. Repeat up to N times

Handles complex questions that require multiple retrieval rounds.

See `agentic_rag.py` for implementation.

## Multimodal RAG

Extends RAG to images, tables, and charts in documents.

- Extract images and tables from PDFs
- Generate descriptions with vision models (GPT-4V)
- Embed images alongside text
- Retrieve both text and image chunks
- Generate answers with visual context

See `multimodal_rag.py` for full pipeline.

## Query Transformation

Transforms user queries to improve retrieval quality.

- **HyDE** (Hypothetical Document Embeddings) — generate a hypothetical answer, embed it, retrieve similar real documents
- **Multi-query** — generate N query variations, retrieve for each, merge results
- **Step-back prompting** — abstract the query to retrieve broader context first

## RAG Evaluation (RAGAS)

Measures RAG quality with standardized metrics.

| Metric | What It Measures |
|---|---|
| Faithfulness | Is the answer grounded in retrieved context? |
| Answer Relevancy | Does the answer address the query? |
| Context Precision | Are retrieved chunks actually relevant? |
| Context Recall | Did we retrieve all necessary information? |

See `rag_evaluation.py` for RAGAS integration.

## Decision Guide

| Use Case | Recommended Technique |
|---|---|
| Simple FAQ | Basic RAG with sentence chunking |
| Long documents | Document-aware chunking + reranking |
| Technical docs | Hybrid search + reranking |
| Multi-hop questions | Graph RAG or Agentic RAG |
| Vague user queries | HyDE + multi-query |
| Documents with images/tables | Multimodal RAG |
| Production system | All of the above + RAGAS evaluation |

## Module Structure

```
09-advanced-rag/
├── README.md                 # This file
├── advanced_chunking.py      # Chunking strategies comparison
├── reranking.py              # Reranking pipeline
├── hybrid_search.py          # BM25 + vector search
├── graph_rag.py              # Knowledge graph RAG
├── agentic_rag.py            # Self-reflecting RAG
├── multimodal_rag.py         # Images, tables, charts
└── rag_evaluation.py         # RAGAS evaluation
```

## Prerequisites

```bash
pip install sentence-transformers chromadb rank_bm25 networkx
pip install langchain langchain-community langchain-openai
pip install ragas datasets pymupdf pillow
```

## Quick Start

```bash
# Run chunking comparison
python advanced_chunking.py

# Run hybrid search demo
python hybrid_search.py

# Evaluate your RAG pipeline
python rag_evaluation.py
```

## Key Takeaways

1. **Chunking matters most** — invest in strategy before model choice
2. **Reranking is cheap insurance** — 2 lines of code, big quality boost
3. **Hybrid search wins** — BM25 + dense beats either alone
4. **Graph RAG for relationships** — when answers require connecting entities
5. **Agentic RAG for hard questions** — iterative retrieval catches what single-pass misses
6. **Evaluate everything** — RAGAS gives you metrics, not vibes
7. **Multimodal is coming** — handle tables and images now or lose information later

## Visual Deep Dive

Continue to **[Module 9 Deep Dive](deep-dive.md)** for hybrid retrieval, reranking, query transformation, graph and agentic retrieval, multimodal provenance, and layered evaluation.
