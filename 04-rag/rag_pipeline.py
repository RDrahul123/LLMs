"""
Module 4: Full RAG Pipeline
==============================
Ingest documents → store embeddings → query with LLM grounding.

Usage:
    python rag_pipeline.py --ingest document.pdf
    python rag_pipeline.py --query "What are the main findings?"
    python rag_pipeline.py --chat   (interactive mode)

Requirements:
    pip install openai chromadb pypdf python-dotenv
"""

import argparse
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

client = OpenAI()
chroma_client = chromadb.PersistentClient(path="./chroma_db")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 75) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def get_embedding(text: str) -> list[float]:
    """Get embedding vector for a piece of text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text.replace("\n", " "),
    )
    return response.data[0].embedding


def get_or_create_collection(name: str = "documents"):
    """Get existing collection or create new one."""
    try:
        return chroma_client.get_collection(name)
    except Exception:
        return chroma_client.create_collection(name)


# ─────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────

def ingest_pdf(filepath: str, collection_name: str = "documents") -> int:
    """Ingest a PDF into the vector database."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"📄 Reading {path.name}...")
    reader = PdfReader(filepath)
    full_text = "\n\n".join(
        page.extract_text() for page in reader.pages if page.extract_text()
    )

    print(f"✂️  Chunking text ({len(full_text)} chars)...")
    chunks = chunk_text(full_text, chunk_size=500, overlap=75)

    print(f"🔢 Embedding {len(chunks)} chunks...")
    collection = get_or_create_collection(collection_name)

    batch_size = 50
    total_added = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = [get_embedding(chunk) for chunk in batch]
        ids = [f"{path.stem}_chunk_{i + j}" for j in range(len(batch))]
        metadatas = [{"source": path.name, "chunk_index": i + j} for j in range(len(batch))]

        collection.add(
            documents=batch,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        total_added += len(batch)
        print(f"  Progress: {total_added}/{len(chunks)} chunks", end="\r")

    print(f"\n✅ Ingested {total_added} chunks from '{path.name}'")
    return total_added


def ingest_text(text: str, source_name: str = "manual", collection_name: str = "documents") -> int:
    """Ingest raw text into the vector database."""
    chunks = chunk_text(text)
    collection = get_or_create_collection(collection_name)

    embeddings = [get_embedding(chunk) for chunk in chunks]
    ids = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )
    print(f"✅ Ingested {len(chunks)} chunks from '{source_name}'")
    return len(chunks)


# ─────────────────────────────────────────────
# Retrieval + Generation
# ─────────────────────────────────────────────

def retrieve(query: str, top_k: int = 4, collection_name: str = "documents") -> list[dict]:
    """Retrieve the most relevant chunks for a query."""
    collection = get_or_create_collection(collection_name)
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "score": round(1 - dist, 4),  # convert distance to similarity
        })

    return retrieved


def rag_query(
    question: str,
    top_k: int = 4,
    collection_name: str = "documents",
    show_sources: bool = True,
) -> str:
    """Run a full RAG query: retrieve context → generate answer."""

    # 1. Retrieve relevant chunks
    chunks = retrieve(question, top_k=top_k, collection_name=collection_name)

    if not chunks:
        return "No documents found. Please ingest some documents first."

    # 2. Build context string
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[Source {i}: {chunk['source']}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    # 3. Generate answer
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions based on provided context. "
                    "Answer ONLY based on the context given. "
                    "If the answer is not in the context, say: 'I couldn't find this in the provided documents.' "
                    "Always cite which source you used (e.g., '[Source 1]')."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.1,  # low temp for factual answers
    )

    answer = response.choices[0].message.content

    if show_sources:
        sources_str = "\n".join(
            f"  • {c['source']} (relevance: {c['score']})" for c in chunks
        )
        answer += f"\n\n📚 Sources consulted:\n{sources_str}"

    return answer


# ─────────────────────────────────────────────
# Interactive Chat Mode
# ─────────────────────────────────────────────

def chat_mode(collection_name: str = "documents"):
    """Interactive chat loop with RAG."""
    print("\n💬 RAG Chat Mode — ask questions about your documents.")
    print("Type 'quit' to exit, 'sources on/off' to toggle source display.\n")

    show_sources = True

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if question.lower() == "sources on":
            show_sources = True
            print("📚 Sources display: ON")
            continue
        if question.lower() == "sources off":
            show_sources = False
            print("📚 Sources display: OFF")
            continue

        print("\nAI: ", end="", flush=True)
        answer = rag_query(question, show_sources=show_sources, collection_name=collection_name)
        print(answer)
        print()


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    parser.add_argument("--ingest", metavar="FILE", help="PDF file to ingest")
    parser.add_argument("--query", metavar="QUESTION", help="Single question to answer")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat mode")
    parser.add_argument("--collection", default="documents", help="Collection name (default: documents)")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")

    args = parser.parse_args()

    if args.ingest:
        ingest_pdf(args.ingest, collection_name=args.collection)
    elif args.query:
        answer = rag_query(args.query, top_k=args.top_k, collection_name=args.collection)
        print(f"\nAnswer:\n{answer}")
    elif args.chat:
        chat_mode(collection_name=args.collection)
    else:
        parser.print_help()
