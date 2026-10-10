"""
Full RAG Pipeline
==================
A complete Retrieval-Augmented Generation pipeline from scratch.

Install: pip install openai chromadb

Usage:
    python rag_pipeline.py --ingest document.txt
    python rag_pipeline.py --query "What is the main topic?"
    python rag_pipeline.py --chat
    python rag_pipeline.py --ingest document.txt --collection my_docs --top-k 5
"""

import argparse
import os
import sys
import textwrap

try:
    import chromadb
except ImportError:
    print("Install chromadb: pip install chromadb")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("Install openai: pip install openai")
    sys.exit(1)


DEFAULT_COLLECTION = "documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start = end - overlap
        if start + overlap >= len(text):
            break
    return chunks


def get_embedding(text: str, client: OpenAI, model: str = "text-embedding-3-small") -> list[float]:
    """Get embedding vector for text."""
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


def get_or_create_collection(name: str = DEFAULT_COLLECTION) -> chromadb.Collection:
    """Get or create a ChromaDB collection."""
    client = chromadb.Client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )


def ingest_pdf(pdf_path: str, collection: chroma.Collection, openai_client: OpenAI):
    """Ingest a PDF file (requires PyPDF2)."""
    try:
        import PyPDF2
    except ImportError:
        print("Install PyPDF2: pip install PyPDF2")
        return

    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

    ingest_text(text, collection, openai_client, source=os.path.basename(pdf_path))


def ingest_text(text: str, collection: chroma.Collection, openai_client: OpenAI, source: str = "unknown"):
    """Ingest text into the collection."""
    chunks = chunk_text(text)
    print(f"Ingesting {len(chunks)} chunks...")

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [f"chunk_{i+j}" for j in range(len(batch))]
        embeddings = [get_embedding(c, openai_client) for c in batch]
        metadatas = [{"source": source, "chunk_index": i + j} for j in range(len(batch))]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=batch,
            metadatas=metadatas,
        )
        print(f"  Ingested chunks {i+1} to {i+len(batch)}")

    print(f"Done! Total documents in collection: {collection.count()}")


def retrieve(query: str, collection: chroma.Collection, openai_client: OpenAI, top_k: int = 3) -> list[dict]:
    """Retrieve relevant chunks for a query."""
    q_embedding = get_embedding(query, openai_client)

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "distance": results["distances"][0][i],
        })

    return retrieved


def rag_query(question: str, collection: chroma.Collection, openai_client: OpenAI, top_k: int = 3) -> str:
    """Answer a question using RAG."""
    if collection.count() == 0:
        return "No documents in the collection. Ingest some documents first."

    docs = retrieve(question, collection, openai_client, top_k)

    context = "\n\n---\n\n".join(
        f"[Source: {d['source']}]\n{d['text']}" for d in docs
    )

    system_prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Rules:
1. Answer ONLY based on the provided context
2. If the context doesn't contain the answer, say "I don't have enough information to answer this."
3. Cite your sources when possible
4. Be concise but thorough

Context:
{context}"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
    )

    answer = response.choices[0].message.content

    # Format with sources
    sources = list(set(d["source"] for d in docs))
    sources_str = ", ".join(sources)

    return f"{answer}\n\n---\nSources: {sources_str}"


def chat_mode(collection: chroma.Collection, openai_client: OpenAI, top_k: int):
    """Interactive chat loop."""
    print("=" * 50)
    print("  RAG CHAT")
    print(f"  Documents in collection: {collection.count()}")
    print(f"  Top-K: {top_k}")
    print("  Commands: quit, exit, count, sources")
    print("=" * 50)
    print()

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if question.lower() == "count":
            print(f"Documents in collection: {collection.count()}")
            continue

        if question.lower() == "sources":
            metas = collection.get(include=["metadatas"])["metadatas"]
            sources = set(m.get("source", "unknown") for m in metas)
            print(f"Sources: {', '.join(sources)}")
            continue

        answer = rag_query(question, collection, openai_client, top_k)
        print(f"\n{answer}\n")


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument("--ingest", type=str, help="File to ingest (txt or pdf)")
    parser.add_argument("--query", type=str, help="Ask a question")
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION, help="Collection name")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    args = parser.parse_args()

    if not args.ingest and not args.query and not args.chat:
        parser.print_help()
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)

    openai_client = OpenAI()
    collection = get_or_create_collection(args.collection)

    if args.ingest:
        if args.ingest.endswith(".pdf"):
            ingest_pdf(args.ingest, collection, openai_client)
        elif args.ingest.endswith(".txt"):
            with open(args.ingest, "r", encoding="utf-8") as f:
                text = f.read()
            ingest_text(text, collection, openai_client, source=os.path.basename(args.ingest))
        else:
            print("Unsupported file type. Use .txt or .pdf")
            sys.exit(1)

    if args.query:
        answer = rag_query(args.query, collection, openai_client, args.top_k)
        print(f"\n{answer}")

    if args.chat:
        chat_mode(collection, openai_client, args.top_k)


if __name__ == "__main__":
    main()
