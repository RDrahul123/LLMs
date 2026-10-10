"""
Interactive PDF chat application using RAG (Retrieval-Augmented Generation).

Loads PDFs from a directory, chunks and embeds them into ChromaDB,
and provides an interactive chat loop with source page attribution.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("Error: chromadb not installed. Run: pip install chromadb")
    sys.exit(1)

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: PyMuPDF not installed. Run: pip install pymupdf")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai not installed. Run: pip install openai")
    sys.exit(1)


def extract_text_from_pdf(pdf_path: str) -> list[dict[str, Any]]:
    """Extract text from each page of a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dicts with 'text', 'page', and 'source' keys.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            pages.append({
                "text": text.strip(),
                "page": page_num + 1,
                "source": os.path.basename(pdf_path),
            })
    doc.close()
    return pages


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: The text to chunk.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap

    return chunks


def load_pdfs(
    pdf_dir: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict[str, Any]]:
    """Load and chunk all PDFs from a directory.

    Args:
        pdf_dir: Directory containing PDF files.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of document chunks with metadata.
    """
    pdf_dir_path = Path(pdf_dir)
    if not pdf_dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {pdf_dir}")

    pdf_files = list(pdf_dir_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")

    print(f"Found {len(pdf_files)} PDF(s) in {pdf_dir}")

    all_documents = []
    for pdf_path in pdf_files:
        print(f"  Processing: {pdf_path.name}")
        pages = extract_text_from_pdf(str(pdf_path))
        for page_info in pages:
            page_chunks = chunk_text(page_info["text"], chunk_size, chunk_overlap)
            for i, chunk in enumerate(page_chunks):
                all_documents.append({
                    "text": chunk,
                    "metadata": {
                        "source": page_info["source"],
                        "page": page_info["page"],
                        "chunk_index": i,
                    },
                })

    print(f"Total chunks created: {len(all_documents)}")
    return all_documents


def build_index(
    documents: list[dict[str, Any]],
    collection_name: str = "pdf_chat",
    persist_dir: str = "./chroma_db",
) -> chromadb.Collection:
    """Build a ChromaDB index from documents.

    Args:
        documents: List of document chunks with metadata.
        collection_name: Name of the ChromaDB collection.
        persist_dir: Directory for persistent storage.

    Returns:
        The ChromaDB collection with indexed documents.
    """
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Check if already indexed
    if collection.count() > 0:
        print(f"Collection '{collection_name}' already has {collection.count()} documents.")
        return collection

    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        collection.add(
            documents=[d["text"] for d in batch],
            metadatas=[d["metadata"] for d in batch],
            ids=[f"doc_{i + j}" for j in range(len(batch))],
        )
        print(f"  Indexed {min(i + batch_size, len(documents))}/{len(documents)} chunks")

    print(f"Index built: {collection.count()} documents in '{collection_name}'")
    return collection


def retrieve_context(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve relevant chunks for a query.

    Args:
        collection: The ChromaDB collection.
        query: User query string.
        n_results: Number of results to retrieve.

    Returns:
        List of retrieved chunks with metadata and scores.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    context = []
    for i in range(len(results["documents"][0])):
        context.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i] if results["distances"] else None,
        })

    return context


def generate_answer(
    client: OpenAI,
    query: str,
    context: list[dict[str, Any]],
    model: str = "gpt-4o-mini",
) -> str:
    """Generate an answer using retrieved context.

    Args:
        client: OpenAI client instance.
        query: User question.
        context: Retrieved document chunks.
        model: Model to use for generation.

    Returns:
        Generated answer string.
    """
    context_text = "\n\n---\n\n".join(
        f"[Source: {c['metadata']['source']}, Page {c['metadata']['page']}]\n{c['text']}"
        for c in context
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions based on the provided "
                "PDF documents. Always cite the source and page number for your answers. "
                "If the information is not in the provided context, say so clearly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context from PDF documents:\n\n{context_text}\n\n"
                f"Question: {query}\n\n"
                "Answer the question based on the context above. Cite sources."
            ),
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content or "No response generated."


def display_sources(context: list[dict[str, Any]]) -> None:
    """Display the source pages used for the answer."""
    print("\nSources:")
    seen = set()
    for c in context:
        source_key = (c["metadata"]["source"], c["metadata"]["page"])
        if source_key not in seen:
            seen.add(source_key)
            print(f"  - {c['metadata']['source']}, Page {c['metadata']['page']}")


def interactive_chat(collection: chromadb.Collection, model: str = "gpt-4o-mini") -> None:
    """Run an interactive chat loop.

    Args:
        collection: The indexed ChromaDB collection.
        model: Model to use for generating answers.
    """
    client = OpenAI()

    print("\n" + "=" * 60)
    print("PDF Chat - Ask questions about your documents")
    print("Commands: 'quit' to exit, 'sources' to show sources, 'clear' to clear screen")
    print("=" * 60)

    chat_history: list[dict[str, str]] = []

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if query.lower() == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            continue

        if query.lower() == "sources":
            print("\nUse 'quit' to exit, or ask a question.")
            continue

        # Retrieve relevant chunks
        context = retrieve_context(collection, query)

        if not context:
            print("\nAssistant: I couldn't find any relevant information in the documents.")
            continue

        # Generate answer
        print("\nAssistant: ", end="", flush=True)
        answer = generate_answer(client, query, context, model=model)
        print(answer)

        # Show sources
        display_sources(context)

        # Update chat history
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})

        # Keep history manageable
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chat with your PDF documents using RAG"
    )
    parser.add_argument(
        "--dir", "-d",
        default="./pdfs",
        help="Directory containing PDF files (default: ./pdfs)",
    )
    parser.add_argument(
        "--chat", "-c",
        action="store_true",
        default=True,
        help="Start interactive chat mode (default: True)",
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in characters (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap in characters (default: 200)",
    )
    parser.add_argument(
        "--persist-dir",
        default="./chroma_db",
        help="ChromaDB persistence directory (default: ./chroma_db)",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="Number of context chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only build the index without starting chat",
    )

    args = parser.parse_args()

    # Validate API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    try:
        # Load and index PDFs
        documents = load_pdfs(
            args.dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        collection = build_index(
            documents,
            persist_dir=args.persist_dir,
        )

        if args.index_only:
            print("Index built successfully.")
            return

        # Start interactive chat
        interactive_chat(collection, model=args.model)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
