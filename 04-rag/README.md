# Module 4 — RAG (Retrieval Augmented Generation) 🔵

> **Level:** Intermediate | **Time:** ~3 hours | **Code:** Python

---

## The Problem RAG Solves

LLMs have two critical limitations:
1. **Knowledge cutoff** — they don't know recent events
2. **No access to your data** — they can't read your PDFs, docs, or databases

**RAG** fixes this by giving the LLM relevant information at query time.

---

## How RAG Works

```
                    ┌─────────────────────────────────┐
                    │         YOUR DOCUMENTS          │
                    │  (PDFs, URLs, docs, databases)  │
                    └──────────────┬──────────────────┘
                                   │
                            1. CHUNK TEXT
                                   │
                            2. EMBED CHUNKS
                            (convert to vectors)
                                   │
                            3. STORE IN VECTOR DB
                                   │
User Query ──► 4. EMBED QUERY ──► 5. SIMILARITY SEARCH
                                   │
                         6. RETRIEVE TOP K CHUNKS
                                   │
                    ┌──────────────▼──────────────────┐
                    │   LLM PROMPT = Query + Chunks   │
                    └──────────────┬──────────────────┘
                                   │
                            7. LLM GENERATES
                            GROUNDED ANSWER
                                   │
                              Final Answer
```

---

## Key Concepts

### 📦 Chunking
Splitting documents into smaller pieces so they fit in the LLM's context window and can be retrieved precisely.

```python
# Simple chunking by character count
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # overlap prevents cutting mid-sentence
    return chunks
```

**Chunk size tips:**
- Too small (< 100 chars) → loses context
- Too large (> 2000 chars) → retrieves too much noise
- Sweet spot: 200–800 characters with 10–20% overlap

---

### 🔢 Embeddings
Converting text into numerical vectors that capture meaning.

```python
from openai import OpenAI

client = OpenAI()

def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding  # list of 1536 floats

# Similar texts → similar vectors
embed("I love dogs")      # [0.12, -0.34, 0.89, ...]
embed("I adore puppies")  # [0.11, -0.31, 0.87, ...]  ← very close!
embed("The stock rose")   # [0.92, 0.14, -0.22, ...]  ← far away
```

---

### 🗄️ Vector Databases
Store embeddings and retrieve the most similar ones to a query.

| Database | Best For | Free Tier |
|---|---|---|
| **FAISS** | Local, in-memory | ✅ Always free |
| **ChromaDB** | Local, persistent | ✅ Always free |
| **Pinecone** | Production, cloud | ✅ Limited |
| **Weaviate** | Enterprise | ✅ Limited |
| **pgvector** | PostgreSQL users | ✅ Self-hosted |

---

## Build a RAG Pipeline from Scratch

### Step 1: Install Dependencies
```bash
pip install openai chromadb pypdf python-dotenv
```

### Step 2: Ingest Documents
```python
import chromadb
from openai import OpenAI
from pypdf import PdfReader

client = OpenAI()
chroma = chromadb.Client()
collection = chroma.create_collection("my_docs")

def ingest_pdf(filepath: str):
    # Extract text
    reader = PdfReader(filepath)
    text = "\n".join(page.extract_text() for page in reader.pages)

    # Chunk
    chunks = chunk_text(text, chunk_size=500, overlap=50)

    # Embed and store
    embeddings = [
        client.embeddings.create(model="text-embedding-3-small", input=chunk).data[0].embedding
        for chunk in chunks
    ]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print(f"✅ Ingested {len(chunks)} chunks from {filepath}")
```

### Step 3: Query
```python
def rag_query(question: str, top_k: int = 3) -> str:
    # Embed the question
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    # Retrieve relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    context = "\n\n---\n\n".join(results["documents"][0])

    # Ask LLM with context
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer questions based ONLY on the provided context. "
                    "If the answer is not in the context, say 'I don't know based on the provided documents.'"
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )

    return response.choices[0].message.content
```

### Step 4: Run It
```python
ingest_pdf("your_document.pdf")

answer = rag_query("What are the main conclusions of this report?")
print(answer)
```

---

## RAG Best Practices

### ✅ Do This
- **Rerank results** — use a cross-encoder to rerank retrieved chunks for better precision
- **Add metadata** — store source, page number, date with chunks for citations
- **Hybrid search** — combine vector search with keyword search (BM25)
- **Evaluate** — use frameworks like RAGAS to measure retrieval quality

### ❌ Avoid This
- Chunks that are too large (model gets confused by irrelevant info)
- No overlap between chunks (misses context at chunk boundaries)
- Embedding the question differently than the documents
- Trusting retrieval blindly without checking relevance scores

---

## 📁 Files in This Module

| File | Description |
|---|---|
| [`rag_pipeline.py`](./rag_pipeline.py) | Full RAG pipeline with ChromaDB |
| [`chat_with_pdf.py`](./chat_with_pdf.py) | Interactive "chat with your PDF" app |
| [`embeddings_demo.py`](./embeddings_demo.py) | Visualizing what embeddings are |

---

## ✅ What's Next?

👉 **[Module 5 — Fine-Tuning](../05-fine-tuning/README.md)**
