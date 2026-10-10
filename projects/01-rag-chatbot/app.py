import os
import hashlib
import uuid
from pathlib import Path
from typing import Optional

import chromadb
import fitz  # PyMuPDF
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from openai import OpenAI

console = Console()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))
DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs"))
PERSIST_DIR = Path("_chroma_db")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def _pdf_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def extract_text_from_pdf(path: Path) -> str:
    doc = fitz.open(str(path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]

# ---------------------------------------------------------------------------
# ChromaDB setup
# ---------------------------------------------------------------------------

chroma_client = chromadb.PersistentClient(path=str(PERSIST_DIR))
collection = chroma_client.get_or_create_collection(
    name="rag_chatbot",
    metadata={"hnsw:space": "cosine"},
)


def ingest_pdfs():
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        console.print(f"[yellow]No docs/ directory found. Created {DOCS_DIR.resolve()}[/yellow]")
        console.print("[yellow]Drop some PDFs in there and restart.[/yellow]")
        return

    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        console.print("[yellow]No PDF files found in docs/.[/yellow]")
        return

    table = Table(title="Ingesting PDFs")
    table.add_column("File", style="cyan")
    table.add_column("Chunks", justify="right", style="green")

    for pdf in pdf_files:
        pdf_id = _pdf_hash(pdf)
        existing = collection.get(where={"pdf_id": pdf_id})
        if existing and existing["ids"]:
            table.add_row(pdf.name, f"[dim]skipped (already indexed)[/dim]")
            continue

        text = extract_text_from_pdf(pdf)
        chunks = chunk_text(text)
        ids = [f"{pdf_id}_{i}" for i in range(len(chunks))]
        metadatas = [{"pdf_id": pdf_id, "source": pdf.name, "chunk_idx": i} for i in range(len(chunks))]

        # Embed in batches of 100
        for batch_start in range(0, len(chunks), 100):
            batch = chunks[batch_start : batch_start + 100]
            batch_ids = ids[batch_start : batch_start + 100]
            batch_meta = metadatas[batch_start : batch_start + 100]
            if client:
                embeddings = client.embeddings.create(input=batch, model=EMBED_MODEL).data
                vecs = [e.embedding for e in embeddings]
            else:
                vecs = [[0.0] * 1536 for _ in batch]  # placeholder
            collection.add(ids=batch_ids, documents=batch, metadatas=batch_meta, embeddings=vecs)

        table.add_row(pdf.name, str(len(chunks)))

    console.print(table)

# ---------------------------------------------------------------------------
# Retrieval & generation
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


# In-memory conversation store  {session_id: [{"role": ..., "content": ...}]}
sessions: dict[str, list[dict]] = {}


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    if client:
        q_emb = client.embeddings.create(input=[query], model=EMBED_MODEL).data[0].embedding
    else:
        q_emb = [0.0] * 1536
    results = collection.query(query_embeddings=[q_emb], n_results=k)
    docs = []
    for i, doc in enumerate(results["documents"][0]):
        docs.append({
            "content": doc,
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "score": results["distances"][0][i] if results.get("distances") else 0,
        })
    return docs


def generate_answer(query: str, contexts: list[dict], history: list[dict]) -> str:
    context_block = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['content']}" for c in contexts
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions based on provided documents. "
                "Always cite which source(s) you used. If the documents don't contain the answer, say so.\n\n"
                f"## Retrieved Documents\n\n{context_block}"
            ),
        },
        *history[-10:],  # last 10 turns
        {"role": "user", "content": query},
    ]
    if client:
        resp = client.chat.completions.create(model=LLM_MODEL, messages=messages, temperature=0.3)
        return resp.choices[0].message.content
    return (
        "[No OPENAI_API_KEY set — returning raw context only]\n\n"
        + context_block
    )

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="RAG Chatbot")


@app.on_event("startup")
async def startup():
    ingest_plibs = ingest_pdfs()
    console.print(Panel("[bold green]RAG Chatbot ready![/bold green]", border_style="green"))
    console.print("Open [cyan]http://127.0.0.1:8000[/cyan] in your browser.\n")


@app.post("/chat")
async def chat(req: ChatRequest):
    sid = req.session_id or str(uuid.uuid4())
    history = sessions.setdefault(sid, [])

    console.print(f"\n[bold cyan]User:[/bold cyan] {req.message}")

    sources = retrieve(req.message)
    answer = generate_answer(req.message, sources, history)

    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": answer})

    console.print(f"[bold green]Assistant:[/bold green] {answer}")
    table = Table(title="Sources")
    table.add_column("Document", style="cyan")
    table.add_column("Score", justify="right")
    for s in sources:
        table.add_row(s["source"], f"{s['score']:.4f}")
    console.print(table)

    return JSONResponse({"answer": answer, "sources": sources, "session_id": sid})


@app.get("/history")
async def history(session_id: str):
    return JSONResponse(sessions.get(session_id, []))


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG Chatbot</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;display:flex;justify-content:center;align-items:center;min-height:100vh}
  #chat{width:720px;max-height:90vh;display:flex;flex-direction:column;border:1px solid #334155;border-radius:12px;overflow:hidden;background:#1e293b}
  #header{padding:16px 20px;background:#0f172a;font-size:18px;font-weight:700;border-bottom:1px solid #334155}
  #messages{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:12px}
  .msg{padding:12px 16px;border-radius:8px;max-width:85%;line-height:1.5}
  .user{background:#2563eb;align-self:flex-end;color:#fff}
  .assistant{background:#334155;align-self:flex-start}
  .sources{font-size:12px;color:#94a3b8;margin-top:6px}
  #input-row{display:flex;gap:8px;padding:12px 16px;border-top:1px solid #334155;background:#0f172a}
  #input-row input{flex:1;padding:10px 14px;border:1px solid #475569;border-radius:8px;background:#1e293b;color:#e2e8f0;font-size:14px}
  #input-row button{padding:10px 20px;background:#2563eb;color:#fff;border:none;border-radius:8px;cursor:pointer;font-weight:600}
  #input-row button:hover{background:#1d4ed8}
</style>
</head>
<body>
<div id="chat">
  <div id="header">RAG Chatbot</div>
  <div id="messages"></div>
  <div id="input-row">
    <input id="inp" placeholder="Ask about your PDFs..." autofocus>
    <button onclick="send()">Send</button>
  </div>
</div>
<script>
let sid = null;
const msgs = document.getElementById('messages');
const inp = document.getElementById('inp');

inp.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });

async function send() {
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  addMsg(text, 'user');
  const res = await fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: text, session_id: sid})
  });
  const data = await res.json();
  sid = data.session_id;
  const srcNames = [...new Set(data.sources.map(s => s.source))];
  addMsg(data.answer + (srcNames.length ? '<div class="sources">Sources: ' + srcNames.join(', ') + '</div>' : ''), 'assistant');
}

function addMsg(html, role) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.innerHTML = html;
  msgs.appendChild(d);
  msgs.scrollTop = msgs.scrollHeight;
}
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


if __name__ == "__main__":
    import uvicorn
    console.print(Panel("[bold]Starting RAG Chatbot[/bold]", border_style="blue"))
    uvicorn.run(app, host="127.0.0.1", port=8000)
