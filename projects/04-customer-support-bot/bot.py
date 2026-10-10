#!/usr/bin/env python3
"""Customer Support Bot — RAG with intent classification, sentiment, and escalation."""

import hashlib
import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter

import chromadb
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
KB_DIR = Path(os.getenv("KB_DIR", "knowledge_base"))
PERSIST_DIR = Path("_chroma_support")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------------------------------------------------------------------------
# Knowledge base loading
# ---------------------------------------------------------------------------

INTENTS = ["billing", "technical", "general", "complaint"]
SENTIMENTS = ["positive", "neutral", "negative", "angry"]


def load_knowledge_base(path: Path) -> list[dict]:
    """Load Q&A pairs from text files in the knowledge base directory."""
    entries = []
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        # Create a sample KB
        sample = path / "faq.txt"
        sample.write_text(
            "Q: How do I reset my password?\n"
            "A: Go to Settings > Security > Reset Password. You'll receive a reset email.\n\n"
            "Q: What is your refund policy?\n"
            "A: Full refunds within 30 days. Email support@company.com with your order ID.\n\n"
            "Q: How do I contact support?\n"
            "A: Email support@company.com or call 1-800-555-0123, Mon-Fri 9AM-5PM.\n\n"
            "Q: How do I upgrade my plan?\n"
            "A: Go to Settings > Billing > Change Plan. You can upgrade instantly.\n\n"
            "Q: My app crashes on startup\n"
            "A: Try clearing cache (Settings > Storage > Clear Cache). If it persists, reinstall.\n\n"
            "Q: I was charged twice\n"
            "A: This is rare. Email billing@company.com with your receipt and we'll refund immediately.\n\n"
            "Q: The service is too slow\n"
            "A: Check status.company.com for outages. Try switching regions in Settings > Network.\n\n"
            "Q: I want to cancel my subscription\n"
            "A: Go to Settings > Billing > Cancel. You'll keep access until the billing period ends.\n\n"
            "Q: My data is not syncing\n"
            "A: Ensure you're online, then go to Settings > Sync > Force Sync.\n\n"
            "Q: Do you offer API access?\n"
            "A: Yes, Pro plans and above include API access. See docs.company.com/api.\n",
            encoding="utf-8",
        )
        console.print(f"[yellow]Created sample FAQ at {sample.resolve()}[/yellow]")

    for f in sorted(path.iterdir()):
        if f.suffix.lower() in (".txt", ".md"):
            text = f.read_text(encoding="utf-8", errors="replace")
            # Split on double newlines for Q&A pairs
            pairs = re.split(r"\n\n+", text)
            for pair in pairs:
                pair = pair.strip()
                if pair:
                    entries.append({"content": pair, "source": f.name})
    return entries


def ingest_kb(collection, entries: list[dict]):
    existing = collection.get()
    if existing and existing["ids"]:
        console.print(f"[dim]Knowledge base already indexed ({len(existing['ids'])} entries).[/dim]")
        return

    ids = [f"kb_{i}" for i in range(len(entries))]
    documents = [e["content"] for e in entries]
    metadatas = [{"source": e["source"]} for e in entries]

    if client and entries:
        for batch_start in range(0, len(entries), 100):
            batch = documents[batch_start:batch_start + 100]
            batch_ids = ids[batch_start:batch_start + 100]
            batch_meta = metadatas[batch_start:batch_start + 100]
            embs = client.embeddings.create(input=batch, model=EMBED_MODEL).data
            vecs = [e.embedding for e in embs]
            collection.add(ids=batch_ids, documents=batch, metadatas=batch_meta, embeddings=vecs)
    elif entries:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

    console.print(f"[green]Indexed {len(entries)} knowledge base entries.[/green]")

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_intent(text: str) -> str:
    """Rule-based intent classification with LLM fallback."""
    lower = text.lower()
    billing_kw = ["charge", "refund", "bill", "payment", "subscription", "cancel", "plan", "invoice", "cost"]
    tech_kw = ["crash", "bug", "error", "slow", "sync", "install", "update", "broken", "not working", "fail"]
    complaint_kw = ["terrible", "awful", "worst", "hate", "angry", "furious", "unacceptable", "scam", "horrible"]

    scores = {intent: 0 for intent in INTENTS}
    for kw in billing_kw:
        if kw in lower:
            scores["billing"] += 1
    for kw in tech_kw:
        if kw in lower:
            scores["technical"] += 1
    for kw in complaint_kw:
        if kw in lower:
            scores["complaint"] += 1

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "general"
    return best


def detect_sentiment(text: str) -> str:
    """Rule-based sentiment detection."""
    lower = text.lower()
    angry_kw = ["furious", "unacceptable", "hate", "worst", "scam", "angry", "disgusted", "ridiculous"]
    negative_kw = ["bad", "poor", "frustrated", "annoyed", "disappointed", "slow", "broken", "terrible"]
    positive_kw = ["great", "love", "excellent", "amazing", "thank", "perfect", "wonderful", "happy"]

    angry_score = sum(1 for kw in angry_kw if kw in lower)
    negative_score = sum(1 for kw in negative_kw if kw in lower)
    positive_score = sum(1 for kw in positive_kw if kw in lower)

    if angry_score > 0:
        return "angry"
    if negative_score > positive_score:
        return "negative"
    if positive_score > negative_score:
        return "positive"
    return "neutral"

# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------

def should_escalate(intent: str, sentiment: str, history: list[dict]) -> bool:
    """Determine if the conversation should be escalated to a human."""
    if sentiment == "angry":
        return True
    if intent == "complaint":
        return True
    # Escalate after 3+ negative turns
    recent = history[-6:]
    negative_count = sum(1 for m in recent if m.get("sentiment") in ("negative", "angry"))
    if negative_count >= 3:
        return True
    return False

# ---------------------------------------------------------------------------
# RAG + Response
# ---------------------------------------------------------------------------

def retrieve_kb(query: str, collection, k: int = 3) -> list[dict]:
    if client:
        q_emb = client.embeddings.create(input=[query], model=EMBED_MODEL).data[0].embedding
    else:
        q_emb = [0.0] * 1536
    results = collection.query(query_embeddings=[q_emb], n_results=k)
    docs = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        docs.append({"content": doc, "source": meta.get("source", "unknown")})
    return docs


SENTIMENT_INSTRUCTIONS = {
    "positive": "The customer is happy. Be warm, acknowledge their satisfaction, and offer additional help.",
    "neutral": "Respond professionally and clearly. Be helpful and direct.",
    "negative": "The customer is frustrated. Be empathetic, apologetic, and offer concrete solutions.",
    "angry": "The customer is very upset. Acknowledge their frustration sincerely, apologize, and offer immediate escalation options.",
}

SYSTEM_PROMPT = """\
You are a customer support assistant for a SaaS product. Follow these rules:
1. Only answer based on the knowledge base context provided.
2. If you don't have enough information, say so honestly.
3. Be concise — customers want quick answers.
4. Adapt your tone to the customer's sentiment.
5. For billing issues, always reference their account.
6. For technical issues, provide step-by-step instructions.
7. Never make up information not in the knowledge base.
"""


def generate_response(
    query: str,
    contexts: list[dict],
    intent: str,
    sentiment: str,
    history: list[dict],
) -> str:
    context_block = "\n\n---\n\n".join(
        f"[{c['source']}]\n{c['content']}" for c in contexts
    )
    sentiment_instruction = SENTIMENT_INSTRUCTIONS.get(sentiment, "")

    messages = [
        {
            "role": "system",
            "content": (
                f"{SYSTEM_PROMPT}\n\n"
                f"## Customer Sentiment: {sentiment}\n{sentiment_instruction}\n\n"
                f"## Detected Intent: {intent}\n\n"
                f"## Knowledge Base Context\n\n{context_block}"
            ),
        },
        *history[-8:],
        {"role": "user", "content": query},
    ]

    if not client:
        return "[No API key — showing raw context]\n\n" + context_block

    resp = client.chat.completions.create(model=LLM_MODEL, messages=messages, temperature=0.4)
    return resp.choices[0].message.content

# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@dataclass
class Analytics:
    total_messages: int = 0
    intents: Counter = field(default_factory=Counter)
    sentiments: Counter = field(default_factory=Counter)
    escalations: int = 0
    resolved: int = 0

    def record(self, intent: str, sentiment: str, escalated: bool):
        self.total_messages += 1
        self.intents[intent] += 1
        self.sentiments[sentiment] += 1
        if escalated:
            self.escalations += 1
        else:
            self.resolved += 1

    def display(self):
        table = Table(title="Session Analytics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total messages", str(self.total_messages))
        table.add_row("Intents", ", ".join(f"{k}: {v}" for k, v in self.intents.most_common()))
        table.add_row("Sentiments", ", ".join(f"{k}: {v}" for k, v in self.sentiments.most_common()))
        table.add_row("Escalated", str(self.escalations))
        table.add_row("Resolved", str(self.resolved))
        console.print(table)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = chroma_client.get_or_create_collection(
        name="support_kb", metadata={"hnsw:space": "cosine"}
    )

    console.print(Panel("[bold]Customer Support Bot[/bold]", border_style="blue"))

    entries = load_knowledge_base(KB_DIR)
    ingest_kb(collection, entries)

    history: list[dict] = []
    analytics = Analytics()

    console.print("\n[dim]Type /analytics, /history, /reset, or /quit. Otherwise, ask your question.[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold cyan]👤 You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # Commands
        cmd = user_input.lower()
        if cmd == "/quit":
            console.print("[dim]Goodbye![/dim]")
            break
        if cmd == "/reset":
            history.clear()
            analytics = Analytics()
            console.print("[yellow]Conversation reset.[/yellow]\n")
            continue
        if cmd == "/history":
            if not history:
                console.print("[dim]No history yet.[/dim]")
            for msg in history:
                role = "👤" if msg["role"] == "user" else "🤖"
                console.print(f"{role} {msg['content'][:300]}")
            console.print()
            continue
        if cmd == "/analytics":
            analytics.display()
            console.print()
            continue

        # Classify
        intent = classify_intent(user_input)
        sentiment = detect_sentiment(user_input)

        intent_colors = {"billing": "yellow", "technical": "cyan", "general": "white", "complaint": "red"}
        sent_colors = {"positive": "green", "neutral": "white", "negative": "yellow", "angry": "red"}
        console.print(f"  [dim]Intent: [{intent_colors[intent]}]{intent}[/{intent_colors[intent]}] | "
                       f"Sentiment: [{sent_colors[sentiment]}]{sentiment}[/{sent_colors[sentiment]}][/]")

        # Retrieve
        contexts = retrieve_kb(user_input, collection)

        # Check escalation
        escalated = should_escalate(intent, sentiment, history)
        analytics.record(intent, sentiment, escalated)

        if escalated:
            console.print(Panel(
                "[bold yellow]⚠️  Escalating to Human Agent[/bold yellow]\n\n"
                "I understand this is frustrating. Let me connect you with a human agent "
                "who can help resolve this immediately. Please hold on.\n\n"
                "📧 support@company.com | 📞 1-800-555-0123",
                border_style="yellow",
            ))
            history.append({"role": "user", "content": user_input, "intent": intent, "sentiment": sentiment})
            history.append({"role": "assistant", "content": "[Escalated to human]", "intent": intent, "sentiment": sentiment})
            console.print()
            continue

        # Generate response
        response = generate_response(user_input, contexts, intent, sentiment, history)

        history.append({"role": "user", "content": user_input, "intent": intent, "sentiment": sentiment})
        history.append({"role": "assistant", "content": response, "intent": intent, "sentiment": sentiment})

        console.print()
        console.print(Markdown(response))
        console.print()


if __name__ == "__main__":
    main()
