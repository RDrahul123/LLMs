"""
Agent with Memory Systems

Demonstrates three memory layers for AI agents:
1. Short-term memory (conversation buffer)
2. Summary memory (compressed conversation history)
3. Long-term memory (vector store for facts learned)

The agent remembers user preferences across sessions and retrieves
relevant context to inform its responses.

Requirements:
    pip install openai chromadb

Usage:
    export OPENAI_API_KEY="your-key"
    python agent_with_memory.py
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Short-Term Memory (Conversation Buffer)
# ---------------------------------------------------------------------------


class ConversationBuffer:
    """Maintains the full conversation history within a session.

    This is the simplest form of memory — just a list of messages.
    Limited by the LLM's context window.
    """

    def __init__(self, max_messages: int = 50) -> None:
        self.messages: list[dict[str, str]] = []
        self.max_messages = max_messages

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_messages(self) -> list[dict[str, str]]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()

    def get_context(self) -> str:
        if not self.messages:
            return ""
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.messages)

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        return f"ConversationBuffer(messages={len(self.messages)})"


# ---------------------------------------------------------------------------
# Summary Memory (Compressed History)
# ---------------------------------------------------------------------------


class SummaryMemory:
    """Compresses old conversation messages into summaries.

    When the buffer gets too large, older messages are summarized to
    free up context space while preserving important information.
    """

    def __init__(self, max_buffer_messages: int = 20) -> None:
        self.buffer = ConversationBuffer(max_messages=max_buffer_messages)
        self.summaries: list[str] = []
        self.total_messages_processed: int = 0

    def add(self, role: str, content: str) -> None:
        self.buffer.add(role, content)
        self.total_messages_processed += 1

    def get_context(self) -> str:
        parts: list[str] = []
        if self.summaries:
            parts.append("Previous conversation summary:")
            for s in self.summaries:
                parts.append(f"  - {s}")
        if self.buffer.messages:
            parts.append("\nRecent messages:")
            for m in self.buffer.messages:
                parts.append(f"  {m['role']}: {m['content']}")
        return "\n".join(parts)

    def compress(self, llm_client: Any, model: str = "gpt-4o-mini") -> None:
        """Compress old messages into a summary using the LLM."""
        if len(self.buffer.messages) <= 10:
            return

        old_messages = self.buffer.messages[:len(self.buffer.messages) // 2]
        conversation_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in old_messages
        )

        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize this conversation in 2-3 sentences. "
                            "Focus on key facts, decisions, and user preferences. "
                            "Be concise but preserve important details."
                        ),
                    },
                    {"role": "user", "content": conversation_text},
                ],
                temperature=0,
                max_tokens=200,
            )
            summary = response.choices[0].message.content
            if summary:
                self.summaries.append(summary)
                self.buffer.messages = self.buffer.messages[len(old_messages):]
        except Exception as e:
            print(f"  [SummaryMemory] Compression failed: {e}")

    def clear(self) -> None:
        self.buffer.clear()
        self.summaries.clear()
        self.total_messages_processed = 0

    def __repr__(self) -> str:
        return (
            f"SummaryMemory(summaries={len(self.summaries)}, "
            f"buffer={len(self.buffer)}, processed={self.total_messages_processed})"
        )


# ---------------------------------------------------------------------------
# Long-Term Memory (Vector Store)
# ---------------------------------------------------------------------------


class LongTermMemory:
    """Persistent memory backed by a vector database.

    Stores facts, preferences, and important information learned during
    conversations. Uses embedding similarity for retrieval.
    """

    def __init__(self, collection_name: str = "agent_memory") -> None:
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._init_store()

    def _init_store(self) -> None:
        try:
            import chromadb

            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except ImportError:
            print("  [LongTermMemory] chromadb not installed. Using in-memory fallback.")
            self._client = None
            self._collection = None
            self._fallback_store: list[dict[str, str]] = []

    def store(self, fact: str, category: str = "general", metadata: dict[str, Any] | None = None) -> None:
        """Store a fact in long-term memory."""
        meta = {"category": category, "timestamp": str(time.time())}
        if metadata:
            meta.update(metadata)

        doc_id = f"doc_{self.count() + 1}_{int(time.time())}"

        if self._collection:
            self._collection.add(
                documents=[fact],
                metadatas=[meta],
                ids=[doc_id],
            )
        else:
            self._fallback_store.append({
                "id": doc_id,
                "document": fact,
                "metadata": meta,
            })

    def retrieve(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Retrieve relevant facts from long-term memory."""
        if self._collection:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(n_results, self.count()),
            )
            retrieved = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    retrieved.append({"document": doc, "metadata": meta})
            return retrieved
        else:
            relevant = []
            query_lower = query.lower()
            for item in self._fallback_store:
                if any(
                    word in item["document"].lower()
                    for word in query_lower.split()
                ):
                    relevant.append({
                        "document": item["document"],
                        "metadata": item["metadata"],
                    })
            return relevant[:n_results]

    def count(self) -> int:
        if self._collection:
            return self._collection.count()
        return len(self._fallback_store)

    def clear(self) -> None:
        if self._collection:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            self._fallback_store.clear()

    def get_all(self) -> list[dict[str, Any]]:
        if self._collection:
            results = self._collection.get()
            retrieved = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"]):
                    meta = results["metadatas"][i] if results["metadatas"] else {}
                    retrieved.append({"document": doc, "metadata": meta})
            return retrieved
        return [
            {"document": item["document"], "metadata": item["metadata"]}
            for item in self._fallback_store
        ]

    def __repr__(self) -> str:
        return f"LongTermMemory(facts={self.count()})"


# ---------------------------------------------------------------------------
# Memory-Augmented Agent
# ---------------------------------------------------------------------------


class MemoryAgent:
    """Agent with three layers of memory.

    Combines short-term conversation buffer, summary compression,
    and long-term fact retrieval for context-aware responses.
    """

    SYSTEM_PROMPT = (
        "You are a helpful personal assistant with excellent memory. "
        "You remember past conversations and user preferences. "
        "When relevant past information is available, use it to personalize "
        "your responses. Be concise and helpful."
    )

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_buffer: int = 20,
        verbose: bool = True,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.verbose = verbose

        self.short_term = ConversationBuffer(max_messages=50)
        self.summary_memory = SummaryMemory(max_buffer_messages=max_buffer)
        self.long_term = LongTermMemory(collection_name="agent_ltm")

        self.session_id: str = f"session_{int(time.time())}"

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"  {message}")

    def _extract_facts(self, user_message: str) -> list[str]:
        """Extract important facts or preferences from a message."""
        facts: list[str] = []

        preference_patterns = [
            "I like", "I prefer", "I love", "I hate", "I want",
            "My favorite", "I am", "I'm a", "I work", "I live",
            "Please remember", "Note that", "Keep in mind",
        ]

        for pattern in preference_patterns:
            if pattern.lower() in user_message.lower():
                facts.append(user_message.strip())
                break

        return facts

    def _build_context(self) -> str:
        """Build the full context from all memory layers."""
        parts: list[str] = []

        long_term_results = self.long_term.retrieve(
            self.short_term.get_context() if self.short_term.messages else "user context"
        )
        if long_term_results:
            parts.append("Remembered about the user:")
            for result in long_term_results:
                parts.append(f"  - {result['document']}")

        summary_ctx = self.summary_memory.get_context()
        if summary_ctx:
            parts.append(f"\n{summary_ctx}")

        recent = self.short_term.get_messages()[-10:]
        if recent:
            parts.append("\nRecent conversation:")
            for m in recent:
                parts.append(f"  {m['role']}: {m['content']}")

        return "\n".join(parts)

    def chat(self, user_message: str) -> str:
        """Send a message and get a response.

        Args:
            user_message: The user's message.

        Returns:
            The assistant's response.
        """
        self._log(f"\nUser: {user_message}")

        facts = self._extract_facts(user_message)
        for fact in facts:
            self.long_term.store(fact, category="user_preference")
            self._log(f"  [Memory] Stored fact: {fact[:60]}...")

        self.short_term.add("user", user_message)
        self.summary_memory.add("user", user_message)

        context = self._build_context()

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]

        if context:
            messages.append({
                "role": "system",
                "content": f"Context from memory:\n{context}",
            })

        messages.extend(self.short_term.get_messages())

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500,
            )
            assistant_reply = response.choices[0].message.content or ""
        except Exception as e:
            assistant_reply = f"Error: {e}"

        self.short_term.add("assistant", assistant_reply)
        self.summary_memory.add("assistant", assistant_reply)

        if self.summary_memory.total_messages_processed % 15 == 0:
            self.summary_memory.compress(self.client, self.model)
            self._log("  [Memory] Compressed conversation history")

        self._log(f"  Assistant: {assistant_reply[:150]}...")
        return assistant_reply

    def new_session(self) -> None:
        """Start a new conversation session."""
        self._log(f"\n[New Session] Previous session: {self.session_id}")
        self.short_term.clear()
        self.summary_memory.clear()
        self.session_id = f"session_{int(time.time())}"
        self._log(f"[New Session] Started: {self.session_id}")

    def get_memory_stats(self) -> dict[str, Any]:
        """Get statistics about all memory systems."""
        return {
            "session_id": self.session_id,
            "short_term_messages": len(self.short_term),
            "summary_count": len(self.summary_memory.summaries),
            "long_term_facts": self.long_term.count(),
            "total_processed": self.summary_memory.total_messages_processed,
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def demo_memory_layers() -> None:
    """Demonstrate each memory layer independently."""
    print("=" * 60)
    print("MEMORY LAYERS DEMO")
    print("=" * 60)

    print("\n--- Short-Term Memory (Buffer) ---")
    buffer = ConversationBuffer(max_messages=5)
    for i in range(7):
        role = "user" if i % 2 == 0 else "assistant"
        buffer.add(role, f"Message {i + 1}")
    print(f"  Messages after 7 adds (max=5): {len(buffer)}")
    print(f"  First message: {buffer.messages[0]}")

    print("\n--- Long-Term Memory (Vector Store) ---")
    ltm = LongTermMemory(collection_name="demo_memory")
    ltm.store("User prefers Python over JavaScript", category="preference")
    ltm.store("User works as a data scientist", category="work")
    ltm.store("User's favorite color is blue", category="preference")
    ltm.store("User is learning machine learning", category="learning")

    results = ltm.retrieve("programming language preference")
    print(f"  Stored facts: {ltm.count()}")
    print(f"  Retrieved for 'programming language preference':")
    for r in results:
        print(f"    - {r['document']}")


def demo_session_persistence() -> None:
    """Demonstrate memory persistence across sessions."""
    print("\n" + "=" * 60)
    print("SESSION PERSISTENCE DEMO")
    print("=" * 60)

    try:
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Skipping — OPENAI_API_KEY not set.")
            return

        agent = MemoryAgent(verbose=True)

        print("\n--- Session 1: Establishing Preferences ---")
        agent.chat("Hi! I'm a data scientist who prefers Python.")
        agent.chat("I love using pandas for data analysis.")
        agent.chat("My favorite visualization library is matplotlib.")

        stats = agent.get_memory_stats()
        print(f"\n  Stats: {json.dumps(stats, indent=2)}")

        agent.new_session()

        print("\n--- Session 2: Recall from Memory ---")
        agent.chat("What programming language should I use for a new project?")

        stats = agent.get_memory_stats()
        print(f"\n  Stats: {json.dumps(stats, indent=2)}")

    except ImportError:
        print("Skipping — openai package not installed.")


def main() -> None:
    """Run all memory system examples."""
    print("Agent with Memory Systems")
    print("=" * 60)

    demo_memory_layers()
    demo_session_persistence()

    print("\n" + "=" * 60)
    print("All memory examples completed.")


if __name__ == "__main__":
    main()
