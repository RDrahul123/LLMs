"""
Agentic RAG: Self-Reflecting, Iterative Retrieval

A RAG system that evaluates its own retrieval quality and iterates
until it can answer with confidence or exhausts attempts.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ReflectionAction(Enum):
    ANSWER = "answer"
    REWRITE = "rewrite"
    RETRIEVE_AGAIN = "retrieve_again"
    INSUFFICIENT = "insufficient"


@dataclass
class RetrievalAttempt:
    query: str
    documents: List[str]
    iteration: int
    confidence: float = 0.0
    reflection: str = ""


@dataclass
class AgenticResult:
    answer: str
    confidence: float
    attempts: List[RetrievalAttempt]
    final_query: str
    succeeded: bool


# ---------------------------------------------------------------------------
# 1. Basic RAG retriever
# ---------------------------------------------------------------------------

class BasicRetriever:
    """Simple vector retrieval with ChromaDB."""

    def __init__(self):
        import chromadb
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name="agentic_rag",
            metadata={"hnsw:space": "cosine"},
        )
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                return None
        return self._model

    def add_documents(self, documents: List[str]):
        """Index documents."""
        model = self._get_model()
        ids = [f"doc_{i}" for i in range(len(documents))]
        if model:
            embeddings = model.encode(documents).tolist()
            self.collection.add(documents=documents, embeddings=embeddings, ids=ids)
        else:
            self.collection.add(documents=documents, ids=ids)

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve top-k documents."""
        model = self._get_model()
        kwargs = {"n_results": top_k}
        if model:
            kwargs["query_embeddings"] = [model.encode([query])[0].tolist()]
        else:
            kwargs["query_texts"] = [query]

        results = self.collection.query(**kwargs)
        return results["documents"][0]


# ---------------------------------------------------------------------------
# 2. Self-Reflection Module
# ---------------------------------------------------------------------------

class SelfReflector:
    """Evaluates retrieval quality and decides next action."""

    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except Exception:
                return None
        return self._client

    def evaluate(
        self, query: str, documents: List[str], iteration: int, max_iterations: int = 3
    ) -> tuple[ReflectionAction, float, str]:
        """
        Evaluate if retrieved context is sufficient to answer.

        Returns: (action, confidence, reflection_text)
        """
        # Heuristic fallback if no LLM available
        client = self._get_client()
        if client is None:
            return self._heuristic_evaluate(query, documents, iteration, max_iterations)

        context = "\n\n".join(f"[Doc {i+1}] {doc}" for i, doc in enumerate(documents))

        eval_prompt = f"""You are evaluating retrieval quality for a RAG system.

Query: {query}

Retrieved Context:
{context}

Iteration: {iteration}/{max_iterations}

Evaluate whether the retrieved context is sufficient to answer the query.

Respond with EXACTLY this JSON format:
{{
    "confidence": <0.0 to 1.0>,
    "action": "<answer|rewrite|retrieve_again|insufficient>",
    "reasoning": "<brief explanation>",
    "suggested_query": "<if action is rewrite, provide a better query; otherwise empty string>"
}}

Confidence thresholds:
- 0.8+: context clearly answers the query → action: "answer"
- 0.5-0.8: partial context, might benefit from rewrite → action: "rewrite"
- 0.3-0.5: poor context, try different retrieval → action: "retrieve_again"
- Below 0.3: insufficient information → action: "insufficient"

If iteration == max_iterations, force action to "answer" or "insufficient"."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0,
            )
            import json
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(content)

            action_str = result.get("action", "answer")
            action = ReflectionAction(action_str) if action_str in [a.value for a in ReflectionAction] else ReflectionAction.ANSWER
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "")
            suggested_query = result.get("suggested_query", "")

            # Override on last iteration
            if iteration >= max_iterations and action == ReflectionAction.RETRIEVE_AGAIN:
                action = ReflectionAction.ANSWER if confidence > 0.3 else ReflectionAction.INSUFFICIENT

            return action, confidence, f"{reasoning}\nSuggested query: {suggested_query}" if suggested_query else reasoning

        except Exception as e:
            return self._heuristic_evaluate(query, documents, iteration, max_iterations)

    def _heuristic_evaluate(
        self, query: str, documents: List[str], iteration: int, max_iterations: int
    ) -> tuple[ReflectionAction, float, str]:
        """Fallback heuristic evaluation without LLM."""
        query_words = set(query.lower().split())
        doc_text = " ".join(documents).lower()
        overlap = sum(1 for w in query_words if w in doc_text)
        confidence = min(overlap / max(len(query_words), 1), 1.0)

        if confidence >= 0.7:
            action = ReflectionAction.ANSWER
        elif iteration >= max_iterations:
            action = ReflectionAction.ANSWER if confidence > 0.3 else ReflectionAction.INSUFFICIENT
        elif confidence < 0.3:
            action = ReflectionAction.REWRITE
        else:
            action = ReflectionAction.RETRIEVE_AGAIN

        return action, confidence, f"Heuristic: {overlap}/{len(query_words)} query words found"


# ---------------------------------------------------------------------------
# 3. Query Rewriter
# ---------------------------------------------------------------------------

class QueryRewriter:
    """Rewrite queries to improve retrieval quality."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except Exception:
                return None
        return self._client

    def rewrite(self, original_query: str, context: str, reflection: str) -> str:
        """Rewrite the query based on reflection feedback."""
        client = self._get_client()
        if client is None:
            return self._simple_rewrite(original_query)

        prompt = f"""Rewrite this search query to retrieve better information.

Original query: {original_query}

Current context (not sufficient):
{context[:500]}

Reflection: {reflection}

Generate a rewritten query that would retrieve more relevant information.
Return ONLY the rewritten query, nothing else."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            rewritten = response.choices[0].message.content.strip()
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            return rewritten
        except Exception:
            return self._simple_rewrite(original_query)

    def _simple_rewrite(self, query: str) -> str:
        """Simple keyword-based rewrite."""
        words = query.split()
        if len(words) <= 3:
            return query
        return " ".join(words[:len(words)//2])


# ---------------------------------------------------------------------------
# 4. Source Verifier
# ---------------------------------------------------------------------------

class SourceVerifier:
    """Verify and score source documents for reliability."""

    def verify(self, documents: List[str], query: str) -> List[Dict]:
        """Score each document for relevance and reliability."""
        verified = []
        query_words = set(query.lower().split())

        for i, doc in enumerate(documents):
            doc_words = set(doc.lower().split())
            relevance = len(query_words & doc_words) / max(len(query_words), 1)
            length_score = min(len(doc) / 500, 1.0)

            verified.append({
                "index": i,
                "text": doc,
                "relevance_score": round(relevance, 3),
                "length_score": round(length_score, 3),
                "combined_score": round(relevance * 0.7 + length_score * 0.3, 3),
            })

        verified.sort(key=lambda x: x["combined_score"], reverse=True)
        return verified


# ---------------------------------------------------------------------------
# 5. Full Agentic RAG Pipeline
# ---------------------------------------------------------------------------

class AgenticRAGPipeline:
    """Self-reflecting RAG with iterative retrieval."""

    def __init__(self, max_iterations: int = 3, min_confidence: float = 0.6):
        self.retriever = BasicRetriever()
        self.reflector = SelfReflector(min_confidence)
        self.rewriter = QueryRewriter()
        self.verifier = SourceVerifier()
        self.max_iterations = max_iterations

    def ingest(self, documents: List[str]):
        """Add documents to the index."""
        self.retriever.add_documents(documents)
        print(f"Ingested {len(documents)} documents")

    def query(self, question: str) -> AgenticResult:
        """Run the agentic RAG pipeline with reflection loop."""
        print(f"\n{'='*60}")
        print(f"Agentic RAG Query: {question}")
        print(f"{'='*60}")

        attempts: List[RetrievalAttempt] = []
        current_query = question

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n--- Iteration {iteration}/{self.max_iterations} ---")
            print(f"Query: {current_query}")

            # Retrieve
            docs = self.retriever.retrieve(current_query, top_k=5)
            print(f"Retrieved {len(docs)} documents")

            # Reflect
            action, confidence, reflection = self.reflector.evaluate(
                current_query, docs, iteration, self.max_iterations
            )
            print(f"Confidence: {confidence:.2f} | Action: {action.value}")
            print(f"Reflection: {reflection[:200]}")

            attempt = RetrievalAttempt(
                query=current_query,
                documents=docs,
                iteration=iteration,
                confidence=confidence,
                reflection=reflection,
            )
            attempts.append(attempt)

            if action == ReflectionAction.ANSWER:
                print("\n✓ Sufficient context found, generating answer")
                answer = self._generate_answer(question, docs, attempts)
                return AgenticResult(
                    answer=answer,
                    confidence=confidence,
                    attempts=attempts,
                    final_query=current_query,
                    succeeded=True,
                )

            elif action == ReflectionAction.REWRITE:
                print("→ Rewriting query...")
                current_query = self.rewriter.rewrite(current_query, " ".join(docs), reflection)
                print(f"New query: {current_query}")

            elif action == ReflectionAction.RETRIEVE_AGAIN:
                print("→ Retrying retrieval with same query...")

            elif action == ReflectionAction.INSUFFICIENT:
                print("\n✗ Insufficient information after all attempts")
                answer = self._generate_answer(question, docs, attempts, partial=True)
                return AgenticResult(
                    answer=answer,
                    confidence=confidence,
                    attempts=attempts,
                    final_query=current_query,
                    succeeded=False,
                )

        # Exhausted all iterations
        last_docs = attempts[-1].documents if attempts else []
        answer = self._generate_answer(question, last_docs, attempts, partial=True)
        return AgenticResult(
            answer=answer,
            confidence=attempts[-1].confidence if attempts else 0,
            attempts=attempts,
            final_query=current_query,
            succeeded=False,
        )

    def _generate_answer(
        self, question: str, documents: List[str],
        attempts: List[RetrievalAttempt], partial: bool = False
    ) -> str:
        """Generate final answer from retrieved documents."""
        context = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(documents))
        prefix = "Based on the available information (partial results):" if partial else "Based on the retrieved context:"

        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"{prefix}\nIf information is insufficient, acknowledge limitations."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback: return concatenated context
            return f"[Generation unavailable. Retrieved context: {context[:500]}]"

    def _print_summary(self, result: AgenticResult):
        """Print execution summary."""
        print(f"\n{'='*60}")
        print("EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Succeeded: {result.succeeded}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Iterations: {len(result.attempts)}")
        print(f"Final query: {result.final_query}")
        for i, attempt in enumerate(result.attempts):
            print(f"\n  Attempt {i+1}:")
            print(f"    Query: {attempt.query[:80]}")
            print(f"    Docs: {len(attempt.documents)}")
            print(f"    Confidence: {attempt.confidence:.2f}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS = [
    "Transformer models use self-attention mechanisms to process sequences in parallel.",
    "BERT is a bidirectional transformer pre-trained on masked language modeling.",
    "GPT models are autoregressive transformers trained on next token prediction.",
    "Fine-tuning adapts pre-trained models to specific downstream tasks.",
    "Transfer learning reuses knowledge from source tasks for target tasks.",
    "Attention is all you need introduced the Transformer architecture in 2017.",
    "PyTorch and TensorFlow are the main frameworks for deep learning research.",
    "Docker containers package applications with their dependencies.",
    "Kubernetes orchestrates container deployment and scaling.",
    "GraphQL provides flexible querying for APIs.",
    "REST APIs use standard HTTP methods for communication.",
    "PostgreSQL is an advanced open-source relational database.",
    "Redis provides in-memory caching and data structures.",
    "Machine learning requires large datasets for training.",
    "Data augmentation increases training data diversity.",
    "Batch normalization stabilizes neural network training.",
    "Dropout prevents overfitting in neural networks.",
    "Learning rate scheduling improves training convergence.",
]


def demo():
    """Run the agentic RAG demo."""
    print("=" * 70)
    print("AGENTIC RAG DEMO: Self-Reflecting Retrieval")
    print("=" * 70)

    pipeline = AgenticRAGPipeline(max_iterations=3, min_confidence=0.6)
    pipeline.ingest(SAMPLE_DOCUMENTS)

    queries = [
        "How do Transformers handle long-range dependencies in sequences?",
        "What is the difference between BERT and GPT architectures?",
        "How does containerization help with machine learning deployment?",
    ]

    for q in queries:
        result = pipeline.query(q)
        pipeline._print_summary(result)
        print(f"\n[Answer] {result.answer[:300]}...")
        print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()
