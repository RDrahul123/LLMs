"""
Reranking Pipeline for RAG

Retrieves initial candidates with ChromaDB, reranks with cross-encoders,
and generates answers. Includes Reciprocal Rank Fusion for multi-source reranking.
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Document:
    text: str
    metadata: Dict = field(default_factory=dict)
    score: float = 0.0


@dataclass
class RerankResult:
    documents: List[Document]
    method: str
    query: str


# ---------------------------------------------------------------------------
# 1. Initial retrieval with ChromaDB
# ---------------------------------------------------------------------------

class InitialRetriever:
    """Dense retrieval using ChromaDB."""

    def __init__(self, collection_name: str = "rag_docs"):
        import chromadb
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embed_fn = None

    def _get_embed_fn(self):
        if self._embed_fn is None:
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("all-MiniLM-L6-v2")
                self._embed_fn = lambda texts: model.encode(texts).tolist()
            except ImportError:
                self._embed_fn = None
        return self._embed_fn

    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """Add documents to the collection."""
        embed_fn = self._get_embed_fn()
        ids = [f"doc_{i}" for i in range(len(documents))]
        if metadatas is None:
            metadatas = [{} for _ in documents]

        if embed_fn:
            embeddings = embed_fn(documents)
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
        else:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )

    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """Retrieve top-k documents."""
        embed_fn = self._get_embed_fn()
        kwargs = {"n_results": top_k}

        if embed_fn:
            kwargs["query_embeddings"] = [embed_fn([query])[0]]
        else:
            kwargs["query_texts"] = [query]

        results = self.collection.query(**kwargs)

        docs = []
        for i in range(len(results["documents"][0])):
            docs.append(Document(
                text=results["documents"][0][i],
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                score=1 - results["distances"][0][i] if results.get("distances") else 0.0,
            ))
        return docs


# ---------------------------------------------------------------------------
# 2. Cross-encoder reranking
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """Rerank documents using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                print(f"[CrossEncoderReranker] sentence-transformers not installed")
                return None
        return self._model

    def rerank(self, query: str, documents: List[Document], top_n: int = 5) -> RerankResult:
        """Rerank documents and return top_n results."""
        model = self._get_model()
        if model is None:
            return RerankResult(documents=documents[:top_n], method="passthrough", query=query)

        pairs = [(query, doc.text) for doc in documents]
        scores = model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc.score = float(score)

        reranked = sorted(documents, key=lambda d: d.score, reverse=True)[:top_n]
        return RerankResult(documents=reranked, method="cross_encoder", query=query)


# ---------------------------------------------------------------------------
# 3. Cohere rerank API
# ---------------------------------------------------------------------------

class CohereReranker:
    """Rerank using Cohere's rerank API."""

    def __init__(self, model: str = "rerank-english-v3.0"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            api_key = os.environ.get("COHERE_API_KEY")
            if not api_key:
                print("[CohereReranker] COHERE_API_KEY not set")
                return None
            try:
                import cohere
                self._client = cohere.Client(api_key)
            except ImportError:
                print("[CohereReranker] cohere package not installed: pip install cohere")
                return None
        return self._client

    def rerank(self, query: str, documents: List[Document], top_n: int = 5) -> RerankResult:
        """Rerank documents using Cohere API."""
        client = self._get_client()
        if client is None:
            return RerankResult(documents=documents[:top_n], method="passthrough", query=query)

        try:
            texts = [doc.text for doc in documents]
            response = client.rerank(
                query=query,
                documents=texts,
                top_n=top_n,
                model=self.model,
            )

            reranked = []
            for result in response.results:
                doc = documents[result.index]
                doc.score = result.relevance_score
                reranked.append(doc)

            return RerankResult(documents=reranked, method="cohere", query=query)
        except Exception as e:
            print(f"[CohereReranker] Rerank failed: {e}")
            return RerankResult(documents=documents[:top_n], method="passthrough", query=query)


# ---------------------------------------------------------------------------
# 4. Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    k: int = 60,
    top_n: int = 5,
) -> List[Document]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank_i)) across all lists.
    """
    doc_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            key = doc.text[:100]
            score = 1.0 / (k + rank + 1)
            doc_scores[key] = doc_scores.get(key, 0) + score
            if key not in doc_map:
                doc_map[key] = doc

    sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
    result = []
    for key in sorted_keys[:top_n]:
        doc = doc_map[key]
        doc.score = doc_scores[key]
        result.append(doc)

    return result


# ---------------------------------------------------------------------------
# 5. Full pipeline: retrieve → rerank → generate
# ---------------------------------------------------------------------------

class RerankPipeline:
    """Complete RAG pipeline with reranking."""

    def __init__(
        self,
        initial_top_k: int = 10,
        final_top_n: int = 3,
        reranker: str = "cross_encoder",
    ):
        self.retriever = InitialRetriever()
        self.initial_top_k = initial_top_k
        self.final_top_n = final_top_n

        if reranker == "cross_encoder":
            self.reranker = CrossEncoderReranker()
        elif reranker == "cohere":
            self.reranker = CohereReranker()
        else:
            self.reranker = CrossEncoderReranker()

    def ingest(self, documents: List[str]):
        """Add documents to the retrieval index."""
        self.retriever.add_documents(documents)
        print(f"Ingested {len(documents)} documents")

    def query(self, question: str, generate: bool = False) -> Dict:
        """Run the full pipeline."""
        print(f"\nQuery: {question}")
        print("-" * 60)

        # Step 1: Initial retrieval
        initial_docs = self.retriever.retrieve(question, top_k=self.initial_top_k)
        print(f"\n[Initial retrieval] {len(initial_docs)} documents")
        for i, doc in enumerate(initial_docs[:3]):
            print(f"  {i+1}. (score={doc.score:.3f}) {doc.text[:80]}...")

        # Step 2: Rerank
        rerank_result = self.reranker.rerank(question, initial_docs, top_n=self.final_top_n)
        print(f"\n[Reranking] method={rerank_result.method}, top {self.final_top_n}:")
        for i, doc in enumerate(rerank_result.documents):
            print(f"  {i+1}. (score={doc.score:.3f}) {doc.text[:80]}...")

        # Step 3: Generate (if requested)
        answer = None
        if generate:
            context = "\n\n".join(d.text for d in rerank_result.documents)
            answer = self._generate(question, context)
            print(f"\n[Answer] {answer}")

        return {
            "question": question,
            "initial_docs": initial_docs,
            "reranked_docs": rerank_result.documents,
            "method": rerank_result.method,
            "answer": answer,
        }

    def _generate(self, question: str, context: str) -> str:
        """Generate answer using OpenAI API."""
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer based on the provided context. If the context doesn't contain enough information, say so."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Generation unavailable: {e}]"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS = [
    "Python is a high-level programming language known for its simplicity and readability. It supports multiple paradigms including object-oriented, functional, and procedural programming.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data. Common algorithms include linear regression, decision trees, and neural networks.",
    "Deep learning uses neural networks with many layers. CNNs process images, RNNs handle sequences, and Transformers power modern NLP models like GPT and BERT.",
    "Natural language processing (NLP) deals with interaction between computers and human language. Applications include translation, sentiment analysis, and chatbots.",
    "Computer vision enables machines to interpret visual information from the world. Tasks include image classification, object detection, and semantic segmentation.",
    "Reinforcement learning trains agents through reward signals. The agent explores an environment and learns optimal policies through trial and error.",
    "Data preprocessing is crucial for machine learning. Steps include handling missing values, normalization, feature encoding, and train-test splitting.",
    "Transfer learning reuses pre-trained models for new tasks. It reduces training time and data requirements, especially for domain-specific applications.",
    "MLOps combines machine learning with DevOps practices. It covers model deployment, monitoring, versioning, and continuous integration of ML systems.",
    "Ethical AI addresses bias, fairness, transparency, and accountability in AI systems. It ensures AI benefits society without unintended harm.",
]


def demo_reranking():
    """Demonstrate reranking pipeline."""
    print("=" * 70)
    print("RERANKING PIPELINE DEMO")
    print("=" * 70)

    pipeline = RerankPipeline(
        initial_top_k=5,
        final_top_n=3,
        reranker="cross_encoder",
    )
    pipeline.ingest(SAMPLE_DOCUMENTS)

    # Show before/after
    pipeline.query("What is deep learning and how does it relate to NLP?")

    print("\n" + "=" * 70)
    print("COMPARISON: With vs Without Reranking")
    print("=" * 70)

    question = "How do Transformers work in natural language processing?"

    # Without reranking: just use initial retrieval
    initial = pipeline.retriever.retrieve(question, top_k=3)
    print("\n[Without Reranking] Top 3 from initial retrieval:")
    for i, doc in enumerate(initial):
        print(f"  {i+1}. {doc.text[:80]}...")

    # With reranking
    result = pipeline.query(question)
    print("\n[With Reranking] Top 3 after cross-encoder:")
    for i, doc in enumerate(result["reranked_docs"]):
        print(f"  {i+1}. (score={doc.score:.3f}) {doc.text[:80]}...")


if __name__ == "__main__":
    demo_reranking()
