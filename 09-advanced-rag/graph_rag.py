"""
Graph RAG: Knowledge Graph + Retrieval

Extracts entities and relationships from text, builds a knowledge graph,
and uses graph structure to enhance retrieval.
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    name: str
    entity_type: str
    properties: Dict = field(default_factory=dict)

    def __hash__(self):
        return hash(self.name.lower())

    def __eq__(self, other):
        return isinstance(other, Entity) and self.name.lower() == other.name.lower()


@dataclass
class Relationship:
    source: str
    target: str
    relation_type: str
    weight: float = 1.0
    properties: Dict = field(default_factory=dict)


@dataclass
class GraphChunk:
    text: str
    entities: List[str]
    source_doc: int


# ---------------------------------------------------------------------------
# 1. Entity and Relationship Extraction
# ---------------------------------------------------------------------------

class EntityExtractor:
    """Extract entities and relationships from text using pattern matching and LLM."""

    # Common entity types and patterns
    ENTITY_PATTERNS = {
        "TECHNOLOGY": r'\b(Python|Java|JavaScript|Rust|Go|Docker|Kubernetes|Redis|PostgreSQL|TensorFlow|PyTorch|React|Angular|Vue)\b',
        "CONCEPT": r'\b(machine learning|deep learning|neural network|natural language processing|computer vision|reinforcement learning|transformer|attention mechanism|gradient descent|backpropagation)\b',
        "ALGORITHM": r'\b(BERT|GPT|ResNet|YOLO|Adam|SGD|K-means|DBSCAN|LSTM|GRU|Transformer|BERT|GPT-2|GPT-3|GPT-4)\b',
        "PERSON": r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',
        "ORGANIZATION": r'\b(Google|Microsoft|OpenAI|Meta|DeepMind|Apple|Amazon|IBM|NVIDIA)\b',
    }

    RELATIONSHIP_PATTERNS = [
        (r'(\w+)\s+uses\s+(\w+)', "uses"),
        (r'(\w+)\s+implements\s+(\w+)', "implements"),
        (r'(\w+)\s+is\s+a\s+type\s+of\s+(\w+)', "is_type_of"),
        (r'(\w+)\s+improves\s+(\w+)', "improves"),
        (r'(\w+)\s+enables\s+(\w+)', "enables"),
        (r'(\w+)\s+consists\s+of\s+(\w+)', "consists_of"),
        (r'(\w+)\s+depends\s+on\s+(\w+)', "depends_on"),
        (r'(\w+)\s+trained\s+on\s+(\w+)', "trained_on"),
        (r'(\w+)\s+based\s+on\s+(\w+)', "based_on"),
    ]

    def extract(self, text: str) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from text."""
        entities = self._extract_entities(text)
        relationships = self._extract_relationships(text, entities)
        return entities, relationships

    def _extract_entities(self, text: str) -> List[Entity]:
        """Extract entities using pattern matching."""
        entities = []
        seen = set()

        for etype, pattern in self.ENTITY_PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group(1) if match.lastindex else match.group(0)
                if name.lower() not in seen:
                    seen.add(name.lower())
                    entities.append(Entity(name=name, entity_type=etype))

        return entities

    def _extract_relationships(
        self, text: str, entities: List[Entity]
    ) -> List[Relationship]:
        """Extract relationships between known entities."""
        relationships = []
        entity_names = {e.name.lower() for e in entities}

        for pattern, rel_type in self.RELATIONSHIP_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                source = match.group(1)
                target = match.group(2)
                if source.lower() in entity_names and target.lower() in entity_names:
                    relationships.append(Relationship(
                        source=source,
                        target=target,
                        relation_type=rel_type,
                    ))

        return relationships


# ---------------------------------------------------------------------------
# 2. Knowledge Graph (NetworkX)
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """Knowledge graph built with NetworkX."""

    def __init__(self):
        import networkx as nx
        self.graph = nx.DiGraph()
        self.entity_extractor = EntityExtractor()

    def add_document(self, text: str, doc_id: int):
        """Extract and add entities/relationships from a document."""
        entities, relationships = self.entity_extractor.extract(text)

        for entity in entities:
            self.graph.add_node(
                entity.name,
                entity_type=entity.entity_type,
                **entity.properties,
            )

        for rel in relationships:
            self.graph.add_edge(
                rel.source,
                rel.target,
                relation_type=rel.relation_type,
                weight=rel.weight,
                **rel.properties,
            )

    def get_neighbors(self, entity_name: str, depth: int = 1) -> Set[str]:
        """Get all neighbors within a given depth."""
        visited = set()
        frontier = {entity_name}

        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                if node in self.graph:
                    next_frontier.update(self.graph.successors(node))
                    next_frontier.update(self.graph.predecessors(node))
            frontier = next_frontier - visited

        return visited

    def get_subgraph(self, entities: List[str], depth: int = 2) -> "KnowledgeGraph":
        """Extract a subgraph around given entities."""
        subgraph = KnowledgeGraph()
        all_entities = set()

        for entity in entities:
            if entity in self.graph:
                neighbors = self.get_neighbors(entity, depth)
                all_entities.update(neighbors)

        for node in all_entities:
            if node in self.graph:
                subgraph.graph.add_node(node, **self.graph.nodes[node])

        for u, v, data in self.graph.edges(data=True):
            if u in all_entities and v in all_entities:
                subgraph.graph.add_edge(u, v, **data)

        return subgraph

    def describe(self) -> str:
        """Generate a text description of the graph."""
        lines = []
        lines.append(f"Nodes: {self.graph.number_of_nodes()}")
        lines.append(f"Edges: {self.graph.number_of_edges()}")

        if self.graph.number_of_nodes() > 0:
            lines.append("\nEntities by type:")
            types = {}
            for node, data in self.graph.nodes(data=True):
                etype = data.get("entity_type", "UNKNOWN")
                types.setdefault(etype, []).append(node)
            for etype, names in sorted(types.items()):
                lines.append(f"  {etype}: {', '.join(names[:10])}")

            lines.append("\nRelationships:")
            seen = set()
            for u, v, data in self.graph.edges(data=True):
                rel = data.get("relation_type", "related_to")
                edge_str = f"  {u} --[{rel}]--> {v}"
                if edge_str not in seen:
                    seen.add(edge_str)
                    lines.append(edge_str)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Graph-based Retrieval
# ---------------------------------------------------------------------------

class GraphRetriever:
    """Retrieve documents enhanced with graph context."""

    def __init__(self):
        self.documents: List[str] = []
        self.graph = KnowledgeGraph()

    def ingest(self, documents: List[str]):
        """Ingest documents and build knowledge graph."""
        self.documents = documents
        for i, doc in enumerate(documents):
            self.graph.add_document(doc, i)

    def retrieve(
        self, query: str, top_k: int = 5, graph_depth: int = 1
    ) -> List[Dict]:
        """
        Retrieve documents with graph-enhanced context.

        1. Find entities in query
        2. Get related entities from graph
        3. Score documents by entity overlap + text similarity
        """
        query_entities = self._extract_query_entities(query)
        graph_context = self._build_graph_context(query_entities, graph_depth)
        doc_scores = self._score_documents(query, query_entities, graph_context)

        doc_scores.sort(key=lambda x: x["score"], reverse=True)

        results = []
        for item in doc_scores[:top_k]:
            results.append({
                "text": self.documents[item["doc_idx"]],
                "score": item["score"],
                "entities_matched": item["entities"],
                "graph_context": graph_context,
                "doc_idx": item["doc_idx"],
            })

        return results

    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract entity names from query that exist in the graph."""
        words = set(re.findall(r'\b\w+\b', query.lower()))
        entities = []
        for node in self.graph.graph.nodes():
            node_words = set(node.lower().split())
            if node_words & words or node.lower() in query.lower():
                entities.append(node)
        return entities

    def _build_graph_context(self, entities: List[str], depth: int) -> str:
        """Build textual description of graph neighborhood."""
        if not entities:
            return ""

        subgraph = self.graph.get_subgraph(entities, depth)
        return subgraph.describe()

    def _score_documents(
        self, query: str, query_entities: List[str], graph_context: str
    ) -> List[Dict]:
        """Score documents based on entity overlap and text similarity."""
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))

        doc_scores = []
        for i, doc in enumerate(self.documents):
            doc_lower = doc.lower()

            # Text similarity (simple word overlap)
            doc_words = set(re.findall(r'\b\w+\b', doc_lower))
            word_overlap = len(query_words & doc_words) / max(len(query_words), 1)

            # Entity overlap
            entity_score = 0
            matched_entities = []
            doc_entities = set()
            for node in self.graph.graph.nodes():
                if node.lower() in doc_lower:
                    doc_entities.add(node)
                    if node in query_entities:
                        entity_score += 1
                        matched_entities.append(node)

            # Graph connectivity bonus
            graph_bonus = 0
            if graph_context:
                for entity in query_entities:
                    neighbors = self.graph.get_neighbors(entity, depth=1)
                    for neighbor in neighbors:
                        if neighbor.lower() in doc_lower:
                            graph_bonus += 0.3

            total_score = word_overlap * 0.4 + entity_score * 0.4 + graph_bonus * 0.2

            doc_scores.append({
                "doc_idx": i,
                "score": total_score,
                "entities": matched_entities,
            })

        return doc_scores


# ---------------------------------------------------------------------------
# 4. Full Pipeline
# ---------------------------------------------------------------------------

class GraphRAGPipeline:
    """Complete Graph RAG pipeline."""

    def __init__(self):
        self.retriever = GraphRetriever()

    def ingest(self, documents: List[str]):
        """Ingest documents and build knowledge graph."""
        self.retriever.ingest(documents)
        print(f"Ingested {len(documents)} documents")
        print(f"Knowledge graph: {self.retriever.graph.describe()}")

    def query(self, question: str, top_k: int = 3) -> Dict:
        """Query with graph-enhanced retrieval."""
        print(f"\n{'='*60}")
        print(f"Query: {question}")
        print(f"{'='*60}")

        results = self.retriever.retrieve(question, top_k=top_k)

        print(f"\n[Graph-enhanced retrieval] {len(results)} results:")
        for i, r in enumerate(results):
            print(f"\n  {i+1}. (score={r['score']:.3f})")
            print(f"     Entities: {r['entities_matched']}")
            print(f"     Text: {r['text'][:120]}...")

        # Generate answer with graph context
        answer = self._generate(question, results)
        print(f"\n[Answer] {answer}")

        return {
            "question": question,
            "results": results,
            "answer": answer,
            "graph_stats": {
                "nodes": self.retriever.graph.graph.number_of_nodes(),
                "edges": self.retriever.graph.graph.number_of_edges(),
            },
        }

    def _generate(self, question: str, results: List[Dict]) -> str:
        """Generate answer using graph context."""
        context_parts = []
        for r in results:
            context_parts.append(f"[Entity match: {', '.join(r['entities_matched'])}]\n{r['text']}")
        context = "\n\n".join(context_parts)

        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer based on the provided context. Use the entity relationships to provide comprehensive answers."},
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
    "Python is a programming language created by Guido van Rossum. Python is used for machine learning, web development, and data science. Python implements object-oriented programming concepts.",
    "Machine learning is a subset of artificial intelligence. Machine learning uses algorithms like neural networks and decision trees. Machine learning is enabled by deep learning techniques.",
    "Deep learning is part of machine learning that uses neural networks. Deep learning algorithms include CNNs, RNNs, and Transformers. Deep learning is used for computer vision and NLP.",
    "Natural language processing is a field of AI. NLP uses Transformer models like BERT and GPT. NLP enables machines to understand human language.",
    "Google developed TensorFlow, an open-source machine learning framework. TensorFlow is based on neural networks and is used for deep learning.",
    "OpenAI created GPT, a generative pre-trained transformer. GPT uses the Transformer architecture for natural language processing tasks.",
    "PyTorch is a machine learning framework developed by Meta. PyTorch is based on neural networks and is used for deep learning research.",
    "Docker is a containerization platform. Docker is used to deploy machine learning models. Docker depends on Linux kernel features.",
    "Kubernetes orchestrates Docker containers. Kubernetes is developed by Google. Kubernetes enables scaling of containerized applications.",
    "Transfer learning is a technique in deep learning. Transfer learning uses pre-trained models like BERT and GPT. Transfer learning enables training on small datasets.",
]


def demo():
    """Run Graph RAG demo."""
    print("=" * 70)
    print("GRAPH RAG DEMO")
    print("=" * 70)

    pipeline = GraphRAGPipeline()
    pipeline.ingest(SAMPLE_DOCUMENTS)

    queries = [
        "What is the relationship between deep learning and NLP?",
        "How does Python relate to machine learning frameworks?",
        "What models use the Transformer architecture?",
    ]

    for q in queries:
        pipeline.query(q)


if __name__ == "__main__":
    demo()
