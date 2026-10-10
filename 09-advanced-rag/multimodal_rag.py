"""
Multimodal RAG: Images, Tables, and Charts

Extends RAG to handle non-text content in documents.
Extracts images/tables from PDFs, generates descriptions,
and retrieves both text and visual content.
"""

import os
import re
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TextChunk:
    text: str
    chunk_type: str = "text"
    metadata: Dict = field(default_factory=dict)


@dataclass
class ImageChunk:
    description: str
    image_path: Optional[str] = None
    page_number: int = 0
    chunk_type: str = "image"
    metadata: Dict = field(default_factory=dict)


@dataclass
class TableChunk:
    content: str  # Markdown table
    page_number: int = 0
    chunk_type: str = "table"
    metadata: Dict = field(default_factory=dict)


@dataclass
class MultimodalDocument:
    text_chunks: List[TextChunk] = field(default_factory=list)
    image_chunks: List[ImageChunk] = field(default_factory=list)
    table_chunks: List[TableChunk] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. PDF Content Extraction
# ---------------------------------------------------------------------------

class PDFExtractor:
    """Extract text, images, and tables from PDFs."""

    def __init__(self):
        self._pdf_reader = None

    def _get_reader(self):
        if self._pdf_reader is None:
            try:
                import fitz  # PyMuPDF
                self._pdf_reader = fitz
            except ImportError:
                print("[PDFExtractor] PyMuPDF not installed: pip install pymupdf")
                return None
        return self._pdf_reader

    def extract_text(self, pdf_path: str) -> List[TextChunk]:
        """Extract text chunks from PDF."""
        fitz = self._get_reader()
        if fitz is None:
            return []

        doc = fitz.open(pdf_path)
        chunks = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            paragraphs = re.split(r'\n\s*\n', text)

            for para in paragraphs:
                para = para.strip()
                if len(para) > 50:
                    chunks.append(TextChunk(
                        text=para,
                        page_number=page_num + 1,
                        metadata={"source": pdf_path, "page": page_num + 1},
                    ))

        doc.close()
        return chunks

    def extract_images(self, pdf_path: str, output_dir: str = "extracted_images") -> List[ImageChunk]:
        """Extract images from PDF and save them."""
        fitz = self._get_reader()
        if fitz is None:
            return []

        os.makedirs(output_dir, exist_ok=True)
        doc = fitz.open(pdf_path)
        image_chunks = []
        img_idx = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()

            for img_index in range(len(image_list)):
                xref = image_list[img_index][0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_path = os.path.join(output_dir, f"img_{page_num}_{img_idx}.{image_ext}")

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                image_chunks.append(ImageChunk(
                    image_path=image_path,
                    page_number=page_num + 1,
                    metadata={"source": pdf_path, "page": page_num + 1},
                ))
                img_idx += 1

        doc.close()
        return image_chunks

    def extract_tables(self, pdf_path: str) -> List[TableChunk]:
        """
        Extract tables from PDF using basic heuristics.
        For production, use libraries like camelot or tabula.
        """
        fitz = self._get_reader()
        if fitz is None:
            return []

        doc = fitz.open(pdf_path)
        table_chunks = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            # Heuristic: detect table-like structures (lines with | or consistent spacing)
            lines = text.split("\n")
            table_lines = []
            in_table = False

            for line in lines:
                if "|" in line or (line.count("\t") >= 2):
                    in_table = True
                    table_lines.append(line)
                elif in_table:
                    if len(table_lines) >= 2:
                        table_text = "\n".join(table_lines)
                        table_chunks.append(TableChunk(
                            content=table_text,
                            page_number=page_num + 1,
                            metadata={"source": pdf_path, "page": page_num + 1},
                        ))
                    table_lines = []
                    in_table = False

            if in_table and len(table_lines) >= 2:
                table_text = "\n".join(table_lines)
                table_chunks.append(TableChunk(
                    content=table_text,
                    page_number=page_num + 1,
                    metadata={"source": pdf_path, "page": page_num + 1},
                ))

        doc.close()
        return table_chunks

    def extract_all(self, pdf_path: str, output_dir: str = "extracted_images") -> MultimodalDocument:
        """Extract all content from PDF."""
        return MultimodalDocument(
            text_chunks=self.extract_text(pdf_path),
            image_chunks=self.extract_images(pdf_path, output_dir),
            table_chunks=self.extract_tables(pdf_path),
        )


# ---------------------------------------------------------------------------
# 2. Image Description Generator
# ---------------------------------------------------------------------------

class ImageDescriber:
    """Generate descriptions for images using vision models."""

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

    def describe_image(self, image_path: str) -> str:
        """Generate a description for an image using GPT-4V."""
        import base64

        client = self._get_client()
        if client is None:
            return f"[Image at {image_path} - description unavailable without API]"

        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            ext = os.path.splitext(image_path)[1].lower()
            mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}
            mime_type = mime_map.get(ext, "image/png")

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail for use in a RAG system. Focus on the key information, data, or concepts shown."},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
                        ],
                    }
                ],
                max_tokens=300,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Image description failed: {e}]"

    def describe_table(self, table_text: str) -> str:
        """Generate a description for a table."""
        client = self._get_client()
        if client is None:
            return f"[Table: {table_text[:200]}]"

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Describe this table concisely for a RAG system. Highlight key data points and patterns."},
                    {"role": "user", "content": f"Table content:\n{table_text[:2000]}"},
                ],
                max_tokens=200,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Table description failed: {e}]"


# ---------------------------------------------------------------------------
# 3. Multimodal Embedding & Retrieval
# ---------------------------------------------------------------------------

class MultimodalRetriever:
    """Embed and retrieve both text and image content."""

    def __init__(self):
        import chromadb
        self.client = chromadb.Client()
        self.text_collection = self.client.create_collection(
            name="multimodal_text",
            metadata={"hnsw:space": "cosine"},
        )
        self.image_collection = self.client.create_collection(
            name="multimodal_images",
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

    def add_text_chunks(self, chunks: List[TextChunk]):
        """Add text chunks to the index."""
        model = self._get_model()
        texts = [c.text for c in chunks]
        ids = [f"text_{i}" for i in range(len(texts))]
        metadatas = [{"type": "text", "page": c.metadata.get("page", 0)} for c in chunks]

        if model:
            embeddings = model.encode(texts).tolist()
            self.text_collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
        else:
            self.text_collection.add(documents=texts, ids=ids, metadatas=metadatas)

    def add_image_descriptions(self, chunks: List[ImageChunk]):
        """Add image descriptions to the index."""
        model = self._get_model()
        descriptions = [c.description for c in chunks]
        ids = [f"img_{i}" for i in range(len(descriptions))]
        metadatas = [{"type": "image", "page": c.page_number, "path": c.image_path or ""} for c in chunks]

        if model:
            embeddings = model.encode(descriptions).tolist()
            self.image_collection.add(documents=descriptions, embeddings=embeddings, ids=ids, metadatas=metadatas)
        else:
            self.image_collection.add(documents=descriptions, ids=ids, metadatas=metadatas)

    def add_table_chunks(self, chunks: List[TableChunk]):
        """Add table content to the text index."""
        model = self._get_model()
        texts = [c.content for c in chunks]
        ids = [f"table_{i}" for i in range(len(texts))]
        metadatas = [{"type": "table", "page": c.page_number} for c in chunks]

        if model:
            embeddings = model.encode(texts).tolist()
            self.text_collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
        else:
            self.text_collection.add(documents=texts, ids=ids, metadatas=metadatas)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve from both text and image collections."""
        model = self._get_model()
        all_results = []

        # Search text collection
        text_kwargs = {"n_results": top_k}
        if model:
            text_kwargs["query_embeddings"] = [model.encode([query])[0].tolist()]
        else:
            text_kwargs["query_texts"] = [query]

        try:
            text_results = self.text_collection.query(**text_kwargs)
            for i in range(len(text_results["documents"][0])):
                all_results.append({
                    "text": text_results["documents"][0][i],
                    "type": text_results["metadatas"][0][i].get("type", "text"),
                    "score": 1 - text_results["distances"][0][i] if text_results.get("distances") else 0,
                    "metadata": text_results["metadatas"][0][i],
                })
        except Exception:
            pass

        # Search image collection
        img_kwargs = {"n_results": top_k}
        if model:
            img_kwargs["query_embeddings"] = [model.encode([query])[0].tolist()]
        else:
            img_kwargs["query_texts"] = [query]

        try:
            img_results = self.image_collection.query(**img_kwargs)
            for i in range(len(img_results["documents"][0])):
                all_results.append({
                    "text": img_results["documents"][0][i],
                    "type": "image",
                    "score": 1 - img_results["distances"][0][i] if img_results.get("distances") else 0,
                    "metadata": img_results["metadatas"][0][i],
                })
        except Exception:
            pass

        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]


# ---------------------------------------------------------------------------
# 4. Full Multimodal RAG Pipeline
# ---------------------------------------------------------------------------

class MultimodalRAGPipeline:
    """Complete multimodal RAG pipeline."""

    def __init__(self):
        self.extractor = PDFExtractor()
        self.describer = ImageDescriber()
        self.retriever = MultimodalRetriever()

    def ingest_pdf(self, pdf_path: str):
        """Extract and index all content from a PDF."""
        print(f"Processing: {pdf_path}")

        doc = self.extractor.extract_all(pdf_path)
        print(f"  Text chunks: {len(doc.text_chunks)}")
        print(f"  Images: {len(doc.image_chunks)}")
        print(f"  Tables: {len(doc.table_chunks)}")

        # Add text chunks
        if doc.text_chunks:
            self.retriever.add_text_chunks(doc.text_chunks)
            print(f"  Indexed {len(doc.text_chunks)} text chunks")

        # Describe and add images
        for i, img in enumerate(doc.image_chunks):
            print(f"  Describing image {i+1}/{len(doc.image_chunks)}...")
            img.description = self.describer.describe_image(img.image_path)
        if doc.image_chunks:
            self.retriever.add_image_descriptions(doc.image_chunks)
            print(f"  Indexed {len(doc.image_chunks)} image descriptions")

        # Add tables
        for table in doc.table_chunks:
            table_content = f"Table: {table.content}"
            self.retriever.add_table_chunks([table])
        if doc.table_chunks:
            print(f"  Indexed {len(doc.table_chunks)} tables")

    def ingest_text_documents(self, documents: List[str]):
        """Ingest plain text documents."""
        chunks = [TextChunk(text=doc) for doc in documents]
        self.retriever.add_text_chunks(chunks)
        print(f"Ingested {len(documents)} text documents")

    def query(self, question: str, top_k: int = 5) -> Dict:
        """Query the multimodal RAG system."""
        print(f"\n{'='*60}")
        print(f"Query: {question}")
        print(f"{'='*60}")

        results = self.retriever.retrieve(question, top_k=top_k)

        print(f"\nRetrieved {len(results)} results:")
        for i, r in enumerate(results):
            type_label = r["type"].upper()
            print(f"  {i+1}. [{type_label}] (score={r['score']:.3f}) {r['text'][:100]}...")

        answer = self._generate(question, results)
        print(f"\n[Answer] {answer}")

        return {"question": question, "results": results, "answer": answer}

    def _generate(self, question: str, results: List[Dict]) -> str:
        """Generate answer incorporating multimodal context."""
        context_parts = []
        for r in results:
            prefix = f"[{r['type'].upper()}]" if r["type"] != "text" else ""
            context_parts.append(f"{prefix} {r['text']}")
        context = "\n\n".join(context_parts)

        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer based on the provided multimodal context. Reference images and tables when relevant."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Generation unavailable: {e}]"


# ---------------------------------------------------------------------------
# Demo (text-only mode, no PDF needed)
# ---------------------------------------------------------------------------

def demo():
    """Run multimodal RAG demo with synthetic content."""
    print("=" * 70)
    print("MULTIMODAL RAG DEMO")
    print("=" * 70)

    pipeline = MultimodalRAGPipeline()

    # Simulate multimodal content
    text_chunks = [
        "Machine learning models require large datasets for training. Data quality directly impacts model performance.",
        "Neural networks consist of layers of interconnected nodes. Deep learning uses networks with many layers.",
        "Transfer learning allows reusing pre-trained models for new tasks. This reduces training time and data requirements.",
        "The Transformer architecture uses self-attention mechanisms. Transformers process sequences in parallel, unlike RNNs.",
        "Convolutional neural networks are effective for image processing. CNNs use filters to detect features in images.",
    ]

    image_chunks = [
        ImageChunk(
            description="A diagram showing the Transformer architecture with encoder and decoder blocks. The self-attention mechanism is highlighted, showing how each token attends to all other tokens in the sequence.",
            page_number=1,
        ),
        ImageChunk(
            description="A bar chart comparing model performance on ImageNet. ResNet-50 achieves 76.1% accuracy, EfficientNet-B7 achieves 84.3%, and Vision Transformer (ViT) achieves 88.5%.",
            page_number=3,
        ),
        ImageChunk(
            description="A confusion matrix for a classification model showing high accuracy for classes A and B, but lower performance for class C due to class imbalance.",
            page_number=5,
        ),
    ]

    table_chunks = [
        TableChunk(
            content="Model | Parameters | Accuracy | Speed\nResNet-50 | 25M | 76.1% | Fast\nEfficientNet-B7 | 66M | 84.3% | Medium\nViT-Large | 307M | 88.5% | Slow",
            page_number=2,
        ),
    ]

    # Add content to retriever
    pipeline.retriever.add_text_chunks([TextChunk(text=t) for t in text_chunks])
    pipeline.retriever.add_image_descriptions(image_chunks)
    pipeline.retriever.add_table_chunks(table_chunks)

    # Query
    pipeline.query("How does the Transformer architecture compare to other models?")
    pipeline.query("What are the trade-offs between model size and accuracy?")


if __name__ == "__main__":
    demo()
