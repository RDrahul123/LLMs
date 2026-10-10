import unittest

import pymupdf

from rag import chunk_document, extract_text, retrieve, unique_sources


class DocumentPipelineTests(unittest.TestCase):
    def test_plain_text_is_extracted_and_normalized(self):
        self.assertEqual(extract_text("notes.txt", b"  hello\n  world "), "hello world")

    def test_pdf_pages_are_extracted(self):
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 72), "A searchable PDF passage")
        content = document.tobytes()
        document.close()
        self.assertIn("searchable PDF passage", extract_text("sample.pdf", content))

    def test_unsupported_file_type_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_text("notes.docx", b"not a supported document")

    def test_chunking_preserves_order_and_source(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
        chunks = chunk_document(text, "notes.md", chunk_size=24, overlap=6)
        self.assertGreater(len(chunks), 1)
        self.assertEqual([chunk.chunk_index for chunk in chunks], list(range(len(chunks))))
        self.assertEqual(unique_sources(chunks), ["notes.md"])
        self.assertTrue(all(len(chunk.text) <= 24 for chunk in chunks))

    def test_bm25_ranks_relevant_source_first(self):
        chunks = [
            *chunk_document("The starter plan includes three projects and five team members.", "plans.md"),
            *chunk_document("Two factor authentication is available in personal security settings.", "security.md"),
            *chunk_document("Use the expense portal to request home office reimbursement.", "expenses.md"),
        ]
        results = retrieve("How many projects are included in the starter plan?", chunks, top_k=2)
        self.assertTrue(results)
        self.assertEqual(results[0][0].source, "plans.md")
        self.assertGreater(results[0][1], 0)

    def test_query_without_matching_terms_returns_no_chunks(self):
        chunks = chunk_document("The handbook describes account settings.", "handbook.md")
        self.assertEqual(retrieve("volcanic geology", chunks), [])

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_document("some text", "notes.txt", chunk_size=10, overlap=10)


if __name__ == "__main__":
    unittest.main()
