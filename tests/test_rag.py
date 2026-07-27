"""Comprehensive tests for the RAG pipeline components.

Tests cover the pipeline orchestration, chunker, PDF processor,
vector store configuration, and the FastAPI application endpoints.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import io

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.utils.chunker import chunk_documents
from src.utils.pdf_processor import PDFProcessor
from src.core.config import Settings, settings
from src.models.schemas import (
    QueryRequest,
    QueryResponse,
    Source,
    HealthResponse,
    IngestResponse,
)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class TestChunker:
    """Tests for the recursive character text splitter."""

    def test_splits_long_text(self) -> None:
        docs = [Document(page_content="word " * 2000, metadata={"source": "a.pdf", "page": 1})]
        chunks = chunk_documents(docs)
        assert len(chunks) > 1

    def test_preserves_metadata(self) -> None:
        docs = [Document(page_content="Hello world.", metadata={"source": "b.pdf", "page": 3})]
        chunks = chunk_documents(docs)
        assert chunks[0].metadata["source"] == "b.pdf"
        assert chunks[0].metadata["page"] == 3

    def test_empty_input(self) -> None:
        assert chunk_documents([]) == []

    def test_small_document_single_chunk(self) -> None:
        docs = [Document(page_content="Short.", metadata={"source": "c.pdf", "page": 1})]
        chunks = chunk_documents(docs)
        assert len(chunks) == 1

    def test_chunk_content_not_empty(self) -> None:
        docs = [Document(page_content="A" * 2000, metadata={"source": "d.pdf", "page": 1})]
        chunks = chunk_documents(docs)
        for chunk in chunks:
            assert len(chunk.page_content) > 0

    def test_multiple_documents(self) -> None:
        docs = [
            Document(page_content="Doc one " * 500, metadata={"source": "a.pdf", "page": 1}),
            Document(page_content="Doc two " * 500, metadata={"source": "b.pdf", "page": 1}),
        ]
        chunks = chunk_documents(docs)
        assert len(chunks) > 1

    def test_chunk_size_respected(self) -> None:
        text = "word " * 1000
        docs = [Document(page_content=text, metadata={"source": "e.pdf", "page": 1})]
        chunks = chunk_documents(docs)
        for chunk in chunks:
            assert len(chunk.page_content) <= settings.chunk_size + 100


# ---------------------------------------------------------------------------
# PDF Processor
# ---------------------------------------------------------------------------

class TestPDFProcessor:
    """Tests for the PDF text extraction utility."""

    @pytest.fixture
    def fake_pdf(self, tmp_path: Path) -> Path:
        """Create a minimal PDF file for testing."""
        import fitz
        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Hello from test PDF")
        doc.save(str(pdf_path))
        doc.close()
        return pdf_path

    def test_extract_text(self, fake_pdf: Path) -> None:
        proc = PDFProcessor(fake_pdf)
        text = proc.extract_text()
        assert "Hello from test PDF" in text

    def test_extract_text_by_page(self, fake_pdf: Path) -> None:
        proc = PDFProcessor(fake_pdf)
        pages = proc.extract_text_by_page()
        assert len(pages) == 1
        assert pages[0]["page"] == 1
        assert "Hello from test PDF" in pages[0]["text"]

    def test_extract_metadata(self, fake_pdf: Path) -> None:
        proc = PDFProcessor(fake_pdf)
        meta = proc.extract_metadata()
        assert meta["pages"] == 1
        assert isinstance(meta["title"], str)
        assert isinstance(meta["author"], str)

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(Exception):
            PDFProcessor(tmp_path / "nonexistent.pdf").extract_text()


# ---------------------------------------------------------------------------
# Config / Settings
# ---------------------------------------------------------------------------

class TestSettings:
    """Verify configuration loads sensible defaults."""

    def test_chunk_size_positive(self) -> None:
        assert settings.chunk_size > 0

    def test_chunk_overlap_non_negative(self) -> None:
        assert settings.chunk_overlap >= 0

    def test_top_k_positive(self) -> None:
        assert settings.top_k > 0

    def test_collection_name(self) -> None:
        assert settings.collection_name == "med-psych"

    def test_default_provider(self) -> None:
        s = Settings()
        assert s.llm_provider in ("openai", "ollama", "gemini")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TestSchemas:
    """Validate Pydantic model construction and defaults."""

    def test_query_request_defaults(self) -> None:
        req = QueryRequest(question="What is MDD?")
        assert req.top_k is None

    def test_query_request_with_top_k(self) -> None:
        req = QueryRequest(question="Test", top_k=10)
        assert req.top_k == 10

    def test_source_optional_score(self) -> None:
        s = Source(file="doc.pdf", page=1)
        assert s.score is None

    def test_source_with_score(self) -> None:
        s = Source(file="doc.pdf", page=5, score=0.87)
        assert s.score == 0.87

    def test_query_response(self) -> None:
        src = Source(file="a.pdf", page=1)
        resp = QueryResponse(answer="Answer text", sources=[src])
        assert resp.answer == "Answer text"
        assert len(resp.sources) == 1
        assert resp.metrics is None

    def test_health_response(self) -> None:
        resp = HealthResponse(status="ok", llm="openai/gpt-4", docs_indexed=100)
        assert resp.docs_indexed == 100

    def test_ingest_response(self) -> None:
        resp = IngestResponse(chunks=42, document="paper.pdf")
        assert resp.chunks == 42


# ---------------------------------------------------------------------------
# Pipeline functions (mocked)
# ---------------------------------------------------------------------------

class TestPipelineIngest:
    """Tests for the ingest_pdf pipeline function (mocked)."""

    @patch("src.core.pipeline.get_vector_store")
    @patch("src.core.pipeline.PDFProcessor")
    def test_ingest_returns_chunk_count(self, MockProcessor, mock_store_fn) -> None:
        from src.core.pipeline import ingest_pdf

        mock_proc = MagicMock()
        mock_proc.extract_text_by_page.return_value = [
            {"page": 1, "text": "Page one"},
            {"page": 2, "text": "Page two"},
        ]
        MockProcessor.return_value = mock_proc

        mock_store = MagicMock()
        mock_store_fn.return_value = mock_store

        with patch("src.core.pipeline.chunk_documents", return_value=[1, 2, 3]):
            count = ingest_pdf("/fake/doc.pdf")
            assert count == 3
            mock_store.add_documents.assert_called_once()

    @patch("src.core.pipeline.get_vector_store")
    @patch("src.core.pipeline.PDFProcessor")
    def test_ingest_single_page(self, MockProcessor, mock_store_fn) -> None:
        from src.core.pipeline import ingest_pdf

        mock_proc = MagicMock()
        mock_proc.extract_text_by_page.return_value = [
            {"page": 1, "text": "Only page"},
        ]
        MockProcessor.return_value = mock_proc
        mock_store = MagicMock()
        mock_store_fn.return_value = mock_store

        with patch("src.core.pipeline.chunk_documents", return_value=[1]):
            count = ingest_pdf("/fake/one.pdf")
            assert count == 1


class TestPipelineRetrieve:
    """Tests for the retrieve pipeline function (mocked)."""

    @patch("src.core.pipeline.get_vector_store")
    def test_retrieve_uses_top_k(self, mock_store_fn) -> None:
        from src.core.pipeline import retrieve

        mock_store = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [Document(page_content="doc", metadata={})]
        mock_store.as_retriever.return_value = mock_retriever
        mock_store_fn.return_value = mock_store

        docs = retrieve("test query", k=3)
        mock_store.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 3},
        )
        assert len(docs) == 1

    @patch("src.core.pipeline.get_vector_store")
    def test_retrieve_default_top_k(self, mock_store_fn) -> None:
        from src.core.pipeline import retrieve

        mock_store = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_store.as_retriever.return_value = mock_retriever
        mock_store_fn.return_value = mock_store

        retrieve("q")
        mock_store.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": settings.top_k},
        )


# ---------------------------------------------------------------------------
# FastAPI endpoints (mocked)
# ---------------------------------------------------------------------------

from src.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """GET /health — mocked vector store."""

    @patch("src.main.get_vector_store")
    def test_health_ok(self, mock_store_fn) -> None:
        mock_store = MagicMock()
        mock_store._collection.count.return_value = 10
        mock_store_fn.return_value = mock_store
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["docs_indexed"] == 10


class TestQueryEndpoint:
    """POST /query — mocked retrieve and generate."""

    @patch("src.main.generate")
    @patch("src.main.retrieve")
    def test_query_empty_question(self, mock_retrieve, mock_generate) -> None:
        resp = client.post("/query", json={"question": "   "})
        assert resp.status_code == 400

    @patch("src.main.generate")
    @patch("src.main.retrieve")
    def test_query_no_docs(self, mock_retrieve, mock_generate) -> None:
        mock_retrieve.return_value = []
        resp = client.post("/query", json={"question": "What is X?"})
        assert resp.status_code == 200
        assert "No relevant documents" in resp.json()["answer"]

    @patch("src.main.generate")
    @patch("src.main.retrieve")
    def test_query_with_docs(self, mock_retrieve, mock_generate) -> None:
        mock_retrieve.return_value = [
            Document(page_content="Content", metadata={"source": "a.pdf", "page": 1})
        ]
        mock_generate.return_value = "Generated answer"
        resp = client.post("/query", json={"question": "What is X?"})
        assert resp.status_code == 200
        assert resp.json()["answer"] == "Generated answer"
        assert len(resp.json()["sources"]) == 1


class TestIngestEndpoint:
    """POST /ingest — mocked PDF ingestion."""

    @patch("src.main.ingest_pdf")
    def test_non_pdf_rejected(self, mock_ingest) -> None:
        resp = client.post(
            "/ingest",
            files={"file": ("data.csv", b"csv,content", "text/csv")},
        )
        assert resp.status_code == 400

    @patch("src.main.ingest_pdf")
    def test_pdf_accepted(self, mock_ingest) -> None:
        mock_ingest.return_value = 5
        resp = client.post(
            "/ingest",
            files={"file": ("report.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert resp.status_code == 200
        assert resp.json()["chunks"] == 5
        assert resp.json()["document"] == "report.pdf"
