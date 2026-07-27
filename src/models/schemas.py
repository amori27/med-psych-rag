"""Pydantic models for the Med-Psych RAG API."""

from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Incoming query payload."""

    question: str
    top_k: int | None = None


class Source(BaseModel):
    """Reference to a source document and page."""

    file: str
    page: int
    score: float | None = None


class QueryResponse(BaseModel):
    """Response payload for the /query endpoint."""

    answer: str
    sources: list[Source]
    metrics: dict | None = None


class HealthResponse(BaseModel):
    """Health-check response payload."""

    status: str
    llm: str
    docs_indexed: int


class IngestResponse(BaseModel):
    """Response payload after PDF ingestion."""

    chunks: int
    document: str
