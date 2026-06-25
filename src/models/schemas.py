from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class Source(BaseModel):
    file: str
    page: int
    score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    metrics: dict | None = None


class HealthResponse(BaseModel):
    status: str
    llm: str
    docs_indexed: int


class IngestResponse(BaseModel):
    chunks: int
    document: str
