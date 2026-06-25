"""FastAPI application entry point."""

from fastapi import FastAPI, UploadFile, HTTPException
from src.models.schemas import QueryRequest, QueryResponse, HealthResponse, IngestResponse, Source
from src.core.pipeline import retrieve, generate, ingest_pdf
from src.utils.vector_store import get_vector_store
from src.core.config import settings

app = FastAPI(title="Med-Psych RAG", version="2.0.0")


@app.get("/health", response_model=HealthResponse)
def health():
    store = get_vector_store()
    count = store._collection.count() if store._collection else 0
    model_name = "gemini-2.0-flash" if settings.llm_provider == "gemini" else settings.llm_model
    return HealthResponse(
        status="ok",
        llm=f"{settings.llm_provider}/{model_name}",
        docs_indexed=count,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    docs = retrieve(request.question, request.top_k)
    if not docs:
        return QueryResponse(answer="No relevant documents found.", sources=[], metrics=None)

    context = "\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}, p.{d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )
    answer = generate(request.question, context)

    sources = [
        Source(
            file=d.metadata.get("source", "unknown"),
            page=d.metadata.get("page", 0),
        )
        for d in docs
    ]
    return QueryResponse(answer=answer, sources=sources)


@app.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    path = f"data/{file.filename}"
    with open(path, "wb") as f:
        f.write(file.file.read())

    try:
        chunks = ingest_pdf(path)
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {e}")

    return IngestResponse(chunks=chunks, document=file.filename)
