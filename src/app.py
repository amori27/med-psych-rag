from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.retrieval.retriever import retrieve
from src.generation.generator import generate_answer
from src.config import Config

app = FastAPI(title="Med-Psych RAG System", version="1.0.0")


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    source_count: int


@app.get("/health")
def health():
    return {"status": "ok", "embedding_model": Config.embedding_model}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    k = request.top_k or Config.top_k
    docs = retrieve(request.question)

    if not docs:
        return QueryResponse(
            answer="No relevant documents found in the knowledge base.",
            sources=[],
            source_count=0,
        )

    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}, "
        f"Page {d.metadata.get('page', 'N/A')}]\n{d.page_content}"
        for d in docs
    )

    answer = generate_answer(request.question, context)

    sources = [
        {
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page", "N/A"),
        }
        for d in docs
    ]

    return QueryResponse(answer=answer, sources=sources, source_count=len(sources))
