from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.config import Config
from src.generation.generator import generate_answer

app = FastAPI(title="Med-Psych RAG System", version="1.0.0")

_retriever = None


def get_retriever():
    global _retriever
    if Config.demo_mode or Config.llm_provider == "demo":
        from src.retrieval.local_retriever import LocalRetriever
        if _retriever is None:
            _retriever = LocalRetriever()
            _retriever.load()
        return _retriever, "local"
    else:
        from src.retrieval.retriever import retrieve as _retrieve
        return _retrieve, "remote"


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    source_count: int


@app.get("/health")
def health():
    return {
        "status": "ok",
        "embedding_model": Config.embedding_model,
        "llm_provider": Config.llm_provider,
        "demo_mode": Config.demo_mode,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if not Config.demo_mode and Config.llm_provider == "openai" and not Config.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY not configured. Set DEMO=true for local testing.",
        )

    retriever, retriever_type = get_retriever()

    try:
        if retriever_type == "local":
            docs = retriever.retrieve(request.question, request.top_k)
        else:
            docs = retriever(request.question)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Retrieval failed: {e}. Ensure documents have been ingested.",
        )

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

    return QueryResponse(
        answer=answer, sources=sources, source_count=len(sources)
    )
