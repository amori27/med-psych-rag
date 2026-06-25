from src.models.schemas import QueryRequest, QueryResponse, Source, HealthResponse


def test_query_request():
    req = QueryRequest(question="What is GAD?")
    assert req.question == "What is GAD?"
    assert req.top_k is None


def test_query_request_with_top_k():
    req = QueryRequest(question="Test", top_k=3)
    assert req.top_k == 3


def test_source_model():
    src = Source(file="doc.pdf", page=5, score=0.95)
    assert src.file == "doc.pdf"
    assert src.page == 5


def test_query_response():
    src = Source(file="doc.pdf", page=1)
    resp = QueryResponse(answer="Test answer", sources=[src])
    assert len(resp.sources) == 1
    assert resp.answer == "Test answer"


def test_health_response():
    resp = HealthResponse(status="ok", llm="ollama/llama3", docs_indexed=42)
    assert resp.status == "ok"
    assert resp.docs_indexed == 42
