from langchain_core.documents import Document
from src.utils.chunker import chunk_documents


def test_chunker_splits_text():
    docs = [Document(page_content="Word " * 2000, metadata={"source": "test.pdf", "page": 1})]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1


def test_chunker_preserves_metadata():
    docs = [Document(page_content="Short text.", metadata={"source": "test.pdf", "page": 1})]
    chunks = chunk_documents(docs)
    assert chunks[0].metadata["source"] == "test.pdf"
    assert chunks[0].metadata["page"] == 1


def test_chunker_empty_documents():
    assert chunk_documents([]) == []


def test_chunker_small_document():
    docs = [Document(page_content="Hello world.", metadata={"source": "a.pdf", "page": 1})]
    chunks = chunk_documents(docs)
    assert len(chunks) == 1
