"""Orchestrates the full RAG pipeline: ingest, retrieve, generate."""

from pathlib import Path
from langchain_core.documents import Document
from src.core.config import settings
from src.utils.pdf_processor import PDFProcessor
from src.utils.chunker import chunk_documents
from src.utils.vector_store import get_vector_store


def ingest_pdf(file_path: str | Path) -> int:
    path = Path(file_path)
    processor = PDFProcessor(path)
    pages = processor.extract_text_by_page()

    docs = [
        Document(
            page_content=p["text"],
            metadata={"source": path.name, "page": p["page"]},
        )
        for p in pages
    ]
    chunks = chunk_documents(docs)
    store = get_vector_store()
    store.add_documents(chunks)
    return len(chunks)


def retrieve(query: str, k: int | None = None) -> list[Document]:
    store = get_vector_store()
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k or settings.top_k},
    )
    return retriever.invoke(query)


def generate(question: str, context: str) -> str:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a precise medical assistant. Answer using only the context below. "
            "Cite sources with page numbers.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ])

    chain = prompt | llm
    return chain.invoke({"context": context, "question": question}).content
