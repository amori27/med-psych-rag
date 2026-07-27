"""Orchestrates the full RAG pipeline: ingest, retrieve, generate.

The pipeline extracts text from PDFs, chunks it into overlapping segments,
indexes the chunks in a vector store, and uses an LLM to generate answers
from the most relevant retrieved documents.
"""

from pathlib import Path
from langchain_core.documents import Document
from src.core.config import settings
from src.utils.pdf_processor import PDFProcessor
from src.utils.chunker import chunk_documents
from src.utils.vector_store import get_vector_store


def ingest_pdf(file_path: str | Path) -> int:
    """Extract text from a PDF, chunk it, and index into the vector store.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Number of chunks created and indexed.
    """
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
    """Retrieve the top-k documents most similar to *query*.

    Args:
        query: Natural-language search query.
        k: Number of results (defaults to ``settings.top_k``).

    Returns:
        List of LangChain ``Document`` objects ordered by similarity.
    """
    store = get_vector_store()
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k or settings.top_k},
    )
    return retriever.invoke(query)


def _get_llm():
    """Return the configured LLM instance based on ``settings.llm_provider``."""
    match settings.llm_provider:
        case "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=settings.gemini_api_key,
                temperature=0,
            )
        case "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=settings.llm_model,
                base_url=settings.ollama_base_url,
                temperature=0,
            )
        case _:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=settings.llm_model,
                api_key=settings.openai_api_key,
                temperature=0,
            )


def generate(question: str, context: str) -> str:
    """Generate an answer to *question* using *context* from retrieved docs.

    Args:
        question: User question.
        context: Concatenated text from retrieved documents.

    Returns:
        LLM-generated answer string.
    """
    from langchain_core.prompts import ChatPromptTemplate

    llm = _get_llm()
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
