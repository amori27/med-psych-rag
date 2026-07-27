"""Document chunking utilities for the RAG pipeline.

Splits LangChain ``Document`` objects into smaller overlapping chunks
suitable for embedding and retrieval.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core.config import settings


def chunk_documents(documents: list) -> list:
    """Split *documents* into overlapping chunks.

    Uses ``RecursiveCharacterTextSplitter`` with the chunk size and overlap
    configured in ``settings``.

    Args:
        documents: List of LangChain ``Document`` objects.

    Returns:
        Flat list of chunked ``Document`` objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
