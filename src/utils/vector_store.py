"""ChromaDB vector store setup for document embeddings.

Provides functions to get the embedding function and vector store instance
based on the configured LLM provider.
"""

from pathlib import Path
from langchain_chroma import Chroma
from src.core.config import settings

PERSIST_DIR = Path("data/chroma_db")


def get_embedding_function():
    """Return the embedding function matching the configured LLM provider.

    Uses Google Generative AI embeddings for Gemini, and OpenAI embeddings
    for all other providers.
    """
    if settings.llm_provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.gemini_api_key,
        )
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


def get_vector_store() -> Chroma:
    """Return a Chroma vector store with the configured collection and embeddings."""
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embedding_function(),
        persist_directory=str(PERSIST_DIR),
    )
