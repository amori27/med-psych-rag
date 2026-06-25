from pathlib import Path
from langchain_chroma import Chroma
from src.core.config import settings

PERSIST_DIR = Path("data/chroma_db")


def get_embedding_function():
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
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embedding_function(),
        persist_directory=str(PERSIST_DIR),
    )
