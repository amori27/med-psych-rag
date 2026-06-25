from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from src.core.config import settings

PERSIST_DIR = Path("data/chroma_db")


def get_embedding_function():
    return OpenAIEmbeddings(model=settings.embedding_model)


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embedding_function(),
        persist_directory=str(PERSIST_DIR),
    )
