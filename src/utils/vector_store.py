from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from src.core.config import settings

PERSIST_DIR = Path("data/chroma_db")


def get_embedding_function():
    return HuggingFaceBgeEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embedding_function(),
        persist_directory=str(PERSIST_DIR),
    )
