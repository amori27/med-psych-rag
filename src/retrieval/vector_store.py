from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.embeddings.embedder import get_embedding_function

PERSIST_DIR = Path("chroma_db")


def get_vector_store(collection_name: str = "med-psych") -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_function(),
        persist_directory=str(PERSIST_DIR),
    )


def add_documents(documents: list[Document], collection_name: str = "med-psych"):
    vector_store = get_vector_store(collection_name)
    vector_store.add_documents(documents)
    return vector_store
