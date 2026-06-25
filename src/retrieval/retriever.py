from src.config import Config
from src.retrieval.vector_store import get_vector_store


def get_retriever(collection_name: str = "med-psych"):
    vector_store = get_vector_store(collection_name)
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": Config.top_k},
    )


def retrieve(query: str, collection_name: str = "med-psych") -> list:
    retriever = get_retriever(collection_name)
    return retriever.invoke(query)
