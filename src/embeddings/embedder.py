from langchain_huggingface import HuggingFaceEmbeddings
from src.config import Config


def get_embedding_function() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=Config.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
