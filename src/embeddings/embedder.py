from langchain_openai import OpenAIEmbeddings
from src.config import Config


def get_embedding_function() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=Config.embedding_model,
        api_key=Config.openai_api_key,
    )
