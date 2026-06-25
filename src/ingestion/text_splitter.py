from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import Config


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=Config.chunk_size,
        chunk_overlap=Config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def split_documents(documents: list) -> list:
    splitter = create_text_splitter()
    return splitter.split_documents(documents)
