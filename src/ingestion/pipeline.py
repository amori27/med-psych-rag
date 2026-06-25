from pathlib import Path
from langchain_core.documents import Document
from src.ingestion.pdf_processor import PDFProcessor
from src.ingestion.text_splitter import split_documents
from src.retrieval.vector_store import add_documents


def process_pdf(file_path: str | Path, collection_name: str = "med-psych") -> int:
    processor = PDFProcessor(file_path)
    pages = processor.extract_text_by_page()

    documents = []
    for page in pages:
        documents.append(
            Document(
                page_content=page["text"],
                metadata={
                    "source": str(file_path),
                    "page": page["page"],
                    "title": processor.extract_metadata().get("title", ""),
                },
            )
        )

    chunks = split_documents(documents)
    add_documents(chunks, collection_name)
    return len(chunks)
