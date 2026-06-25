import fitz
from pathlib import Path


class PDFProcessor:
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def extract_text(self) -> str:
        doc = fitz.open(str(self.file_path))
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text

    def extract_text_by_page(self) -> list[dict]:
        doc = fitz.open(str(self.file_path))
        pages = []
        for i, page in enumerate(doc):
            pages.append({"page": i + 1, "text": page.get_text()})
        doc.close()
        return pages

    def extract_metadata(self) -> dict:
        doc = fitz.open(str(self.file_path))
        meta = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "pages": len(doc),
        }
        doc.close()
        return meta
