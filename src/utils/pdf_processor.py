"""PDF text extraction using PyMuPDF (fitz).

Provides per-page and full-document text extraction plus basic metadata.
"""

from pathlib import Path
import fitz


class PDFProcessor:
    """Extract text and metadata from a PDF file.

    Args:
        file_path: Path to a valid PDF file.
    """

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def extract_text(self) -> str:
        """Return the full text of the PDF as a single string."""
        doc = fitz.open(str(self.file_path))
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text

    def extract_text_by_page(self) -> list[dict]:
        """Return a list of ``{"page": int, "text": str}`` dicts.

        Page numbers are 1-indexed.
        """
        doc = fitz.open(str(self.file_path))
        pages = []
        for i, page in enumerate(doc):
            pages.append({"page": i + 1, "text": page.get_text()})
        doc.close()
        return pages

    def extract_metadata(self) -> dict:
        """Return basic PDF metadata: title, author, and page count."""
        doc = fitz.open(str(self.file_path))
        meta = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "pages": len(doc),
        }
        doc.close()
        return meta
