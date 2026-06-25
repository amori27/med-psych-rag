from pathlib import Path
from src.ingestion.pdf_processor import PDFProcessor


def test_extract_metadata_with_nonexistent_file():
    processor = PDFProcessor("nonexistent.pdf")
    try:
        processor.extract_metadata()
    except Exception:
        pass


def test_extract_text_returns_string():
    path = Path("data/sample.pdf")
    if path.exists():
        processor = PDFProcessor(path)
        text = processor.extract_text()
        assert isinstance(text, str)
