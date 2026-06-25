import pytest
from pathlib import Path
from src.utils.pdf_processor import PDFProcessor

SAMPLE_PDF = Path("data/sample/sample.pdf")


@pytest.fixture
def processor():
    if not SAMPLE_PDF.exists():
        pytest.skip("Sample PDF not found")
    return PDFProcessor(SAMPLE_PDF)


def test_extract_metadata(processor):
    meta = processor.extract_metadata()
    assert "pages" in meta
    assert meta["pages"] >= 1


def test_extract_text(processor):
    text = processor.extract_text()
    assert isinstance(text, str)
    assert len(text) > 0


def test_extract_text_by_page(processor):
    pages = processor.extract_text_by_page()
    assert len(pages) >= 1
    assert "page" in pages[0]
    assert "text" in pages[0]


def test_file_not_found():
    with pytest.raises(Exception):
        PDFProcessor("nonexistent.pdf").extract_text()
