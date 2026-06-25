import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def sample_pdf() -> Path:
    return Path("data/sample/sample.pdf")
