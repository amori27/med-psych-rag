from src.core.config import settings


def test_settings_defaults():
    assert settings.chunk_size > 0
    assert settings.chunk_overlap >= 0
    assert settings.top_k > 0
    assert settings.collection_name == "med-psych"
