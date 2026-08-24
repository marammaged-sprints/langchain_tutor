import pytest

from config import settings


@pytest.mark.integration
def test_config():
    assert settings.book_path.exists()
    assert settings.embedding_model
    assert settings.google_api_key
    assert settings.persist_directory
    assert settings.collection_name
