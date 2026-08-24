import pytest
from ingestion.build_index import build_or_load_index


@pytest.mark.integration
def test_build_index():
    vector_store, count = build_or_load_index()

    assert vector_store is not None
    assert count > 0
