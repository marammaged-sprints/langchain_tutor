import pytest
from retrieval.vector_store import get_vector_store


@pytest.mark.integration
def test_vector_store():
    vector_store = get_vector_store()

    assert vector_store is not None
    assert vector_store._collection.name
    assert vector_store._collection.count() > 0
