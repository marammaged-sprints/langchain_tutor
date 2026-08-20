from langchain_tutor.retrieval.vector_store import get_vector_store


def test_vector_store():
    vector_store = get_vector_store()

    assert vector_store is not None
    assert vector_store._collection.name
    assert vector_store._collection.count() > 0