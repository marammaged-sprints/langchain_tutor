from langchain_tutor.ingestion.build_index import build_or_load_index


def test_build_index():
    vector_store, count = build_or_load_index()

    assert vector_store is not None
    assert count > 0