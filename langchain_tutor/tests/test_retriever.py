
import pytest
from langchain_tutor.ingestion.build_index import build_or_load_index
from langchain_tutor.retrieval.retriever import retrieve


@pytest.mark.integration
def test_retriever():
    vector_store, count = build_or_load_index()

    assert count > 0

    results = retrieve(
        vector_store,
        "What is a variable in Python?"
    )

    assert results
    assert len(results) > 0

    for result in results:
        assert result.chunk_id
        assert result.content
        assert result.source
        assert result.page is not None
