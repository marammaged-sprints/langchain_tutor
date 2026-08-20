from langchain_tutor.retrieval.vector_store import get_embeddings


def test_embedding():
    embeddings = get_embeddings()

    result = embeddings.embed_query("What is a Python variable?")

    assert result
    assert len(result) > 0