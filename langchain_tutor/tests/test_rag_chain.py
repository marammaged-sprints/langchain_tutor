import pytest

from langchain_tutor.models import RetrievedChunk
from langchain_tutor.rag_chain import select_relevant, run_rag


def test_gate_rejects_when_nothing_clears_threshold():
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}",
            content="x",
            source="s",
            page=1,
            score=0.95,
        )
        for i in range(5)
    ]

    assert select_relevant(chunks) == []


@pytest.mark.integration
def test_answers_a_question_the_book_covers():
    response = run_rag("What is a variable?")

    assert response.grounded is True
    assert response.citations
    assert response.retrieved_chunks > 0


@pytest.mark.integration
def test_refuses_question_outside_the_book():
    response = run_rag("What is the capital of France?")

    assert response.grounded is False
    assert response.query_type == "out_of_scope"
    assert response.citations == []