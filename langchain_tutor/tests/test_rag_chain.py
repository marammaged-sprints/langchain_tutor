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

def test_answers_a_question_the_book_covers():
    response = run_rag("What is a variable?")

    assert response.grounded is True
    assert response.citations
    assert response.retrieved_chunks > 0