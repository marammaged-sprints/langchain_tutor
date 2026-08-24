import pytest

import rag_chain
from models import RetrievedChunk
from rag_chain import get_store, select_relevant, run_rag


def test_get_store_builds_or_loads_index(monkeypatch):
    expected_store = object()
    calls = 0

    def fake_build_or_load_index():
        nonlocal calls
        calls += 1
        return expected_store, 612

    monkeypatch.setattr(rag_chain, "_vector_store", None)
    monkeypatch.setattr(
        rag_chain,
        "build_or_load_index",
        fake_build_or_load_index,
    )

    assert get_store() is expected_store
    assert get_store() is expected_store
    assert calls == 1


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
