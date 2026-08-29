import logging

import pytest

from langchain_tutor import rag_chain
from langchain_tutor.models import RetrievedChunk
from langchain_tutor.rag_chain import get_store, select_relevant, run_rag


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


def test_generation_failure_returns_safe_response(monkeypatch, caplog):
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}",
            content="Relevant Python content.",
            source="Think Python",
            page=i + 1,
            score=0.1,
        )
        for i in range(3)
    ]

    class FailingChain:
        def invoke(self, _inputs):
            raise RuntimeError("invalid structured model output")

    class FailingPrompt:
        def __or__(self, _model):
            return FailingChain()

    monkeypatch.setattr(
        rag_chain,
        "retrieve_for_question",
        lambda question, history, trace: chunks,
    )
    monkeypatch.setattr(rag_chain, "prompt", FailingPrompt())
    monkeypatch.setattr(rag_chain, "get_chat_model", lambda: object())

    trace = {}
    with caplog.at_level(logging.ERROR, logger=rag_chain.__name__):
        response = run_rag("What is a Python variable?", trace=trace)

    assert response == rag_chain.RAGResponse(
        answer=(
            "Something went wrong while generating the answer. "
            "Please try again."
        ),
        query_type="out_of_scope",
        grounded=False,
        citations=[],
        retrieved_chunks=3,
        refusal_reason="Generation failed.",
    )
    assert trace["timings_seconds"]["generation"] >= 0
    assert trace["timings_seconds"]["total"] >= 0

    log_record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Structured generation failed"
    )
    assert log_record.exc_info is not None


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
