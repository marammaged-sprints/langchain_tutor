import logging

import pytest

from langchain_tutor import rag_chain
from langchain_tutor.models import BookCitation, RAGResponse, RetrievedChunk
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


def test_retrieval_refusal_logs_query_and_scores(monkeypatch, caplog):
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}",
            content="content",
            source="Think Python",
            score=score,
        )
        for i, score in enumerate((0.2, 0.4, 0.5))
    ]

    def fake_retrieve(question, history, trace):
        trace["search_query"] = "rewritten recursion query"
        return chunks

    monkeypatch.setattr(rag_chain, "retrieve_for_question", fake_retrieve)

    with caplog.at_level(logging.INFO, logger=rag_chain.__name__):
        response = run_rag("How does recursion work?")

    assert response.query_type == "out_of_scope"
    message = next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("rag question=")
    )
    assert "question='How does recursion work?'" in message
    assert "rewritten='rewritten recursion query'" in message
    assert "retrieved=3 relevant=1" in message
    assert "min_top_k=3 threshold=0.350" in message
    assert "scores=[0.2, 0.4, 0.5]" in message
    assert "outcome=retrieval_refused" in message


def test_successful_answer_logs_retrieval_diagnostics(monkeypatch, caplog):
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}",
            content="A variable refers to a Python value.",
            source="Think Python",
            page=i + 1,
            score=score,
        )
        for i, score in enumerate((0.1, 0.2, 0.3))
    ]

    def fake_retrieve(question, history, trace):
        trace["search_query"] = "python variable"
        return chunks

    class SuccessfulChain:
        def invoke(self, _inputs):
            return RAGResponse(
                answer="A variable refers to a value.",
                query_type="definition",
                grounded=True,
                citations=[
                    BookCitation(
                        chunk_id="c0",
                        excerpt="A variable refers to a Python value.",
                    )
                ],
                retrieved_chunks=0,
            )

    class SuccessfulPrompt:
        def __or__(self, _model):
            return SuccessfulChain()

    monkeypatch.setattr(rag_chain, "retrieve_for_question", fake_retrieve)
    monkeypatch.setattr(rag_chain, "prompt", SuccessfulPrompt())
    monkeypatch.setattr(rag_chain, "get_chat_model", lambda: object())

    with caplog.at_level(logging.INFO, logger=rag_chain.__name__):
        response = run_rag("What is a variable?")

    assert response.grounded is True
    message = next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("rag question=")
    )
    assert "rewritten='python variable'" in message
    assert "retrieved=3 relevant=3" in message
    assert "scores=[0.1, 0.2, 0.3]" in message
    assert "citations=1 grounded=True outcome=answered" in message


@pytest.mark.parametrize(
    (
        "query_type",
        "grounded",
        "refusal_reason",
        "retrieval_gate_passed",
        "expected",
    ),
    [
        ("definition", True, None, True, "answered"),
        ("definition", False, None, True, "unverified"),
        ("out_of_scope", False, None, True, "model_refused"),
        (
            "out_of_scope",
            False,
            "Generation failed.",
            True,
            "generation_failed",
        ),
        (
            "out_of_scope",
            False,
            "Not enough relevant book context was retrieved.",
            False,
            "retrieval_refused",
        ),
    ],
)
def test_rag_outcome_keeps_failure_modes_distinct(
    query_type,
    grounded,
    refusal_reason,
    retrieval_gate_passed,
    expected,
):
    response = RAGResponse(
        answer="answer",
        query_type=query_type,
        grounded=grounded,
        retrieved_chunks=0,
        refusal_reason=refusal_reason,
    )

    assert rag_chain._rag_outcome(response, retrieval_gate_passed) == expected


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
