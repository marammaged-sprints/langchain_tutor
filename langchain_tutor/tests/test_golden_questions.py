import pytest

from langchain_tutor.evaluation.golden_questions import IN_SCOPE, OUT_OF_SCOPE
from langchain_tutor.rag_chain import run_rag


@pytest.mark.integration
@pytest.mark.parametrize(
    ("question", "expected_terms"),
    IN_SCOPE,
    ids=[question for question, _ in IN_SCOPE],
)
def test_in_scope_questions_are_answered(question, expected_terms):
    response = run_rag(question)

    assert response.grounded is True, (
        f"refused a question the book covers: {question}; "
        f"reason={response.refusal_reason}"
    )
    assert response.citations
    assert any(
        term.lower() in response.answer.lower()
        for term in expected_terms
    ), f"answer did not contain any expected term {expected_terms}: {question}"


@pytest.mark.integration
@pytest.mark.parametrize(
    "question",
    OUT_OF_SCOPE,
    ids=OUT_OF_SCOPE,
)
def test_out_of_scope_questions_are_refused(question):
    response = run_rag(question)

    assert response.query_type == "out_of_scope", (
        f"answered something outside the book: {question}"
    )
    assert response.grounded is False
