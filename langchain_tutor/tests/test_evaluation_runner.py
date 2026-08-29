from langchain_tutor.evaluation import run_questions
from langchain_tutor.evaluation.run_questions import (
    looks_like_refusal,
    parse_question_cases,
)
from langchain_tutor.models import RAGResponse


def test_parses_direct_and_conversational_cases():
    lines = ["A. Direct retrieval"]
    lines.extend(f"{number}. Question {number}?" for number in range(1, 81))
    lines.append("C. Conversational follow-up tests")

    for number in range(81, 91):
        lines.extend(
            [
                str(number),
                "First:",
                f"First question {number}?",
                "Then:",
                f"Follow-up question {number}?",
            ]
        )

    lines.append("D. Unsupported tests")
    lines.extend(f"{number}. Question {number}?" for number in range(91, 101))

    cases = parse_question_cases("\n".join(lines))

    assert len(cases) == 100
    assert cases[0]["questions"] == ["Question 1?"]
    assert cases[80]["questions"] == [
        "First question 81?",
        "Follow-up question 81?",
    ]
    assert cases[89]["id"] == 90
    assert cases[90]["expected_should_answer"] is False


def test_detects_refusal_hidden_in_a_grounded_answer():
    assert looks_like_refusal("I cannot answer that from the book.") is True
    assert looks_like_refusal("I can't answer from this context.") is True
    assert looks_like_refusal("A variable is a name for a value.") is False


def test_runner_preserves_dropped_citation_count(monkeypatch):
    response = RAGResponse(
        answer="An answer whose citation could not be verified.",
        query_type="conceptual",
        grounded=False,
        citations=[],
        dropped_citation_count=2,
        retrieved_chunks=5,
    )
    monkeypatch.setattr(
        run_questions,
        "call_rag_with_retries",
        lambda **kwargs: (response, {"search_query": "dictionary lists"}),
    )
    case = {
        "id": 72,
        "category": "multi_chunk_synthesis",
        "questions": ["Compare setdefault with defaultdict."],
        "expected_should_answer": True,
    }

    result = run_questions.evaluate_case(case, attempts=1, retry_delay=0)

    assert result["turns"][0]["dropped_citation_count"] == 2
    assert result["turns"][0]["answer_displayed"] is False


def test_runner_records_uncited_answer_as_displayed(monkeypatch):
    response = RAGResponse(
        answer="A correct answer without citations.",
        query_type="conceptual",
        grounded=False,
        citations=[],
        dropped_citation_count=0,
        retrieved_chunks=5,
    )
    monkeypatch.setattr(
        run_questions,
        "call_rag_with_retries",
        lambda **kwargs: (response, {"search_query": "dictionary lists"}),
    )
    case = {
        "id": 72,
        "category": "multi_chunk_synthesis",
        "questions": ["Compare setdefault with defaultdict."],
        "expected_should_answer": True,
    }

    result = run_questions.evaluate_case(case, attempts=1, retry_delay=0)

    turn = result["turns"][0]
    assert turn["answer_displayed"] is True
    assert turn["actual_should_answer"] is True
    assert result["passed"] is False
