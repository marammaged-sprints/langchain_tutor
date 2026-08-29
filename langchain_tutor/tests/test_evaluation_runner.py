from langchain_tutor.evaluation.run_questions import (
    looks_like_refusal,
    parse_question_cases,
)


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
