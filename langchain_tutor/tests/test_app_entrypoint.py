from pathlib import Path

from streamlit.testing.v1 import AppTest

from langchain_tutor import app as tutor_app
from langchain_tutor.app import main
from langchain_tutor.models import RAGResponse


def test_package_app_exposes_main():
    assert callable(main)


def test_root_streamlit_entrypoint_starts_without_error():
    entrypoint = Path(__file__).parents[2] / "streamlit_app.py"

    app = AppTest.from_file(entrypoint).run(timeout=30)

    assert not app.exception


def test_out_of_scope_response_displays_the_gate_refusal(monkeypatch):
    markdown_messages = []
    warning_messages = []
    monkeypatch.setattr(tutor_app.st, "markdown", markdown_messages.append)
    monkeypatch.setattr(tutor_app.st, "warning", warning_messages.append)

    response = RAGResponse(
        answer="I can't answer that from the book.",
        query_type="out_of_scope",
        grounded=False,
        retrieved_chunks=0,
        refusal_reason="Not enough relevant book context was retrieved.",
    )

    handled = tutor_app.render_refusal(response)

    assert handled is True
    assert markdown_messages == ["I can't answer that from the book."]
    assert warning_messages == []


def test_invalid_citations_display_verification_warning(monkeypatch):
    markdown_messages = []
    warning_messages = []
    monkeypatch.setattr(tutor_app.st, "markdown", markdown_messages.append)
    monkeypatch.setattr(tutor_app.st, "warning", warning_messages.append)

    response = RAGResponse(
        answer="An answer that could not be verified.",
        query_type="conceptual",
        grounded=False,
        dropped_citation_count=1,
        retrieved_chunks=3,
        refusal_reason="The answer was not supported by the retrieved context.",
    )

    handled = tutor_app.render_refusal(response)

    assert handled is True
    assert markdown_messages == []
    assert warning_messages == [
        "I couldn't verify this answer against the book because one or "
        "more citations were invalid. Try rephrasing the question."
    ]


def test_uncited_answer_displays_caveat_and_remains_visible(monkeypatch):
    info_messages = []
    warning_messages = []
    monkeypatch.setattr(tutor_app.st, "info", info_messages.append)
    monkeypatch.setattr(tutor_app.st, "warning", warning_messages.append)

    response = RAGResponse(
        answer="A correct answer generated from retrieved book context.",
        query_type="conceptual",
        grounded=False,
        citations=[],
        dropped_citation_count=0,
        retrieved_chunks=5,
    )

    handled = tutor_app.render_refusal(response)

    assert handled is False
    assert info_messages == [
        "Answer produced from the book, but without specific citations."
    ]
    assert warning_messages == []


def test_generation_failure_displays_its_own_error(monkeypatch):
    error_messages = []
    markdown_messages = []
    warning_messages = []
    monkeypatch.setattr(tutor_app.st, "error", error_messages.append)
    monkeypatch.setattr(tutor_app.st, "markdown", markdown_messages.append)
    monkeypatch.setattr(tutor_app.st, "warning", warning_messages.append)

    response = RAGResponse(
        answer=(
            "Something went wrong while generating the answer. "
            "Please try again."
        ),
        query_type="out_of_scope",
        grounded=False,
        retrieved_chunks=3,
        refusal_reason="Generation failed.",
    )

    handled = tutor_app.render_refusal(response)

    assert handled is True
    assert error_messages == [response.answer]
    assert markdown_messages == []
    assert warning_messages == []
