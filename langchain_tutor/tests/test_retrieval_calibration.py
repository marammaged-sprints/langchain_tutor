import pytest

from langchain_tutor.evaluation.calibrate_retrieval import (
    min_top_k_score,
    recommend_threshold,
    score_threshold,
)
from langchain_tutor.models import RetrievedChunk


def _chunk(score: float | None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(score),
        content="content",
        source="Think Python",
        score=score,
    )


def test_min_top_k_score_uses_the_gate_deciding_distance():
    chunks = [_chunk(0.3), _chunk(0.1), _chunk(0.2), _chunk(None)]

    assert min_top_k_score(chunks, min_top_k=3) == pytest.approx(0.3)


def test_recommend_threshold_separates_measured_distributions():
    in_scope_scores = [0.10, 0.20, 0.25]
    out_of_scope_scores = [0.60, 0.70]

    threshold, accuracy, in_correct, out_correct = recommend_threshold(
        in_scope_scores,
        out_of_scope_scores,
    )

    assert 0.25 < threshold < 0.60
    assert accuracy == 1.0
    assert in_correct == 3
    assert out_correct == 2
    assert score_threshold(
        threshold,
        in_scope_scores,
        out_of_scope_scores,
    ) == (1.0, 3, 2)


def test_recommend_threshold_requires_both_question_classes():
    with pytest.raises(
        ValueError,
        match="Both in-scope and out-of-scope scores are required",
    ):
        recommend_threshold([0.2], [])
