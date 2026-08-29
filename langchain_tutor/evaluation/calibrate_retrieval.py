"""Measure the relevance gate against the curated golden questions."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite
from statistics import mean

from langchain_tutor.config import settings
from langchain_tutor.evaluation.golden_questions import IN_SCOPE, OUT_OF_SCOPE
from langchain_tutor.models import RetrievedChunk
from langchain_tutor.rag_chain import retrieve_for_question


@dataclass(frozen=True)
class CalibrationResult:
    label: str
    question: str
    scores: tuple[float, ...]
    gate_score: float


def min_top_k_score(
    chunks: list[RetrievedChunk],
    min_top_k: int,
) -> float:
    """Return the distance that decides whether enough chunks clear the gate."""
    scores = sorted(
        chunk.score
        for chunk in chunks
        if chunk.score is not None
    )
    if len(scores) < min_top_k:
        return inf
    return scores[min_top_k - 1]


def score_threshold(
    threshold: float,
    in_scope_scores: list[float],
    out_of_scope_scores: list[float],
) -> tuple[float, int, int]:
    """Return balanced accuracy and correct counts for a candidate threshold."""
    in_scope_correct = sum(score <= threshold for score in in_scope_scores)
    out_of_scope_correct = sum(
        score > threshold for score in out_of_scope_scores
    )
    in_scope_rate = in_scope_correct / len(in_scope_scores)
    out_of_scope_rate = out_of_scope_correct / len(out_of_scope_scores)
    balanced_accuracy = (in_scope_rate + out_of_scope_rate) / 2
    return balanced_accuracy, in_scope_correct, out_of_scope_correct


def recommend_threshold(
    in_scope_scores: list[float],
    out_of_scope_scores: list[float],
) -> tuple[float, float, int, int]:
    """Choose the measured threshold with the best balanced accuracy."""
    if not in_scope_scores or not out_of_scope_scores:
        raise ValueError("Both in-scope and out-of-scope scores are required")

    observed = sorted(
        {
            score
            for score in in_scope_scores + out_of_scope_scores
            if isfinite(score)
        }
    )
    if not observed:
        raise ValueError("At least one measured gate score is required")

    candidates = [observed[0] - 1e-6]
    candidates.extend(
        (left + right) / 2
        for left, right in zip(observed, observed[1:])
    )
    candidates.append(observed[-1] + 1e-6)

    ranked = []
    for threshold in candidates:
        accuracy, in_correct, out_correct = score_threshold(
            threshold,
            in_scope_scores,
            out_of_scope_scores,
        )
        ranked.append(
            (accuracy, out_correct, in_correct, -threshold, threshold)
        )

    accuracy, out_correct, in_correct, _, threshold = max(ranked)
    return threshold, accuracy, in_correct, out_correct


def measure_questions() -> list[CalibrationResult]:
    results = []
    cases = [
        ("in_scope", question)
        for question, _ in IN_SCOPE
    ]
    cases.extend(
        ("out_of_scope", question)
        for question in OUT_OF_SCOPE
    )

    for label, question in cases:
        chunks = retrieve_for_question(question)
        scores = tuple(
            sorted(
                chunk.score
                for chunk in chunks
                if chunk.score is not None
            )
        )
        results.append(
            CalibrationResult(
                label=label,
                question=question,
                scores=scores,
                gate_score=min_top_k_score(chunks, settings.min_top_k),
            )
        )

    return results


def main() -> None:
    results = measure_questions()

    print(
        "Lower Chroma distance scores are more relevant. "
        f"Gate: top_k={settings.top_k}, min_top_k={settings.min_top_k}, "
        f"threshold={settings.retrieval_score_threshold:.4f}"
    )
    for result in results:
        scores = ", ".join(f"{score:.4f}" for score in result.scores)
        passes = result.gate_score <= settings.retrieval_score_threshold
        print(
            f"{result.label:12} gate_score={result.gate_score:.4f} "
            f"passes={str(passes):5} scores=[{scores}]\n"
            f"  {result.question}"
        )

    in_scope_scores = [
        result.gate_score
        for result in results
        if result.label == "in_scope"
    ]
    out_of_scope_scores = [
        result.gate_score
        for result in results
        if result.label == "out_of_scope"
    ]
    current = score_threshold(
        settings.retrieval_score_threshold,
        in_scope_scores,
        out_of_scope_scores,
    )
    recommendation = recommend_threshold(
        in_scope_scores,
        out_of_scope_scores,
    )

    print("\nCurrent gate")
    print(
        f"  in-scope accepted: {current[1]}/{len(in_scope_scores)}\n"
        f"  out-of-scope refused: {current[2]}/{len(out_of_scope_scores)}\n"
        f"  balanced accuracy: {current[0]:.1%}"
    )
    print("Measured recommendation")
    print(
        f"  threshold: {recommendation[0]:.4f}\n"
        f"  in-scope accepted: {recommendation[2]}/{len(in_scope_scores)}\n"
        f"  out-of-scope refused: {recommendation[3]}/{len(out_of_scope_scores)}\n"
        f"  balanced accuracy: {recommendation[1]:.1%}\n"
        f"  mean in-scope gate score: {mean(in_scope_scores):.4f}\n"
        f"  mean out-of-scope gate score: {mean(out_of_scope_scores):.4f}"
    )


if __name__ == "__main__":
    main()
