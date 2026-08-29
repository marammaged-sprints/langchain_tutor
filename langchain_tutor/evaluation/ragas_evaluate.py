from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import os
import statistics
import sys
import time
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_tutor.config import settings
from langchain_tutor.evaluation.evaluation_prompt import (
    COMPLETENESS_RUBRICS,
    CORRECTNESS_RUBRICS,
    EVALUATION_PROMPT_VERSION,
)


# Avoid Ragas' analytics background worker in a command-line batch job. Apart
# from keeping evaluation data local, this lets the Python process exit cleanly.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")


METRIC_NAMES = (
    "retrieval_relevance",
    "answer_correctness",
    "faithfulness",
    "answer_completeness",
    "response_time",
)

REFUSAL_PREFIXES = (
    "i can't answer",
    "i cannot answer",
    "i can’t answer",
    "i'm unable to answer",
    "i am unable to answer",
)

FALLBACK_ANSWER_PREFIXES = (
    "i found relevant passages, but i couldn't verify",
    "i found relevant passages, but i could not verify",
)


def _install_ragas_vertexai_compatibility() -> None:
    """Work around the released Ragas/modern LangChain import mismatch.

    Ragas 0.3.9 imports ``ChatVertexAI`` from a module removed by modern
    ``langchain-community`` even when VertexAI is not used. This no-op type only
    satisfies Ragas' provider type check. The evaluator itself uses Gemini via
    ``ChatGoogleGenerativeAI``.
    """

    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    try:
        __import__(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise

        compatibility_module = types.ModuleType(module_name)
        compatibility_module.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[module_name] = compatibility_module


_metrics: dict[str, Any] | None = None


@dataclass(frozen=True)
class _FallbackSingleTurnSample:
    """Duck-typed sample used only with injected metrics in unit tests."""

    user_input: str
    response: str
    retrieved_contexts: list[str]


def _missing_ragas(exc: ModuleNotFoundError) -> bool:
    return bool(exc.name and (
        exc.name == "ragas" or exc.name.startswith("ragas.")
    ))


def _ragas_available() -> bool:
    try:
        return importlib.util.find_spec("ragas") is not None
    except ModuleNotFoundError as exc:
        if _missing_ragas(exc):
            return False
        raise


def _ragas_install_error() -> RuntimeError:
    return RuntimeError(
        "Ragas evaluation requires the optional 'ragas' package. "
        "Install the project evaluation dependencies with "
        "'python -m pip install -r requirements.txt'."
    )


def _create_single_turn_sample(**values: Any) -> Any:
    """Create a Ragas sample lazily, with a test-only fallback if absent."""
    if not _ragas_available():
        return _FallbackSingleTurnSample(**values)

    _install_ragas_vertexai_compatibility()
    try:
        from ragas.dataset_schema import SingleTurnSample
    except ModuleNotFoundError as exc:
        if not _missing_ragas(exc):
            raise
        return _FallbackSingleTurnSample(**values)

    return SingleTurnSample(**values)


def get_ragas_metrics() -> dict[str, Any]:
    """Build and cache the four LLM-backed Ragas metrics."""

    global _metrics

    if _metrics is None:
        if not _ragas_available():
            raise _ragas_install_error()

        _install_ragas_vertexai_compatibility()
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import (
                Faithfulness,
                LLMContextPrecisionWithoutReference,
                RubricsScore,
            )
        except ModuleNotFoundError as exc:
            if not _missing_ragas(exc):
                raise
            raise _ragas_install_error() from exc

        judge = ChatGoogleGenerativeAI(
            model=settings.chat_model,
            google_api_key=settings.require_api_key(),
            temperature=0.0,
        )
        ragas_llm = LangchainLLMWrapper(judge)
        _metrics = {
            "retrieval_relevance": LLMContextPrecisionWithoutReference(
                llm=ragas_llm
            ),
            "answer_correctness": RubricsScore(
                name="answer_correctness_without_reference",
                rubrics=CORRECTNESS_RUBRICS,
                llm=ragas_llm,
            ),
            "faithfulness": Faithfulness(llm=ragas_llm),
            "answer_completeness": RubricsScore(
                name="answer_completeness_without_reference",
                rubrics=COMPLETENESS_RUBRICS,
                llm=ragas_llm,
            ),
        }

    return _metrics


def _chunk_contents(chunks: Any) -> list[str]:
    if not isinstance(chunks, list):
        return []

    contents = []
    for chunk in chunks:
        if isinstance(chunk, str) and chunk.strip():
            contents.append(chunk.strip())
        elif isinstance(chunk, dict):
            content = chunk.get("content")
            if isinstance(content, str) and content.strip():
                contents.append(content.strip())
    return contents


def _citation_excerpts(citations: Any) -> list[str]:
    if not isinstance(citations, list):
        return []

    return [
        citation["excerpt"].strip()
        for citation in citations
        if isinstance(citation, dict)
        and isinstance(citation.get("excerpt"), str)
        and citation["excerpt"].strip()
    ]


def prepare_records(
    answer_payload: dict[str, Any],
    fallback_response_time: float | None = None,
) -> list[dict[str, Any]]:
    """Flatten answer cases and select the best available retrieval evidence."""

    records: list[dict[str, Any]] = []

    for case in answer_payload.get("results", []):
        for turn in case.get("turns", []):
            trace = turn.get("trace") or {}
            timings = trace.get("timings_seconds") or {}
            response_time = timings.get("total", fallback_response_time)
            if timings.get("total") is not None:
                response_time_source = "trace.timings_seconds.total"
            elif fallback_response_time is not None:
                response_time_source = "fallback_average"
            else:
                response_time_source = "none"

            contexts = _chunk_contents(trace.get("relevant_chunks"))
            context_source = "trace.relevant_chunks"
            if not contexts:
                contexts = _chunk_contents(trace.get("retrieved_chunks"))
                context_source = "trace.retrieved_chunks"
            if not contexts:
                contexts = _citation_excerpts(turn.get("citations"))
                context_source = (
                    "citation_excerpts_proxy" if contexts else "none"
                )

            records.append(
                {
                    "case_id": case["id"],
                    "turn": turn["turn"],
                    "category": case["category"],
                    "expected_should_answer": case[
                        "expected_should_answer"
                    ],
                    "question": turn["question"],
                    "history": turn.get("history", ""),
                    "answer": turn.get("answer") or "",
                    "query_type": turn.get("query_type"),
                    "grounded": bool(turn.get("grounded")),
                    "refusal_reason": turn.get("refusal_reason"),
                    "citations": turn.get("citations", []),
                    "search_query": trace.get("search_query"),
                    "retrieved_contexts": contexts,
                    "context_source": context_source,
                    "response_time_seconds": response_time,
                    "response_time_source": response_time_source,
                }
            )

    return records


def build_evaluation_question(record: dict[str, Any]) -> str:
    expected = (
        "Answer the question using the Think Python book context."
        if record["expected_should_answer"]
        else (
            "Refuse clearly because this question is outside the supported "
            "Think Python book content; do not answer from outside knowledge."
        )
    )
    history = record.get("history") or "(no previous conversation)"
    return (
        f"Expected behavior: {expected}\n"
        f"Conversation history: {history}\n"
        f"Question: {record['question']}"
    )


def normalize_rubric_score(raw_score: float) -> float:
    """Convert a Ragas RubricsScore result from 1-5 to the 0-1 scale."""

    return max(0.0, min(1.0, (raw_score - 1.0) / 4.0))


def score_response_time(
    seconds: float | None,
    source: str = "none",
) -> dict[str, Any]:
    """Score measured end-to-end latency with a deterministic 0-1 rubric."""

    if seconds is None:
        return {
            "score": None,
            "raw_seconds": None,
            "evaluator": "deterministic_latency_rubric",
            "status": "not_measured",
            "source": source,
        }

    if seconds <= 3:
        score = 1.0
    elif seconds <= 6:
        score = 0.75
    elif seconds <= 10:
        score = 0.5
    elif seconds <= 15:
        score = 0.25
    else:
        score = 0.0

    return {
        "score": score,
        "raw_seconds": round(float(seconds), 4),
        "evaluator": "deterministic_latency_rubric",
        "status": "scored",
        "source": source,
    }


def _is_refusal(answer: str) -> bool:
    return answer.strip().lower().startswith(REFUSAL_PREFIXES)


def behavior_passed(record: dict[str, Any]) -> bool:
    """Check the visible answer contract independently of Ragas scores."""

    answer = record["answer"].strip()
    normalized = answer.lower()
    citations = record.get("citations") or []

    if record["expected_should_answer"]:
        return bool(
            answer
            and not _is_refusal(answer)
            and not normalized.startswith(FALLBACK_ANSWER_PREFIXES)
            and record.get("grounded")
            and citations
        )

    return bool(
        _is_refusal(answer)
        and not record.get("grounded")
        and not citations
    )


async def _score_metric_with_retries(
    metric: Any,
    sample: Any,
    attempts: int,
    retry_delay: float,
    metric_timeout: float,
) -> float:
    for attempt in range(1, attempts + 1):
        try:
            result = float(
                await metric.single_turn_ascore(
                    sample,
                    timeout=metric_timeout,
                )
            )
            if math.isnan(result):
                raise ValueError("Ragas returned NaN")
            return result
        except Exception as exc:
            if attempt == attempts:
                raise

            delay = retry_delay * attempt
            print(
                f"Ragas attempt {attempt}/{attempts} failed "
                f"({type(exc).__name__}); retrying in {delay:g}s",
                flush=True,
            )
            await asyncio.sleep(delay)

    raise AssertionError("Retry loop ended unexpectedly")


async def evaluate_record_async(
    record: dict[str, Any],
    metrics: dict[str, Any],
    attempts: int,
    retry_delay: float,
    metric_timeout: float = 180.0,
) -> dict[str, Any]:
    """Evaluate one answer turn using Ragas plus measured latency."""

    contexts = record["retrieved_contexts"]
    sample = _create_single_turn_sample(
        user_input=build_evaluation_question(record),
        response=record["answer"],
        retrieved_contexts=contexts,
    )

    if contexts:
        retrieval_evaluator = (
            "ragas.metrics.LLMContextPrecisionWithoutReference"
        )
        retrieval_status = "scored"
    else:
        raw_retrieval = 0.0
        retrieval_evaluator = "deterministic_no_context_rule"
        retrieval_status = "no_context_to_evaluate"
    if contexts:
        faithfulness_evaluator = "ragas.metrics.Faithfulness"
        faithfulness_status = "scored"
    else:
        raw_faithfulness = 1.0 if _is_refusal(record["answer"]) else 0.0
        faithfulness_evaluator = "deterministic_no_context_rule"
        faithfulness_status = "no_context_to_evaluate"
    metric_jobs = {
        "answer_correctness": _score_metric_with_retries(
            metrics["answer_correctness"],
            sample,
            attempts,
            retry_delay,
            metric_timeout,
        ),
        "answer_completeness": _score_metric_with_retries(
            metrics["answer_completeness"],
            sample,
            attempts,
            retry_delay,
            metric_timeout,
        ),
    }
    if contexts:
        metric_jobs["retrieval_relevance"] = _score_metric_with_retries(
            metrics["retrieval_relevance"],
            sample,
            attempts,
            retry_delay,
            metric_timeout,
        )
        metric_jobs["faithfulness"] = _score_metric_with_retries(
            metrics["faithfulness"],
            sample,
            attempts,
            retry_delay,
            metric_timeout,
        )

    metric_names = list(metric_jobs)
    raw_scores = await asyncio.gather(
        *(metric_jobs[name] for name in metric_names)
    )
    scores = dict(zip(metric_names, raw_scores))
    raw_correctness = scores["answer_correctness"]
    raw_completeness = scores["answer_completeness"]
    if contexts:
        raw_retrieval = scores["retrieval_relevance"]
        raw_faithfulness = scores["faithfulness"]

    return {
        "case_id": record["case_id"],
        "turn": record["turn"],
        "category": record["category"],
        "expected_should_answer": record["expected_should_answer"],
        "question": record["question"],
        "context_source": record["context_source"],
        "context_count": len(contexts),
        "retrieval_relevance": {
            "score": round(raw_retrieval, 4),
            "raw_score": round(raw_retrieval, 4),
            "evaluator": retrieval_evaluator,
            "status": retrieval_status,
        },
        "answer_correctness": {
            "score": round(normalize_rubric_score(raw_correctness), 4),
            "raw_score": round(raw_correctness, 4),
            "raw_scale": "1-5",
            "evaluator": "ragas.metrics.RubricsScore",
            "status": "scored_without_reference_answer",
        },
        "faithfulness": {
            "score": round(raw_faithfulness, 4),
            "raw_score": round(raw_faithfulness, 4),
            "evaluator": faithfulness_evaluator,
            "status": faithfulness_status,
        },
        "answer_completeness": {
            "score": round(normalize_rubric_score(raw_completeness), 4),
            "raw_score": round(raw_completeness, 4),
            "raw_scale": "1-5",
            "evaluator": "ragas.metrics.RubricsScore",
            "status": "scored_without_reference_answer",
        },
        "response_time": score_response_time(
            record.get("response_time_seconds"),
            record.get("response_time_source", "none"),
        ),
        "behavior_passed": behavior_passed(record),
    }


def evaluate_record(
    record: dict[str, Any],
    metrics: dict[str, Any],
    attempts: int,
    retry_delay: float,
    metric_timeout: float = 180.0,
) -> dict[str, Any]:
    """Synchronous convenience wrapper used by tests and one-off callers."""

    return asyncio.run(
        evaluate_record_async(
            record,
            metrics,
            attempts,
            retry_delay,
            metric_timeout,
        )
    )


async def evaluate_records_async(
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
    attempts: int,
    retry_delay: float,
    metric_timeout: float,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Evaluate multiple turns concurrently without exceeding the limit."""

    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_with_limit(record: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await evaluate_record_async(
                record,
                metrics,
                attempts,
                retry_delay,
                metric_timeout,
            )

    return await asyncio.gather(
        *(evaluate_with_limit(record) for record in records)
    )


def summarize(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    metric_summary: dict[str, Any] = {}

    for metric_name in METRIC_NAMES:
        scores = [
            evaluation[metric_name]["score"]
            for evaluation in evaluations
            if evaluation[metric_name]["score"] is not None
        ]
        metric_summary[metric_name] = {
            "average_score": (
                round(statistics.mean(scores), 4) if scores else None
            ),
            "scale": "0-1",
            "scored_turns": len(scores),
            "unscored_turns": len(evaluations) - len(scores),
        }

    semantic_scores = [
        metric_summary[name]["average_score"]
        for name in METRIC_NAMES[:-1]
        if metric_summary[name]["average_score"] is not None
    ]
    all_available_scores = [
        metric["average_score"]
        for metric in metric_summary.values()
        if metric["average_score"] is not None
    ]

    category_breakdown: dict[str, Any] = {}
    for category in sorted(
        {evaluation["category"] for evaluation in evaluations}
    ):
        category_evaluations = [
            evaluation
            for evaluation in evaluations
            if evaluation["category"] == category
        ]
        category_average_scores: dict[str, float | None] = {}
        for metric_name in METRIC_NAMES:
            category_scores = [
                evaluation[metric_name]["score"]
                for evaluation in category_evaluations
                if evaluation[metric_name]["score"] is not None
            ]
            category_average_scores[metric_name] = (
                round(statistics.mean(category_scores), 4)
                if category_scores
                else None
            )

        category_breakdown[category] = {
            "turns": len(category_evaluations),
            "behavior_pass_rate": round(
                sum(
                    evaluation["behavior_passed"]
                    for evaluation in category_evaluations
                )
                / len(category_evaluations),
                4,
            ),
            "average_scores": category_average_scores,
        }

    return {
        "evaluated_turns": len(evaluations),
        "behavior_passed_turns": sum(
            evaluation["behavior_passed"] for evaluation in evaluations
        ),
        "behavior_failed_turns": sum(
            not evaluation["behavior_passed"] for evaluation in evaluations
        ),
        "metrics": metric_summary,
        "semantic_average_score": (
            round(statistics.mean(semantic_scores), 4)
            if semantic_scores
            else None
        ),
        "overall_average_score": (
            round(statistics.mean(all_available_scores), 4)
            if all_available_scores
            else None
        ),
        "context_sources": {
            source: sum(
                evaluation["context_source"] == source
                for evaluation in evaluations
            )
            for source in sorted(
                {evaluation["context_source"] for evaluation in evaluations}
            )
        },
        "category_breakdown": category_breakdown,
    }


def save_evaluations(
    output_path: Path,
    source_path: Path,
    evaluations: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_evaluations = sorted(
        evaluations,
        key=lambda item: (item["case_id"], item["turn"]),
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(source_path),
        "evaluation_framework": "ragas",
        "ragas_version": "0.3.9",
        "judge_model": settings.chat_model,
        "prompt_version": EVALUATION_PROMPT_VERSION,
        "score_scale": "0-1 (higher is better)",
        "methodology": {
            "retrieval_relevance": (
                "Ragas LLMContextPrecisionWithoutReference"
            ),
            "answer_correctness": (
                "Ragas RubricsScore using a reference-free correctness "
                "rubric; normalized from 1-5 to 0-1"
            ),
            "faithfulness": "Ragas Faithfulness",
            "answer_completeness": (
                "Ragas RubricsScore using a reference-free completeness "
                "rubric; normalized from 1-5 to 0-1"
            ),
            "response_time": (
                "Deterministic latency rubric applied to measured total "
                "seconds (not an LLM metric)"
            ),
        },
        "limitations": [
            (
                "No human-authored reference answers were supplied, so "
                "answer correctness and completeness use Ragas reference-free "
                "rubrics instead of reference-based AnswerCorrectness."
            ),
            (
                "citation_excerpts_proxy means the answer file predates raw "
                "retrieval traces; rerun evaluation.run_questions --fresh "
                "for unbiased retrieved contexts and measured response times."
            ),
        ],
        "summary": summarize(sorted_evaluations),
        "evaluations": sorted_evaluations,
    }

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def load_completed_evaluations(
    output_path: Path,
) -> dict[tuple[int, int], dict[str, Any]]:
    if not output_path.exists():
        return {}

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if payload.get("evaluation_framework") != "ragas":
        return {}

    return {
        (evaluation["case_id"], evaluation["turn"]): evaluation
        for evaluation in payload.get("evaluations", [])
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved RAG answers with Ragas."
    )
    parser.add_argument(
        "answers_file",
        type=Path,
        nargs="?",
        default=Path("json/question_answers.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("json/ragas_evaluation.json"),
    )
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=10.0)
    parser.add_argument(
        "--metric-timeout",
        type=float,
        default=180.0,
        help="Maximum seconds allowed for one Ragas metric attempt.",
    )
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Maximum answer turns evaluated concurrently.",
    )
    parser.add_argument(
        "--checkpoint-size",
        type=int,
        default=6,
        help="Save the JSON report after this many evaluated turns.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N pending turns (useful for a smoke test).",
    )
    parser.add_argument(
        "--fallback-response-time",
        type=float,
        default=None,
        help=(
            "Use one measured average latency for legacy answer files that "
            "do not contain per-turn timing."
        ),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore Ragas evaluations already stored in the output file.",
    )
    args = parser.parse_args()

    answer_payload = json.loads(
        args.answers_file.read_text(encoding="utf-8")
    )
    records = prepare_records(
        answer_payload,
        fallback_response_time=args.fallback_response_time,
    )
    evaluation_by_key = (
        {} if args.fresh else load_completed_evaluations(args.output)
    )
    record_by_key = {
        (record["case_id"], record["turn"]): record for record in records
    }
    for key, evaluation in evaluation_by_key.items():
        record = record_by_key.get(key)
        if record is not None:
            evaluation["response_time"]["source"] = record[
                "response_time_source"
            ]
    pending = [
        record
        for record in records
        if (record["case_id"], record["turn"])
        not in evaluation_by_key
    ]
    if args.limit is not None:
        pending = pending[:args.limit]

    print(
        f"Loaded {len(records)} answer turns; "
        f"{len(pending)} need Ragas evaluation.",
        flush=True,
    )

    if args.concurrency < 1 or args.checkpoint_size < 1:
        parser.error("--concurrency and --checkpoint-size must be positive")

    metrics = get_ragas_metrics() if pending else {}
    for start in range(0, len(pending), args.checkpoint_size):
        batch = pending[start:start + args.checkpoint_size]
        print(
            f"Evaluating pending turns {start + 1}-"
            f"{start + len(batch)} of {len(pending)}...",
            flush=True,
        )
        batch_evaluations = asyncio.run(
            evaluate_records_async(
                batch,
                metrics,
                attempts=args.attempts,
                retry_delay=args.retry_delay,
                metric_timeout=args.metric_timeout,
                concurrency=args.concurrency,
            )
        )
        for evaluation in batch_evaluations:
            key = (evaluation["case_id"], evaluation["turn"])
            evaluation_by_key[key] = evaluation
        save_evaluations(
            args.output,
            args.answers_file,
            list(evaluation_by_key.values()),
        )
        if args.delay:
            time.sleep(args.delay)

    evaluations = list(evaluation_by_key.values())
    save_evaluations(args.output, args.answers_file, evaluations)
    print(json.dumps(summarize(evaluations), indent=2), flush=True)
    print(f"Saved evaluations to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
