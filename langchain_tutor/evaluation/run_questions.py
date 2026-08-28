from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import settings
from rag_chain import run_rag


SECTION_NAMES = {
    "A": "direct_retrieval",
    "B": "multi_chunk_synthesis",
    "C": "conversational_follow_up",
    "D": "unsupported_guardrail",
}

REFUSAL_PATTERN = re.compile(
    r"^\s*(?:I cannot answer|I can't answer|I am unable to answer)",
    re.IGNORECASE,
)


def looks_like_refusal(answer: str | None) -> bool:
    return bool(answer and REFUSAL_PATTERN.search(answer))


def parse_question_cases(text: str) -> list[dict[str, Any]]:
    """Parse the supplied evaluation brief into 100 numbered test cases."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cases: list[dict[str, Any]] = []
    section: str | None = None
    index = 0

    while index < len(lines):
        line = lines[index]

        section_match = re.match(r"^([A-D])\.\s", line)
        if section_match:
            section = section_match.group(1)
            index += 1
            continue

        direct_match = re.match(r"^(\d+)\.\s+(.+)$", line)
        if direct_match:
            if section is None:
                raise ValueError(f"Question appears before a section: {line}")

            question_id = int(direct_match.group(1))
            cases.append(
                {
                    "id": question_id,
                    "category": SECTION_NAMES[section],
                    "questions": [direct_match.group(2)],
                    "expected_should_answer": section != "D",
                }
            )
            index += 1
            continue

        conversation_match = re.fullmatch(r"(8[1-9]|90)", line)
        if conversation_match:
            if section != "C":
                raise ValueError(
                    f"Conversational case found outside section C: {line}"
                )

            if index + 4 >= len(lines):
                raise ValueError(f"Incomplete conversational case {line}")

            first_label, first_question, then_label, follow_up = lines[
                index + 1:index + 5
            ]
            if first_label != "First:" or then_label != "Then:":
                raise ValueError(
                    f"Unexpected conversational format for case {line}"
                )

            cases.append(
                {
                    "id": int(line),
                    "category": SECTION_NAMES[section],
                    "questions": [first_question, follow_up],
                    "expected_should_answer": True,
                }
            )
            index += 5
            continue

        # Explanatory prose between a section heading and its questions.
        index += 1

    actual_ids = [case["id"] for case in cases]
    expected_ids = list(range(1, 101))
    if actual_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(actual_ids))
        duplicates = sorted(
            question_id
            for question_id in set(actual_ids)
            if actual_ids.count(question_id) > 1
        )
        raise ValueError(
            "Expected question IDs 1-100 in order; "
            f"missing={missing}, duplicates={duplicates}"
        )

    return cases


def call_rag_with_retries(
    question: str,
    history: str,
    attempts: int,
    retry_delay: float,
):
    for attempt in range(1, attempts + 1):
        trace: dict[str, Any] = {}
        try:
            response = run_rag(
                question=question,
                history=history,
                trace=trace,
            )
            return response, trace
        except Exception as exc:
            if attempt == attempts:
                raise

            delay = retry_delay * attempt
            print(
                f"    attempt {attempt}/{attempts} failed "
                f"({type(exc).__name__}); retrying in {delay:g}s",
                flush=True,
            )
            time.sleep(delay)


def evaluate_case(
    case: dict[str, Any],
    attempts: int,
    retry_delay: float,
) -> dict[str, Any]:
    history_lines: list[str] = []
    turns: list[dict[str, Any]] = []

    for turn_number, question in enumerate(case["questions"], start=1):
        history = "\n".join(history_lines)

        try:
            response, trace = call_rag_with_retries(
                question=question,
                history=history,
                attempts=attempts,
                retry_delay=retry_delay,
            )
        except Exception as exc:
            turns.append(
                {
                    "turn": turn_number,
                    "question": question,
                    "history": history,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )
            break

        response_data = response.model_dump(mode="json")
        actual_should_answer = (
            response.grounded
            and response.query_type != "out_of_scope"
            and not response.refusal_reason
            and not looks_like_refusal(response.answer)
        )
        turns.append(
            {
                "turn": turn_number,
                "question": question,
                "history": history,
                "actual_should_answer": actual_should_answer,
                "trace": trace,
                **response_data,
            }
        )

        # Match app.py: an ungrounded response is not added as an assistant
        # message, but the user's preceding question remains in the history.
        history_lines.append(f"User: {question}")
        if response.grounded:
            history_lines.append(f"Assistant: {response.answer}")

    expected_should_answer = case["expected_should_answer"]
    completed = len(turns) == len(case["questions"])
    no_errors = completed and all("error" not in turn for turn in turns)

    if expected_should_answer:
        passed = no_errors and all(
            turn.get("actual_should_answer") is True
            and bool(turn.get("citations"))
            for turn in turns
        )
    else:
        passed = no_errors and all(
            turn.get("actual_should_answer") is False
            and turn.get("grounded") is False
            and turn.get("query_type") == "out_of_scope"
            and turn.get("citations") == []
            for turn in turns
        )

    return {
        "id": case["id"],
        "category": case["category"],
        "expected_should_answer": expected_should_answer,
        "passed": passed,
        "turns": turns,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "completed_cases": len(results),
        "passed_cases": sum(result["passed"] for result in results),
        "failed_cases": sum(not result["passed"] for result in results),
        "errored_cases": sum(
            any("error" in turn for turn in result["turns"])
            for result in results
        ),
        "total_answer_turns": sum(
            len(result["turns"]) for result in results
        ),
    }


def save_results(
    output_path: Path,
    source_path: Path,
    results: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(source_path),
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "total_numbered_cases": 100,
        "summary": summarize(results),
        "results": sorted(results, key=lambda result: result["id"]),
    }

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def load_completed_results(output_path: Path) -> dict[int, dict[str, Any]]:
    if not output_path.exists():
        return {}

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    return {
        result["id"]: result
        for result in payload.get("results", [])
        if result.get("passed") is True
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Think Python RAG evaluation question set."
    )
    parser.add_argument("questions_file", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("json/question_answers.json"),
    )
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=10.0)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore successful cases already stored in the output file.",
    )
    args = parser.parse_args()

    cases = parse_question_cases(
        args.questions_file.read_text(encoding="utf-8")
    )
    result_by_id = (
        {} if args.fresh else load_completed_results(args.output)
    )

    print(
        f"Parsed {len(cases)} cases / "
        f"{sum(len(case['questions']) for case in cases)} turns.",
        flush=True,
    )
    if result_by_id:
        print(
            f"Resuming with {len(result_by_id)} successful cases cached.",
            flush=True,
        )

    for position, case in enumerate(cases, start=1):
        if case["id"] in result_by_id:
            continue

        print(
            f"[{position:03d}/100] Case {case['id']}: "
            f"{case['category']}",
            flush=True,
        )
        result = evaluate_case(
            case,
            attempts=args.attempts,
            retry_delay=args.retry_delay,
        )
        result_by_id[case["id"]] = result
        save_results(
            args.output,
            args.questions_file,
            list(result_by_id.values()),
        )
        print(
            f"    {'PASS' if result['passed'] else 'FAIL'} "
            f"({len(result['turns'])} turn(s))",
            flush=True,
        )
        time.sleep(args.delay)

    results = list(result_by_id.values())
    save_results(args.output, args.questions_file, results)
    print(json.dumps(summarize(results), indent=2), flush=True)
    print(f"Saved results to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
