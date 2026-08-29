import subprocess
import sys

from langchain_tutor.evaluation.ragas_evaluate import (
    behavior_passed,
    evaluate_record,
    normalize_rubric_score,
    prepare_records,
    score_response_time,
    summarize,
)


def test_evaluation_module_imports_when_ragas_is_unavailable():
    script = """
import importlib.abc
import sys

class BlockRagas(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "ragas" or fullname.startswith("ragas."):
            raise ModuleNotFoundError(
                "blocked optional dependency",
                name=fullname,
            )
        return None

sys.meta_path.insert(0, BlockRagas())
from langchain_tutor.evaluation.ragas_evaluate import (
    get_ragas_metrics,
    normalize_rubric_score,
)
assert normalize_rubric_score(5) == 1.0
try:
    get_ragas_metrics()
except RuntimeError as exc:
    assert "optional 'ragas' package" in str(exc)
else:
    raise AssertionError("missing Ragas should produce a clear runtime error")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


class FakeMetric:
    def __init__(self, score):
        self.score = score

    async def single_turn_ascore(self, sample, timeout=None):
        assert sample.user_input
        assert sample.response
        assert timeout
        return self.score


def test_prepares_retrieval_evidence_and_timing():
    payload = {
        "results": [
            {
                "id": 1,
                "category": "direct_retrieval",
                "expected_should_answer": True,
                "turns": [
                    {
                        "turn": 1,
                        "question": "What is a variable?",
                        "answer": "A name that refers to a value.",
                        "query_type": "definition",
                        "grounded": True,
                        "citations": [],
                        "dropped_citation_count": 2,
                        "trace": {
                            "search_query": "Python variable definition",
                            "retrieved_chunks": [{"chunk_id": "c1"}],
                            "relevant_chunks": [{"chunk_id": "c1"}],
                            "timings_seconds": {"total": 2.5},
                        },
                    }
                ],
            }
        ]
    }

    records = prepare_records(payload)

    assert records[0]["search_query"] == "Python variable definition"
    assert records[0]["retrieved_contexts"] == []
    assert records[0]["context_source"] == "none"
    assert records[0]["response_time_seconds"] == 2.5
    assert records[0]["response_time_source"] == (
        "trace.timings_seconds.total"
    )
    assert records[0]["dropped_citation_count"] == 2
    assert records[0]["answer_displayed"] is True


def test_response_time_scoring_boundaries():
    assert score_response_time(3, "fallback_average")["score"] == 1.0
    assert score_response_time(3, "fallback_average")["source"] == (
        "fallback_average"
    )
    assert score_response_time(6)["score"] == 0.75
    assert score_response_time(10)["score"] == 0.5
    assert score_response_time(15)["score"] == 0.25
    assert score_response_time(15.1)["score"] == 0.0
    assert score_response_time(None)["score"] is None


def test_summarizes_five_factor_scores():
    metric = {"score": 0.75}
    evaluation = {
        "case_id": 1,
        "turn": 1,
        "category": "direct_retrieval",
        "context_source": "trace.relevant_chunks",
        "retrieval_relevance": metric,
        "answer_correctness": metric,
        "faithfulness": metric,
        "answer_completeness": metric,
        "response_time": metric,
        "behavior_passed": True,
    }

    summary = summarize([evaluation])

    assert summary["behavior_passed_turns"] == 1
    assert summary["metrics"]["answer_correctness"]["average_score"] == 0.75
    assert summary["overall_average_score"] == 0.75


def test_ragas_metrics_are_used_and_rubric_scores_are_normalized():
    record = {
        "case_id": 1,
        "turn": 1,
        "category": "direct_retrieval",
        "expected_should_answer": True,
        "question": "What is a variable?",
        "history": "",
        "answer": "A variable is a name that refers to a value.",
        "grounded": True,
        "citations": [{"excerpt": "A variable is a name."}],
        "retrieved_contexts": ["A variable is a name that refers to a value."],
        "context_source": "trace.relevant_chunks",
        "response_time_seconds": 4.0,
        "response_time_source": "trace.timings_seconds.total",
    }
    metrics = {
        "retrieval_relevance": FakeMetric(0.8),
        "answer_correctness": FakeMetric(5),
        "faithfulness": FakeMetric(0.9),
        "answer_completeness": FakeMetric(4),
    }

    result = evaluate_record(record, metrics, attempts=1, retry_delay=0)

    assert result["retrieval_relevance"]["score"] == 0.8
    assert result["answer_correctness"]["score"] == 1.0
    assert result["faithfulness"]["score"] == 0.9
    assert result["answer_completeness"]["score"] == 0.75
    assert result["response_time"]["score"] == 0.75
    assert result["dropped_citation_count"] is None
    assert result["answer_displayed"] is False
    assert result["behavior_passed"] is True


def test_behavior_contract_rejects_hidden_refusal_for_supported_question():
    record = {
        "expected_should_answer": True,
        "answer": "I cannot answer that from the supplied context.",
        "grounded": True,
        "citations": [{"excerpt": "some passage"}],
    }

    assert behavior_passed(record) is False
    assert normalize_rubric_score(1) == 0.0
    assert normalize_rubric_score(5) == 1.0
