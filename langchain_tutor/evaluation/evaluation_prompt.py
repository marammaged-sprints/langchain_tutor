"""Ragas rubric prompts used by the reproducible evaluator.

Ragas supplies the metric prompts for context precision and faithfulness. The
two dictionaries below are the project-specific prompts passed to Ragas'
``RubricsScore`` metric for reference-free correctness and completeness.
"""


EVALUATION_PROMPT_VERSION = "ragas-1.0"


CORRECTNESS_RUBRICS = {
    "score1_description": (
        "The response is technically wrong, contradicts the supplied book "
        "context, invents important claims, or gives the wrong behavior for "
        "the stated expected behavior."
    ),
    "score2_description": (
        "The response contains major Python errors or unsupported claims, "
        "although a small part of it is correct."
    ),
    "score3_description": (
        "The response is broadly correct but has a meaningful technical "
        "mistake, ambiguity, or claim not justified by the supplied context."
    ),
    "score4_description": (
        "The response is technically accurate and consistent with the book "
        "context, with only a minor imprecision. A refusal is appropriate only "
        "when the user input explicitly says refusal is expected."
    ),
    "score5_description": (
        "The response is fully technically accurate, consistent with the "
        "supplied book context, and follows the stated expected behavior. If "
        "refusal is expected, it refuses clearly without answering from "
        "outside knowledge."
    ),
}


COMPLETENESS_RUBRICS = {
    "score1_description": (
        "The response does not address the question or omits nearly all of "
        "what was requested."
    ),
    "score2_description": (
        "The response addresses only a small part of the question or lacks "
        "the explanation needed by a beginner."
    ),
    "score3_description": (
        "The response answers the main point but misses a requested part, "
        "comparison, explanation, or useful example."
    ),
    "score4_description": (
        "The response addresses all important parts with enough explanation, "
        "but a minor detail or useful example is absent."
    ),
    "score5_description": (
        "The response fully addresses every part at an appropriate teaching "
        "depth, including an explanation or example when the question calls "
        "for one. When refusal is expected, a concise, clear refusal is "
        "complete."
    ),
}
