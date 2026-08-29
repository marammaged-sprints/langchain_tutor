# RAG evaluation

This folder contains the reproducible evaluation workflow for the Think Python
tutor.

## 1. Generate answers for the 100 cases

Run from the repository root:

```powershell
.\langchain_tutor\venv\Scripts\python.exe `
  -m langchain_tutor.evaluation.run_questions `
  langchain_tutor\evaluation\questions.txt `
  --output langchain_tutor\json\question_answers.json `
  --fresh
```

The runner executes 100 numbered cases and 110 total turns because cases 81-90
contain a first question and a conversational follow-up. It checkpoints the JSON
after every numbered case and stores:

- question, history, and answer;
- grounding decision, refusal reason, verified citations, and the number of
  citations dropped by verification;
- rewritten search query;
- all retrieved chunks and similarity scores;
- chunks that pass the relevance threshold;
- query-rewrite, retrieval, generation, and total response time.

Legacy answer files created before `dropped_citation_count` was added retain a
null value for that field; they cannot reliably distinguish omitted citations
from citations removed during verification. Regenerate with `--fresh` to
capture the distinction.

Without `--fresh`, successful cases already in the output file are preserved and
only missing or failed cases are run again.

## 2. Evaluate the saved answers with Ragas

```powershell
.\langchain_tutor\venv\Scripts\python.exe `
  -m langchain_tutor.evaluation.ragas_evaluate `
  langchain_tutor\json\question_answers.json `
  --output langchain_tutor\json\ragas_evaluation.json `
  --fresh
```

The evaluator checkpoints after every answer turn and stores all scores on a
0-1 scale:

- Retrieval Relevance: Ragas `LLMContextPrecisionWithoutReference`;
- Answer Correctness: Ragas `RubricsScore` with the correctness rubric in
  `evaluation/evaluation_prompt.py`;
- Faithfulness: Ragas `Faithfulness`;
- Answer Completeness: Ragas `RubricsScore` with the completeness rubric in
  `evaluation/evaluation_prompt.py`;
- Response Time: a deterministic score based on measured end-to-end seconds.

Ragas' reference-based `AnswerCorrectness` metric requires human-authored
reference answers. Because this question set does not include those answers,
the evaluator labels correctness and completeness as reference-free rubric
scores instead of pretending they are ground-truth comparisons.

Older answer files do not contain per-turn timings. If a separately measured
average is available, pass it explicitly. The resulting rows are marked
`fallback_average` so they cannot be confused with per-turn measurements:

```powershell
.\langchain_tutor\venv\Scripts\python.exe `
  -m langchain_tutor.evaluation.ragas_evaluate `
  langchain_tutor\json\question_answers.json `
  --fallback-response-time 11.05
```

If a saved answer lacks raw retrieval traces, the evaluator explicitly marks
verified citation excerpts as a proxy. For the cleanest retrieval and latency
evaluation, regenerate `question_answers.json` with `--fresh` first.

The released Ragas package imports a removed LangChain VertexAI path. The Ragas
entry point contains a narrow compatibility shim for that unused provider; it
does not change the tutor or the Gemini judge. Remove the shim when Ragas
publishes the upstream fix.

Using the tutor's own model as judge is convenient but not independent. For a
high-stakes benchmark, add human-authored reference answers, use a separate
judge model, and manually review a representative sample.

## 3. Run the golden behavioral benchmark

The smaller golden set is intended for regression detection after changing the
prompt, embeddings, chunks, or relevance gate:

```powershell
python -m pytest langchain_tutor/tests/test_golden_questions.py -m integration
```

It reports how many supported questions are answered with verified citations
and how many unsupported questions are refused. It requires `GOOGLE_API_KEY`.

Measure the underlying retrieval-score distributions separately with:

```powershell
python -m langchain_tutor.evaluation.calibrate_retrieval
```

That command prints all retrieved distances, evaluates the current gate, and
calculates the threshold with the best balanced accuracy for the configured
`min_top_k`. Treat the recommendation as a measured baseline and rerun it after
changing the embedding model, book, chunking, or golden questions.

### Current golden baseline

With `top_k=5`, `min_top_k=3`, and `retrieval_score_threshold=0.35`:

- retrieval gate: 10/10 in-scope questions accepted;
- retrieval gate: 5/5 out-of-scope questions refused;
- end-to-end answers: 10/10 in-scope questions grounded with citations and an
  expected term;
- end-to-end refusals: 5/5 out-of-scope questions refused;
- highest in-scope gate score: 0.3097;
- lowest out-of-scope gate score: 0.3937;
- measured midpoint recommendation: 0.3517.

The existing `0.35` threshold lies inside the measured separation gap and was
retained to avoid false precision. These measurements depend on the current
book, chunks, embedding model, and query rewriter, so rerun the benchmark when
any of them changes.
