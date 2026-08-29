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
- grounding decision, refusal reason, and verified citations;
- rewritten search query;
- all retrieved chunks and similarity scores;
- chunks that pass the relevance threshold;
- query-rewrite, retrieval, generation, and total response time.

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
