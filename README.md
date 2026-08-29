# Think Python Tutor

A RAG-based Python tutor built with **LangChain, Gemini, Chroma, Pydantic, and Streamlit**.

The application answers questions about *Think Python* using the book as its knowledge base. It retrieves relevant sections from the book using Gemini embeddings and a Chroma vector store, then uses a Gemini chat model to generate grounded answers.

## Features

- Question answering grounded in *Think Python*
- PDF document ingestion
- Configurable document chunking
- Gemini embeddings for semantic search
- Persistent Chroma vector store
- Top-k relevant chunk retrieval
- Query rewriting for improved retrieval
- Gemini chat model for answer generation
- Pydantic models for structured and validated data
- Streamlit chat interface
- Book citations and retrieved context
- Persistent vector index so the database does not need to be rebuilt every time the application starts

## Project Architecture

The application follows a Retrieval-Augmented Generation (RAG) pipeline:

```text
User
 â”‚
 â–¼
Streamlit UI
 â”‚
 â–¼
RAG Chain
 â”‚
 â”œâ”€â”€ Query Rewriter
 â”‚
 â–¼
Retriever
 â”‚
 â–¼
Chroma Vector Store
 â”‚
 â–¼
Relevant Think Python Chunks
 â”‚
 â–¼
Gemini Chat Model
 â”‚
 â–¼
Grounded Answer
```

## Setup

Run commands from the repository root so Python can import `langchain_tutor` as
a package:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with
`source .venv/bin/activate` instead.

Copy the example environment file:

```powershell
Copy-Item langchain_tutor/.env.example langchain_tutor/.env
```

Then add your Google Gemini API key to `langchain_tutor/.env`:

```ini
GOOGLE_API_KEY=your-key-here
```

You can create a key in [Google AI Studio](https://aistudio.google.com/apikey).

## Build the index

Build the vector index once before starting the app:

```powershell
python -m langchain_tutor.ingestion.build_index
```

This reads `langchain_tutor/data/think_python.pdf`, splits and embeds its text,
and writes the Chroma store to `langchain_tutor/data/chroma_db/`. It can take a
few minutes and prints progress for each batch. Re-running the command is safe:
a complete index is reused, while an incomplete index is rebuilt.

Building it explicitly prevents the first question in a fresh deployment from
having to wait while the index is created.

## Run the Streamlit app

```powershell
streamlit run streamlit_app.py
```

For Streamlit Community Cloud, set the main file path to `streamlit_app.py`.
The root entry point imports `langchain_tutor.app`, preserving normal package
imports for the application, tests, and command-line modules.

## Run tests

Run the fast suite without external API calls:

```powershell
python -m pytest langchain_tutor/tests -m "not integration"
```

Run the complete suite after configuring `GOOGLE_API_KEY`:

```powershell
python -m pytest langchain_tutor/tests
```

## Golden questions and retrieval calibration

The integration suite includes 10 representative questions the book should
answer and 5 unsupported questions it should refuse. Run only this behavioral
benchmark with:

```powershell
python -m pytest langchain_tutor/tests/test_golden_questions.py -m integration
```

To inspect the retrieved distance scores and calculate a measured candidate for
`retrieval_score_threshold`, run:

```powershell
python -m langchain_tutor.evaluation.calibrate_retrieval
```

The command reports the current gate's in-scope acceptance and out-of-scope
refusal counts, then recommends the threshold with the best balanced accuracy
for the current `min_top_k`. Lower Chroma distance scores are more relevant.

## Runtime retrieval diagnostics

Each Streamlit question writes one INFO log record containing the original and
rewritten queries, retrieved and relevant chunk counts, distance scores, gate
settings, verified citation count, and final outcome. Outcomes distinguish
retrieval refusals, model refusals, generation failures, unverified answers,
and grounded answers. Streamlit displays these records in its server logs; no
book context or generated answer text is included.

## Inspect document chunks

The chunk inspection utility is kept outside the application package. Run it
as a module from the repository root:

```powershell
python -m scripts.check_chunks
```
