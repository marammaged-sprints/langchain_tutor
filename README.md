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
python -m pip install -r requirements.txt
```

Copy `langchain_tutor/.env.example` to `langchain_tutor/.env` and add your
Google API key.

## Run the Streamlit app

```powershell
streamlit run streamlit_app.py
```

For Streamlit Community Cloud, set the main file path to `streamlit_app.py`.
The root entry point imports `langchain_tutor.app`, preserving normal package
imports for the application, tests, and command-line modules.

## Run tests

```powershell
python -m pytest -m "not integration"
```
