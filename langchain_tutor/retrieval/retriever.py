# the retriever doesnt answer the user questions but only answers --> "Which parts of the book are most relevant to this question?"

from __future__ import annotations
from langchain_tutor.config import settings
from langchain_tutor.models import RetrievedChunk

def retrieve(
    vector_store,   #accepts exisiting chroma database
    query: str,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the most relevant book chunks for a user query."""

    if not query.strip():  #no question
        return []

    k = max(top_k or settings.top_k, settings.min_top_k)

    results = vector_store.similarity_search_with_score(
        query,
        k=k,     #Search the Chroma vector database for the chunks most similar to the user's question, and return the results together with their similarity scores.
                 # e.g. Result 1---> content = "A variable is a name that refers to a value...", score = 0.21
    )

    retrieved_chunks: list[RetrievedChunk] = []

    for document, score in results:
        metadata = document.metadata

        retrieved_chunks.append(
            RetrievedChunk(
                chunk_id=metadata.get("chunk_id", "unknown"),
                content=document.page_content,
                source=metadata.get("source", "unknown"),
                page=metadata.get("page"),
                score=float(score),
            )
        )

    return retrieved_chunks
