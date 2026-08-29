"""Inspect document chunks that may split a Python function definition."""

from __future__ import annotations

from collections.abc import Iterable

from langchain_core.documents import Document

from langchain_tutor.config import settings
from langchain_tutor.ingestion.load_book import load_book
from langchain_tutor.ingestion.split_documents import split_documents


def find_suspicious_chunks(
    chunks: Iterable[Document],
) -> list[tuple[int, Document]]:
    """Return chunks whose function definition may continue in another chunk."""
    return [
        (index, chunk)
        for index, chunk in enumerate(chunks)
        if "def " in chunk.page_content
        and not chunk.page_content.rstrip().endswith(":")
    ]


def main() -> None:
    pages = load_book(
        settings.book_path,
        source_name=settings.book_title,
    )
    chunks = split_documents(
        pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    suspicious_chunks = find_suspicious_chunks(chunks)
    if not suspicious_chunks:
        print("No suspicious chunks found.")
        return

    for index, chunk in suspicious_chunks:
        print(f"CHUNK {index}: {chunk.metadata.get('chunk_id')}")
        print(f"PAGE: {chunk.metadata.get('page')}")
        print(chunk.page_content[-300:])
        print("=" * 80)


if __name__ == "__main__":
    main()
