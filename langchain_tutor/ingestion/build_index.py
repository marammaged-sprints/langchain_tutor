from __future__ import annotations

import shutil
import time
from pathlib import Path

from langchain_tutor.config import settings
from langchain_tutor.ingestion.load_book import load_book
from langchain_tutor.ingestion.split_documents import split_documents
from langchain_tutor.retrieval.vector_store import get_vector_store


def index_exists(persist_directory: Path) -> bool:
    """Check whether a Chroma index exists in the specified directory."""
    persist_directory = Path(persist_directory)

    if not persist_directory.exists():
        return False

    return any(persist_directory.iterdir())


def build_or_load_index(force_rebuild: bool = False):
    """Load an existing Chroma index or build a new one."""

    persist_dir = Path(settings.persist_directory)

    # Delete the existing database when a rebuild is requested.
    if force_rebuild and persist_dir.exists():
        shutil.rmtree(persist_dir)

    pages = load_book(
        settings.book_path,
        source_name=settings.book_title,
    )

    # Make sure the PDF actually contains extractable text.
    if not any(page.page_content.strip() for page in pages):
        raise RuntimeError(
            f"'{settings.book_path}' produced no text on any of its "
            f"{len(pages)} pages."
        )

    # Split the PDF into chunks.
    chunks = split_documents(
        pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # Make sure splitting produced something to index.
    if not chunks:
        raise RuntimeError(
            f"No text extracted from '{settings.book_path}'. "
            "The PDF may be scanned or image-only — "
            "check that PyPDFLoader can read it."
        )

    expected_chunk_count = len(chunks)

    # Load the existing index if it is complete.
    if not force_rebuild and index_exists(persist_dir):
        vs = get_vector_store(persist_directory=persist_dir)

        try:
            chunk_count = vs._collection.count()
        except Exception:
            chunk_count = None

        if chunk_count == expected_chunk_count:
            print("Existing complete index found.")
            return vs, chunk_count

        print(
            f"Incomplete index found: "
            f"{chunk_count} / {expected_chunk_count} chunks."
        )

        shutil.rmtree(persist_dir)

    # Create the directory for the new Chroma database.
    persist_dir.mkdir(parents=True, exist_ok=True)

    vs = get_vector_store(persist_directory=persist_dir)

    print("Building Chroma index...")

    batch_size = 100

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        batch_ids = [
            chunk.metadata["chunk_id"]
            for chunk in batch
        ]

        for attempt in range(3):
            try:
                vs.add_documents(
                    batch,
                    ids=batch_ids,
                )
                break

            except Exception:
                if attempt == 2:
                    raise

                wait_time = 2 ** attempt

                print(
                    f"  Batch failed. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

        indexed_so_far = min(
            start + batch_size,
            len(chunks),
        )

        print(
            f"  Indexed {indexed_so_far}/{len(chunks)} chunks"
        )

    indexed_count = vs._collection.count()

    # Verify that every expected chunk was indexed.
    if indexed_count != expected_chunk_count:
        raise RuntimeError(
            f"Indexing incomplete: expected "
            f"{expected_chunk_count} chunks, "
            f"but Chroma contains {indexed_count}."
        )

    print(f"Indexed chunks: {indexed_count}")

    return vs, indexed_count


if __name__ == "__main__":
    build_or_load_index()