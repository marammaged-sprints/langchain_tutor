import pytest
from config import settings
from ingestion.load_book import load_book
from ingestion.split_documents import split_documents


@pytest.mark.integration
def test_split_documents():
    pages = load_book(
        settings.book_path,
        source_name=settings.book_title,
    )

    chunks = split_documents(
        pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    assert len(chunks) > 0
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert all("chunk_id" in chunk.metadata for chunk in chunks)
