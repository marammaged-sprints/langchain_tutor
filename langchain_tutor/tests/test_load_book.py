import pytest
from langchain_tutor.config import settings
from langchain_tutor.ingestion.load_book import load_book


@pytest.mark.integration
def test_load_book():
    pages = load_book(
        settings.book_path,
        source_name=settings.book_title,
    )

    assert len(pages) > 0
    assert pages[0].page_content.strip()
