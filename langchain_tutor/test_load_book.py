from langchain_tutor.config import settings
from langchain_tutor.ingestion.load_book import load_book

pages = load_book(
    settings.book_path,
    source_name=settings.book_title,
)

print("Number of pages:", len(pages))
print("First page:")
print(pages[0])