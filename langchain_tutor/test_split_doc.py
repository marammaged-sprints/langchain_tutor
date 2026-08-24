from config import settings
from ingestion.load_book import load_book
from ingestion.split_documents import split_documents


pages = load_book(
    settings.book_path,
    source_name=settings.book_title,
)

print("Number of pages:", len(pages))

chunks = split_documents(
    pages,
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)

print("Number of chunks:", len(chunks))
print("\nFirst chunk:")
print(chunks[0])

print("\nFirst chunk metadata:")
print(chunks[0].metadata)