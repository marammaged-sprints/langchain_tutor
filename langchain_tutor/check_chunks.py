from ingestion.load_book import load_book
from ingestion.split_documents import split_documents
from config import settings


pages = load_book(
    settings.book_path,
    source_name=settings.book_title,
)

chunks = split_documents(
    pages,
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)

for i, chunk in enumerate(chunks):
    if "def " in chunk.page_content and not chunk.page_content.rstrip().endswith(":"):
        print(f"CHUNK {i}: {chunk.metadata.get('chunk_id')}")
        print(f"PAGE: {chunk.metadata.get('page')}")
        print(chunk.page_content[-300:])
        print("=" * 80)
