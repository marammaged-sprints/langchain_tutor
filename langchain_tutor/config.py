from pathlib import Path
import os

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


class Settings:
    book_path: Path = BASE_DIR / "data" / "think_python.pdf"
    book_title: str = "Think Python"

    chunk_size: int = 1000
    chunk_overlap: int = 150

    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    embedding_model: str = "gemini-embedding-001"

    persist_directory: Path = BASE_DIR / "data" / "chroma_db"
    collection_name: str = "think_python"
    top_k: int = 5
    min_top_k: int = 3


settings = Settings()