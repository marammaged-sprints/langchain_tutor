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
    def require_api_key(self) -> str:
        if not self.google_api_key:
            raise RuntimeError(
            "GOOGLE_API_KEY is not set — "
            "copy .env.example to .env and fill it in."
        )
        return self.google_api_key
    
    embedding_model: str = "gemini-embedding-001"
    chat_model: str = "gemini-3.6-flash"

    persist_directory: Path = BASE_DIR / "data" / "chroma_db"
    collection_name: str = "think_python"
    top_k: int = 5
    min_top_k: int = 3
    retrieval_score_threshold: float = 0.35


settings = Settings()