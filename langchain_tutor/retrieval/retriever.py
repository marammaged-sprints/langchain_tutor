from __future__ import annotations
from config import settings
from models import RetrievedChunk


def retrieve(vector_store, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    k = max(top_k or settings.top_k, settings.min_top_k)
