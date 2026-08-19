
from pydantic import BaseModel, Field
from typing import List, Optional


class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    source: str
    page: int | None = None
    score: float | None = None


class BookCitation(BaseModel):
    page: int
    excerpt: str
    chunk_id: str



class RetrieverResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)


class RAGResponse(BaseModel):
    answer: str
    query_type: str
    grounded: bool
    citations: List[BookCitation] = Field(default_factory=list)
    retrieved_chunks: int
    refusal_reason: Optional[str] = None
