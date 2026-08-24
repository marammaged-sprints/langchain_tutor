
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal



QueryType = Literal[
    "conceptual",
    "code_example",
    "definition",
    "out_of_scope",
]

class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    source: str
    page: int | None = None
    score: float | None = None


class BookCitation(BaseModel):
    chunk_id: str
    excerpt: str
    page: int | None = None



class RetrieverResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)


class RAGResponse(BaseModel):

    model_config = ConfigDict(extra="forbid")

    answer: str
    query_type: QueryType
    grounded: bool
    citations: List[BookCitation] = Field(default_factory=list)
    retrieved_chunks: int
    refusal_reason: Optional[str] = None
