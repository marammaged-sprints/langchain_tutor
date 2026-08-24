import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from models import RAGResponse, RetrievedChunk
from config import settings
from langchain_tutor.prompts import SYSTEM_PROMPT, HUMAN_PROMPT
from langchain_tutor.retrieval.retriever import retrieve
from langchain_tutor.retrieval.vector_store import get_vector_store
from langchain_tutor.retrieval.query_rewriter import rewrite_query


logger = logging.getLogger(__name__)


_chat_model = None
_vector_store = None


def get_chat_model():
    global _chat_model

    if _chat_model is None:
        _chat_model = ChatGoogleGenerativeAI(
            model=settings.chat_model,
            google_api_key=settings.require_api_key(),
            temperature=0.0,
        ).with_structured_output(RAGResponse)

    return _chat_model


def get_store():
    global _vector_store

    if _vector_store is None:
        _vector_store = get_vector_store()

    return _vector_store


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ]
)


def retrieve_context(query: str):
    return retrieve(
        vector_store=get_store(),
        query=query,
        top_k=settings.top_k,
    )


def retrieve_for_question(question: str, history: str = ""):
    search_query = rewrite_query(
        question=question,
        history=history,
    )

    return retrieve_context(search_query)


def format_context(chunks):
    formatted_chunks = []

    for chunk in chunks:
        formatted_chunks.append(
            f"[chunk_id: {chunk.chunk_id} | page: {chunk.page}]\n"
            f"{chunk.content}"
        )

    return "\n\n---\n\n".join(formatted_chunks)


def select_relevant(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return [
        chunk
        for chunk in chunks
        if chunk.score is not None
        and chunk.score <= settings.retrieval_score_threshold
    ]


def verify_citations(
    response: RAGResponse,
    chunks: list[RetrievedChunk],
) -> RAGResponse:
    """
    Verify that citations produced by the LLM actually refer
    to chunks retrieved by our retriever.
    """

    by_id = {chunk.chunk_id: chunk for chunk in chunks}

    verified_citations = []
    dropped_citations = []

    for citation in response.citations:
        chunk = by_id.get(citation.chunk_id)

        if chunk is None:
            dropped_citations.append(citation.chunk_id)
            continue

        citation.page = chunk.page

        excerpt = citation.excerpt.strip()

        if excerpt and excerpt not in chunk.content:
            logger.warning(
                "Invalid excerpt for citation chunk_id=%s",
                citation.chunk_id,
            )

            citation.excerpt = chunk.content[:200]

        verified_citations.append(citation)

    response.citations = verified_citations

    response.retrieved_chunks = len(chunks)

    response.grounded = (
        bool(verified_citations)
        and not dropped_citations
    )

    if dropped_citations:
        logger.warning(
            "Dropped %d fabricated citation(s): %s",
            len(dropped_citations),
            dropped_citations,
        )

    return response


def run_rag(question: str, history: str = "") -> RAGResponse:
    if not question or not question.strip():
        raise ValueError("run_rag requires a non-empty question")

    chunks = retrieve_for_question(
        question=question,
        history=history,
    )

    relevant_chunks = select_relevant(chunks)

    if len(relevant_chunks) < settings.min_top_k:
        return RAGResponse(
            answer="I can't answer that from the book.",
            query_type="out_of_scope",
            grounded=False,
            citations=[],
            retrieved_chunks=len(relevant_chunks),
            refusal_reason="Not enough relevant book context was retrieved.",
        )

    context = format_context(relevant_chunks)

    chain = prompt | get_chat_model()

    response = chain.invoke({
        "question": question,
        "context": context,
        "history": history or "(no previous turns)",
    })

    return verify_citations(response, relevant_chunks)