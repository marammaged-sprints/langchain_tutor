import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_tutor.config import settings


logger = logging.getLogger(__name__)


_rewriter_model = None


def get_rewriter_model():
    global _rewriter_model

    if _rewriter_model is None:
        _rewriter_model = ChatGoogleGenerativeAI(
            model=settings.chat_model,
            google_api_key=settings.require_api_key(),
            temperature=0.0,
        )

    return _rewriter_model

rewriter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Rewrite the user's question into a concise search query
for retrieving relevant information from the Think Python book.

Use the conversation history only to resolve references such as
"it", "this", or "that".

Return only the rewritten search query.
Do not answer the question.
Do not explain your changes.""",
        ),
        (
            "human",
            """Conversation history:
{history}

User question:
{question}""",
        ),
    ]
)


def _extract_text(response) -> str:
    """Extract text from the LLM response."""
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])

        return "".join(parts)

    return str(content)


def rewrite_query(question: str, history: str = "") -> str:
    """Rewrite a question for retrieval, with a safe fallback."""

    try:
        rewriter_chain = rewriter_prompt | get_rewriter_model()
        response = rewriter_chain.invoke(
            {
                "history": history,
                "question": question,
            }
        )

        rewritten = _extract_text(response).strip()

    except Exception:
        logger.warning(
            "Query rewrite failed; falling back to the raw question.",
            exc_info=True,
        )
        return question

    # Empty output means retrieval should use the original question.
    if not rewritten:
        logger.info(
            "Query rewrite returned empty text; using the original question."
        )
        return question

    # A rewrite should be a short search query, not a paragraph of prose.
    if len(rewritten) > 300 or "\n" in rewritten:
        logger.info(
            "Rewrite looks like prose, not a query; using the original question."
        )
        return question

    return rewritten
