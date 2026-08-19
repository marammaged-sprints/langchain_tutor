from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from config import settings


rewriter_model = ChatGoogleGenerativeAI(
    model=settings.chat_model,
    google_api_key=settings.google_api_key,
)
REWRITER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You rewrite user questions into concise search queries for retrieving
relevant passages from the book "Think Python".

Use the conversation history only to resolve references such as:
- it
- this
- that
- them
- the previous topic

Return only the rewritten search query.

Do not answer the question.
Do not add information that is not present in the conversation.
""",
        ),
        (
            "human",
            """
Conversation history:
{history}

Current user question:
{question}

Rewrite the current question into a clear search query for Think Python.
""",
        ),
    ]
)
rewriter_chain = REWRITER_PROMPT | rewriter_model  #pipe the output of one component into the next.

def rewrite_query(question: str, history: str = "") -> str:  #Takes the user's question and conversation history, then return an improved search query.
    response = rewriter_chain.invoke(
        {
            "history": history,
            "question": question,
        }
    )

    content = response.content #save content

    if isinstance(content, list):  #Is Gemini's content a list?
        return "".join(
            block.get("text", "")  #If yes, we extract the "text" from each block
            for block in content
            if isinstance(block, dict)
        ).strip()
    return content.strip()
