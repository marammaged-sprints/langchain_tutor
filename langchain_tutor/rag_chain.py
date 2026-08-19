from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


from models import RAGResponse
from config import settings
from prompts import SYSTEM_PROMPT, HUMAN_PROMPT
from retrieval.retriever import retrieve  #relevant chunks
from retrieval.vector_store import get_vector_store  #access chroma
from retrieval.query_rewriter import rewrite_query



chat_model = ChatGoogleGenerativeAI(  #creates the Gemini model that will eventually generate the answer.
    model=settings.chat_model,
    google_api_key=settings.google_api_key,
)
structured_chat_model = chat_model.with_structured_output(RAGResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ]
)


chain = prompt | structured_chat_model  #Connect prompt + structured Gemini

vector_store = get_vector_store()   #connection to Chroma

def retrieve_context(query: str):
    return retrieve(
        vector_store=vector_store,
        query=query,
        top_k=settings.top_k,
    )
def retrieve_for_question(question: str, history: str = ""):
    search_query = rewrite_query(
        question=question,
        history=history,
    )

    return retrieve_context(search_query)

def format_context(chunks):   #Take the retrieved chunks and turn them into a clean block of context for the LLM.
    formatted_chunks = []

    for chunk in chunks:
        formatted_chunks.append(
            f"[chunk_id: {chunk.chunk_id} | page: {chunk.page}]\n"
            f"{chunk.content}"
        )

    return "\n\n---\n\n".join(formatted_chunks)

def has_relevant_context(chunks) -> bool:
    if len(chunks) < settings.min_top_k:
        return False

    best_score = min(chunk.score for chunk in chunks)  #lower distance means better similarity.

    return best_score <= settings.retrieval_score_threshold

def run_rag(question: str, history: str = "") -> RAGResponse:
    chunks = retrieve_for_question(
        question=question,
        history=history,
    )
    if not has_relevant_context(chunks):   # to make sure gemini doesnt answer from his knowledge
        return RAGResponse(
            answer="I can't answer that from the book.",
            query_type="out_of_scope",
            grounded=False,
            citations=[],
            retrieved_chunks=len(chunks),
            refusal_reason="Not enough relevant book context was retrieved.",
        )


    context = format_context(chunks)

    response = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    return response

