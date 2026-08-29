import streamlit as st

from langchain_tutor.models import RAGResponse
from langchain_tutor.rag_chain import run_rag


def render_refusal(response: RAGResponse) -> bool:
    """Render a refusal state and report whether the response was handled."""
    if response.refusal_reason == "Generation failed.":
        st.error(response.answer)
        return True

    if response.query_type == "out_of_scope":
        st.markdown(response.answer)
        return True

    if not response.grounded:
        st.warning(
            "I found relevant passages, but I couldn't verify the answer "
            "against them. Try rephrasing the question."
        )
        return True

    return False


def main() -> None:
    st.set_page_config(
        page_title="Think Python Tutor",
        page_icon="📘",
    )

    st.title("📘 Think Python Tutor")
    st.caption(
        "Ask questions about Think Python and get answers grounded in the book."
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about Think Python...")

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        history = "\n".join(
            f"{message['role'].title()}: {message['content']}"
            for message in st.session_state["messages"][:-1][-6:]
        )

        with st.chat_message("assistant"):
            with st.spinner("Searching the book..."):
                response = run_rag(
                    question=question,
                    history=history,
                )

            if not render_refusal(response):
                st.markdown(response.answer)

                if response.citations:
                    st.markdown("### 📚 Sources")

                    for citation in response.citations:
                        st.markdown(
                            f"**Page {citation.page}** — {citation.excerpt}"
                        )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                    }
                )


if __name__ == "__main__":
    main()
