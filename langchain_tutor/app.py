import streamlit as st
from rag_chain import run_rag


st.set_page_config ( page_title= "Think python Tutor", page_icon= "📘")
st.title ("📘 Think Python Tutor")
st.caption ("Ask questions about Think Python and get answers grounded in the book.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):  #becomes "user"
        st.markdown(message["content"])  #becomes "what is a variable?"

question = st.chat_input("Ask a question about Think Python...")


if question:
    # Display the user's question immediately.
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

history = "\n".join(
    f"{message['role'].title()}:{message['content']}"
    for message in st.session_state["messages"][:-1]
)

with st.chat_message("assistant"):
    with st.spinner("Searching the book..."):
        response= run_rag(      
            question=question,
            history=history,
            )
    st.markdown(response.answer)

    if response.citations:
            st.markdown("### 📚 Sources")

            for citation in response.citations:
                st.markdown(
                    f"**Page {citation.page}** — "
                    f"{citation.excerpt}"
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.answer,
        }
    )

        