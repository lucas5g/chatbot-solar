import streamlit as st
from rag import conversational_rag_chain, config, history
from langchain.schema import HumanMessage

st.set_page_config(page_title='Chat Solar')
st.markdown(
    """
    # Chat Solar
    Tire todas as dúvidas sobre a nova plataforma.
    """
)


for message in history.messages:

    if isinstance(message, HumanMessage):
        st.chat_message("user").markdown(message.content)
    else:
        st.chat_message("assistant").markdown(message.content)


question = st.chat_input("Tire dúvidas sobre o solar")

if question:
    st.chat_message("human").markdown(question)

    res = conversational_rag_chain.invoke({"input": question}, config=config)["answer"]

    st.chat_message("assistant").markdown(res)
