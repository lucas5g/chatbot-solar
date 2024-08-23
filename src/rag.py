from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

import os
from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(
    model="llama-3.1-8b-instant", temperature=0, api_key=os.getenv("GROQ_API_KEY")
)
vectorstore = FAISS.load_local(
    "src/data", embeddings=OpenAIEmbeddings(), allow_dangerous_deserialization=True
)


retriever = vectorstore.as_retriever()


### Contextualize question ###
contextualize_q_system_prompt = (
    "Dado um histórico de bate-papo e a última pergunta do usuário "
    "que pode fazer referência ao contexto no histórico de bate-papo, "
    "formule uma pergunta independente que possa ser entendida "
    "sem o histórico de bate-papo. NÃO responda à pergunta, "
    "apenas reformule-a se necessário e, caso contrário, retorne-a como está."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
system_prompt = (
    "Você é um assistente para tarefas de resposta a perguntas. "
    "Use as seguintes partes do contexto recuperado para responder "
    "à pergunta. Se você não sabe a resposta, diga que "
    "não sabe. Use no máximo três frases e mantenha a "
    "resposta concisa."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
config = {"configurable": {"session_id": "abc2"}}
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

history = get_session_history("abc2")
history.add_ai_message("Aqui é o ChatSolar! Como posso ajudar?")
