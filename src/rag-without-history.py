from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)

vectorstore = Chroma(persist_directory="src/data", embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()


prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


if __name__ == 'main':
    rag_chain.invoke("Como acessar?")
    
    
