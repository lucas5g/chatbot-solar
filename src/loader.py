from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

loader = PyPDFDirectoryLoader("../files")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)


vector_store = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings() )

vector_store.save_local("data")

if __name__ == 'main':
    len(docs)
    len(splits)