from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


loader = PyPDFDirectoryLoader("../files")

docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory="data")

if __name__ == 'main':
    len(docs)
    len(splits)