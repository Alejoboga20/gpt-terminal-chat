from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

from reduntant_filter_retriever import RedundantFilterRetriever

from dotenv import load_dotenv

load_dotenv()

# Create a chat model
chat = ChatOpenAI()

# Create an embeddings model
embeddings = OpenAIEmbeddings()

# Create a vector store to reuse the embeddings vector store created in main.py
vector_store = Chroma(
    persist_directory="facts-embeddings/vector_store",
    embedding_function=embeddings
)

retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=vector_store
)

chain = RetrievalQA.from_chain_type(
    llm=chat, retriever=retriever, chain_type="stuff")

result = chain.run("What is an interesting fact about the English language?")
print(result)
