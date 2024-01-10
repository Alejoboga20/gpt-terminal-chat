from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

# Create an embeddings model
embeddings = OpenAIEmbeddings()

# Create a text splitter to create chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)

# Load the file
loader = TextLoader("facts-embeddings/facts.txt")
# Load and split the file
docs = loader.load_and_split(text_splitter=text_splitter)

# Create a vector store
vector_store = Chroma.from_documents(
    documents=docs,
    embeddings=embeddings,
    persist_directory="facts-embeddings/vector_store"
)

for doc in docs:
    print(doc.page_content)
    print("\n")
