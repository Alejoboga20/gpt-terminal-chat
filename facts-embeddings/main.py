from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

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

for doc in docs:
    print(doc.page_content)
    print("\n")
