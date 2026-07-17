from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()  # reads variables from .env into environment

DIMENSIONS=1536

# --- Load ---
file_path = "JS-Notes.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
print(f"Loaded {len(docs)} pages")

# --- Split ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")

# --- Embed ---
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=DIMENSIONS,
)

# --- Store in Chroma ---
vector_store = Chroma.from_documents(
    documents=chunks,
    collection_name="example_collection",
    embedding=embeddings,
    persist_directory="./chroma_langchain_db",  # saves to disk
)
print(f"Stored {len(chunks)} chunk embeddings in Chroma")