from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

DATA_PATH = "D:/_RSSA HCP057/Instruments Guide"
DB_PATH = "vector_db/chroma_db"

documents = []

for file in Path(DATA_PATH).glob("*.pdf"):
    loader = PyPDFLoader(str(file))
    documents.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory=DB_PATH,
    collection_name="instrument_knowledge"
)

print("Vector DB created")