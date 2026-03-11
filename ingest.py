import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
GEMINI_API_KEY = "YOUR_API_KEY_HERE"

def ingest_pdf():
    print("Starting Ingestion Pipeline...")

    # 1. Load the PDF
    file_path = Path(__file__).parent / "Interupts-1.pdf"
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    print(f"Loading PDF from: {file_path.name}")
    loader = PyPDFLoader(file_path=str(file_path))
    docs = loader.load()
    print(f"Loaded {len(docs)} pages.")

    # 2. Split the text into chunks
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(documents=docs)
    print(f"Created {len(split_docs)} overlapping chunks.")

    # 3. Initialize Embeddings
    print("Initializing Gemini Embedding Model...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    # 4. Ingest into Qdrant
    print("Pushing vectors to local Qdrant Database...")
    url = "http://localhost:6333"
    collection_name = "my_first_vector_db"

    # NOTE: Ensure your Qdrant Docker container is running before this step!
    QdrantVectorStore.from_documents(
        documents=split_docs,
        embedding=embeddings,
        url=url,
        collection_name=collection_name,
        
    )

    print(f"Success! Vectors successfully saved to collection: '{collection_name}'")

if __name__ == "__main__":
    ingest_pdf()
