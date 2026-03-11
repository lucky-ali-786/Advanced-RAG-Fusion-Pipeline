# 🚀 Advanced RAG Pipeline with Reciprocal Rank Fusion (RRF)

An end-to-end, high-performance Retrieval-Augmented Generation (RAG) system built to query complex, 100+ page PDF documents. 

Unlike standard RAG pipelines that rely on a single vector search, this project implements **Multi-Query Retrieval** and a custom **Reciprocal Rank Fusion (RRF)** algorithm from scratch to mathematically deduplicate and re-rank the most relevant context chunks before passing them to the LLM.

## ✨ Key Features
* **Multi-Query Generation:** Uses Gemini to dynamically generate 3 variations of the user's prompt to capture broader semantic intent.
* **Parallel Vector Search:** Queries a local Qdrant database simultaneously for all prompt variations.
* **Custom RRF Algorithm:** A scratch-built Python function that deduplicates text chunks and calculates a fusion score based on cross-query rankings.
* **Smart Chunking:** Utilizes `RecursiveCharacterTextSplitter` to manage overlaps and preserve document context.
* **Sub-Second Latency:** Optimized indexing and retrieval pipeline for instant contextual extraction.

## 🧠 Architecture Flow
1. **Ingestion:** PDFs -> LangChain Text Splitter -> Gemini Embeddings -> Qdrant Vector DB.
2. **Multi-Query:** User Input -> LLM generates 3 alternative questions.
3. **Parallel Retrieval:** Qdrant performs similarity search for all 3 queries.
4. **Rank Fusion (RRF):** Results are mathematically merged, deduplicated, and ranked.
5. **Generation:** Top *N* fused chunks + Original Query -> Gemini 2.5 API -> Markdown Response.

## 🛠️ Tech Stack
* **Language:** Python
* **LLM & Embeddings:** Google Gemini 2.5 Flash Lite, Google GenAI Embeddings
* **Vector Database:** Qdrant (Local Docker Instance)
* **Framework:** LangChain (`PyPDFLoader`, `QdrantVectorStore`)

## 🚀 Getting Started

### Prerequisites
1. Python 3.9+
2. Docker (to run Qdrant locally)
3. Google Gemini API Key

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/pdf-rag-pipeline.git](https://github.com/your-username/pdf-rag-pipeline.git)
   cd pdf-rag-pipeline
2. **Install dependencies:**
   ```bash
   pip install langchain langchain-google-genai langchain-qdrant qdrant-client google-genai pypdf
3. **Start the Qdrant Vector Database:**
  ```bash
    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
