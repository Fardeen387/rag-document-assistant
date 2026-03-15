# Retriev.AI — RAG Document Assistant

> Upload a document. Ask anything. Get accurate, source-grounded answers — fast.

**🔗 Live Demo:** [retriev.ai on Vercel](https://rag-document-assistant-beta.vercel.app/) &nbsp;|&nbsp; **ML Backend:** [HuggingFace Spaces](https://huggingface.co/spaces/Fardeen1004/rag-backend/tree/main)

---

## What It Does

Retriev.AI lets you upload any document (PDF, DOCX, or TXT) and ask natural language questions over it. It uses a **hybrid RAG (Retrieval-Augmented Generation)** pipeline to find the most relevant chunks and generate grounded, streamed answers — with a semantic cache to skip redundant LLM calls.

---

## Architecture

```
User uploads document (PDF / DOCX / TXT)
           │
           ▼
   Chunking + Embedding
           │
           ▼
    ChromaDB Vector Store
           │
    ┌──────┴───────┐
    │   User Query │
    │              ▼
    │    Semantic Cache ──hit──▶ Return cached answer
    │         │ miss
    │         ▼
    │  Vector Similarity Search
    │         │
    │         ▼
    │  LLM (HuggingFace Inference API)
    │         │ stream
    │         ▼
    └──────▶ User  ──▶  Cache store
```

---

## Key Features

- 📄 **Multi-format Upload** — supports PDF, DOCX, and TXT documents
- 🔍 **RAG Pipeline** — chunk splitting, embedding, and cosine similarity retrieval via ChromaDB
- ⚡ **Semantic Caching** — caches answers by query embedding to avoid redundant LLM calls; cache hits shown with a badge in the UI
- 🌊 **Streaming Responses** — token-by-token answer streaming for low perceived latency
- 🚀 **Split Deployment** — React frontend on Vercel, ML backend on HuggingFace Spaces

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React, Vercel |
| Backend API | FastAPI, Python |
| Vector Store | ChromaDB |
| Embeddings | HuggingFace Sentence Transformers |
| LLM | HuggingFace Inference API |
| Containerisation | Docker |
| Deployment | Vercel (frontend) · HuggingFace Spaces (backend) |

---

## Running Locally

**Prerequisites:** Python 3.10+, Node.js

```bash
# Clone
git clone https://github.com/Fardeen387/rag-document-assistant
cd rag-document-assistant

# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Create a `.env` file in `/backend`:
```
HF_TOKEN=your_huggingface_token_here
```

---

## What I Learned

- Resolving Docker port binding issues specific to HuggingFace Spaces deployments
- Implementing semantic caching keyed on query embeddings (not file ID + query — a subtle but critical distinction)
- Managing split deployments with CORS across Vercel and HuggingFace Spaces
- Streaming FastAPI responses to a React frontend in real time

---

## Author

**Fardeen Khan** — B.Tech CSE, BIST Bhopal  
Open to AI/ML internship opportunities → [fardeenkhan42504@gmail.com](mailto:fardeenkhan42504@gmail.com) · [LinkedIn](https://linkedin.com/in/fardeenkhan)
