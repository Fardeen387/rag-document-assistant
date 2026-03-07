from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import shutil
import os
from pydantic import BaseModel
from retrieve.process import RAGEngine
from llm.gemini_client import GeminiService

# Ensure an uploads directory exists
os.makedirs("uploads", exist_ok=True)

# 1. Initialize 'app' ONLY ONCE
app = FastAPI(title="Fardeen's RAG API")

# 2. Setup CORS correctly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Define QuestionRequest BEFORE using it in routes
class QuestionRequest(BaseModel):
    query: str
    file_id: str

# 4. Initialize your Engine and Service
engine = RAGEngine()
gemini = GeminiService(api_key="AIzaSyADcFFjrLAbVmCyE66y068D87WpJcnGEcc")

# 5. Routes
@app.get("/")
async def root():
    return FileResponse("index.html")

@app.post("/upload")
async def upload_and_ingest(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    from ingestion.loader import load_pdf_with_metadata
    from ingestion.cleaner import clean_pages
    from ingestion.chunker import create_metadata_chunks

    raw_data = load_pdf_with_metadata(file_path)
    cleaned = clean_pages(raw_data)
    chunks = create_metadata_chunks(cleaned)
    file_id = engine.process_and_ingest(chunks, file_path)

    return {"file_id": file_id, "status": "success"}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        # 1. Check Cache
        cached = engine.check_cache(request.query)
        if cached:
            return {"answer": cached, "source": "cache"}
        
        # 2. Search & Generate
        results = engine.search(request.query, file_id=request.file_id)
        answer = gemini.generate_answer(request.query, results)
        
        # --- THE FIX STARTS HERE ---
        # 3. Only cache if the answer is NOT an error
        error_keywords = ["Error", "503", "demand", "quota", "limit"]
        is_error = any(word.lower() in answer.lower() for word in error_keywords)

        if not is_error:
            engine.add_to_cache(request.query, answer)
            return {"answer": answer, "source": "llm"}
        else:
            # Return the error but DON'T save it to the cache
            return {"answer": answer, "source": "error"}
        # --- THE FIX ENDS HERE ---
        
    except Exception as e:
        return {"answer": f"System Error: {str(e)}", "source": "error"}