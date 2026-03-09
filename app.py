from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
import shutil, os, json
from pydantic import BaseModel
from retrieve.process import RAGEngine
from llm.gemini_client import GeminiService
from dotenv import load_dotenv

load_dotenv()

os.makedirs("uploads", exist_ok=True)

app = FastAPI(title="Fardeen's RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    query: str
    file_id: str

engine = RAGEngine()
gemini = GeminiService(api_key=os.getenv("GEMINI_API_KEY"))

@app.get("/")
async def root():
    return FileResponse("index.html")

@app.get("/health")
async def health():
    return JSONResponse(content={"status": "ok"})

@app.post("/upload")
async def upload_and_ingest(file: UploadFile = File(...)):
    """Synchronous upload — processes fully before returning."""
    try:
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
        return JSONResponse(content={"file_id": file_id, "status": "success"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        cached = engine.check_cache(request.query)
        if cached:
            return {"answer": cached, "source": "cache", "sources": []}

        results = engine.search(request.query, file_id=request.file_id)
        pages = []
        for r in results:
            try:
                page = r.payload['metadata']['page']
                if page is not None and page not in pages:
                    pages.append(int(page))
            except (KeyError, TypeError, AttributeError):
                pass
        pages.sort()

        answer = gemini.generate_answer(request.query, results)
        if "QUOTA EXCEEDED" in answer or "Error" in answer or answer.startswith("System Error"):
            return {"answer": answer, "source": "error", "sources": []}

        engine.add_to_cache(request.query, answer)
        return {"answer": answer, "source": "llm", "sources": pages}

    except Exception as e:
        return {"answer": f"System Error: {str(e)}", "source": "error", "sources": []}

@app.post("/ask/stream")
async def ask_stream(request: QuestionRequest):
    def generate():
        cached = engine.check_cache(request.query)
        if cached:
            yield f"data: {json.dumps({'type': 'meta', 'source': 'cache', 'sources': []})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'text': cached})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        try:
            results = engine.search(request.query, file_id=request.file_id)
            pages = []
            for r in results:
                try:
                    page = r.payload['metadata']['page']
                    if page is not None and page not in pages:
                        pages.append(int(page))
                except (KeyError, TypeError, AttributeError):
                    pass
            pages.sort()
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': f'Search failed: {str(e)}'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'meta', 'source': 'llm', 'sources': pages})}\n\n"

        full_answer = ""
        is_error = False
        for chunk in gemini.stream_answer(request.query, results):
            full_answer += chunk
            yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"
            if chunk.startswith("Error") or chunk.startswith("System Error") or "QUOTA EXCEEDED" in chunk:
                is_error = True
                break

        if not is_error and full_answer.strip():
            engine.add_to_cache(request.query, full_answer)

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
