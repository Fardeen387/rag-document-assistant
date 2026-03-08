import os
import sys
from retrieve.process import RAGEngine
from ingestion.loader import load_pdf_with_metadata
from ingestion.cleaner import clean_pages
from ingestion.chunker import create_metadata_chunks
from llm.gemini_client import GeminiService

load_dotenv()
# Ensure terminal handles special characters correctly
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def run_pro_rag():
    # --- CONFIGURATION ---
    # Fardeen, ensure you use a fresh key from AI Studio
    API_KEY = os.getenv("GEMINI_API_KEY")
    PDF_PATH = "ingestion/ml_notes.pdf"
    
    # Initialize Engine (Hybrid + Cache) and Gemini Service
    engine = RAGEngine()
    gemini = GeminiService(api_key=API_KEY)

    # --- PHASE 1: INGESTION ---
    print(f"\n[1/3] Preparing Document: {PDF_PATH}")
    raw_data = load_pdf_with_metadata(PDF_PATH)
    cleaned = clean_pages(raw_data)
    chunks = create_metadata_chunks(cleaned)
    
    # Ingests using Hybrid (Dense + BM25) logic
    doc_id = engine.process_and_ingest(chunks, PDF_PATH)

    # --- PHASE 2: THE CHAT LOOP ---
    print("\n" + "="*50)
    print(" 🚀 PRO RAG SYSTEM ACTIVE (Hybrid Search + Cache)")
    print(" Type your questions below. Type 'exit' to quit.")
    print("="*50)

    while True:
        query = input("\n💬 USER: ").strip()
        if query.lower() in ['exit', 'quit', 'bye']:
            break
        if not query:
            continue

        # STEP A: Check Semantic Cache first
        # This retrieves results with >0.92 similarity instantly
        cached_answer = engine.check_cache(query)
        
        if cached_answer:
            print(f"\n⚡ [CACHE HIT] AI ANSWER:\n{cached_answer}")
            continue

        # STEP B: Hybrid Search (Vector + BM25 Keywords)
        # Reciprocal Rank Fusion (RRF) merges the results
        print("🔍 Searching across document using Hybrid Retrieval...")
        results = engine.search(query, file_id=doc_id)

        if not results:
            print("⚠️ AI: No relevant information found in the document.")
            continue

        # STEP C: Generate Answer with Gemini
        # We use the top match score for debugging (e.g., 9.59)
        print(f"🧠 Reasoning with Gemini (Top Match Score: {results[0].score:.2f})...")
        answer = gemini.generate_answer(query, results)

        # STEP D: Store in Cache (ONLY if it's a valid answer)
        # This prevents the "Error 404" from being memorized
        if "Error" not in answer and "NOT_FOUND" not in answer:
            engine.add_to_cache(query, answer)
            print("✅ Answer validated and saved to cache.")
        else:
            print("⚠️ Answer contained an API error; skipping cache storage.")

        print("\n" + "-"*50)
        print(f"📖 AI ANSWER:\n{answer}")
        
        # Display Citations for the presentation
        print("\nSOURCES:")
        for res in results[:2]: 
            page = res.payload['metadata'].get('page', 'Unknown')
            print(f"📍 Page {page} (Relevance: {res.score:.2f})")
        print("-" * 50)

if __name__ == "__main__":
    try:
        run_pro_rag()
    except Exception as e:
        print(f"❌ SYSTEM ERROR: {e}")