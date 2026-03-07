from process import RAGEngine
import sys

# Force UTF-8 for Windows Terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def start_chat():
    engine = RAGEngine()
    
    # In a real app, you'd get this ID from your database or session
    # For now, let's process the file once to get the ID
    file_path = "ingestion/ml_notes.pdf"
    print("--- Preparing Document ---")
    doc_id = engine.process_and_ingest(file_path)
    
    print("\n" + "="*50)
    print("READY! Type your questions below.")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50)

    while True:
        query = input("\nYOU: ").strip()
        
        if query.lower() in ['exit', 'quit']:
            print("Ending session. Goodbye!")
            break
            
        if not query:
            continue

        print(f"Searching across 600-page context...")
        results = engine.search(query, file_id=doc_id)

        if not results:
            print("AI: I couldn't find anything relevant in the document.")
            continue

        print("\nTOP SOURCES FOUND:")
        for i, res in enumerate(results):
            # Show the user which page the answer is on
            print(f"- [Page {res.payload['metadata']['page']}] (Match: {res.score:.2f})")
        
        # This is where we will eventually plug in Gemini
        # For now, let's show the most relevant chunk
        best_chunk = results[0].payload['content']
        print(f"\nRELEVANT EXCERPT:\n{best_chunk}")
        print("-" * 50)

if __name__ == "__main__":
    start_chat()