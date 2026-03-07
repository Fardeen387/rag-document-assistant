import sys
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Force terminal to use UTF-8 encoding for printing
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 1. Initialize
client = QdrantClient("http://localhost:6333") 
model = SentenceTransformer('all-MiniLM-L6-v2')
COLLECTION = "ml_notes"

def ask_my_docs(query_text, top_k=3):
    # Vectorize the question
    query_vector = model.encode(query_text).tolist()
    
    try:
        # Using query_points for the latest Qdrant API
        response = client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )
        
        results = response.points

        print(f"\nSEARCHING FOR: '{query_text}'")
        print("-" * 50)
        
        if not results:
            print("No relevant information found.")
            return

        for i, hit in enumerate(results):
            payload = hit.payload
            content = payload.get('content', 'No content')
            metadata = payload.get('metadata', {})
            page = metadata.get('page', 'N/A')
            chunk_id = metadata.get('chunk_id', 'N/A')
            
            print(f"[{i+1}] Score: {hit.score:.2f} | Page: {page} | ID: {chunk_id}")
            print(f"Text: {content[:200]}...")
            print("-" * 30)
        

    except Exception as e:
        # Error message without the emoji to avoid the crash
        print(f"ERROR during search: {str(e)}")

# Execution
if __name__ == "__main__":
    ask_my_docs("Transformation Pipelines")