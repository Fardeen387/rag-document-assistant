from qdrant_client import QdrantClient

# 1. Connect to your local Qdrant (Docker)
client = QdrantClient("http://localhost:6333")

COLLECTION_NAME = "semantic_cache"

try:
    # 2. Delete the collection entirely
    client.delete_collection(collection_name=COLLECTION_NAME)
    print(f"✅ Successfully deleted {COLLECTION_NAME}")
    
    # 3. Optional: Re-create it immediately (Empty)
    # This ensures your app doesn't crash on the next 'ask'
    from qdrant_client.http.models import VectorParams, Distance
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    print(f"✨ Fresh '{COLLECTION_NAME}' collection created and ready!")

except Exception as e:
    print(f"❌ Error clearing cache: {e}")