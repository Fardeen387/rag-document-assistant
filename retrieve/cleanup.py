from qdrant_client impoq QdrantClient

client = QdrantClient(host="localhost", port=6333)

def clear_data(collection_name="dynamic_rag"):
    if client.collection_exists(collection_name):
        print(f"Deleting Duplicates in '{collection_name}'...")
        client.delete_collection(collection_name)
        print("Success! Collection is empty.")
    else:
        print("Collection not found.")

if __main__ == "__main__":
    clear_data()