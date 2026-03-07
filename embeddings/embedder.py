from sentence_transformers import SentenceTransformer

def get_embedding_model():
    """Initializes the pure SentenceTransformer model."""
    print("Loading SentenceTransformer: all-MiniLM-L6-v2...")
    # This downloads the model on the first run (approx 80MB)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def generate_embeddings(documents, model):
    """
    Directly encodes document content into vectors.
    Returns a list of vectors (NumPy arrays).
    """
    # Extract just the text strings from our structured Document objects
    texts = [doc.page_content for doc in documents]
    
    print(f"Encoding {len(texts)} chunks...")
    
    # model.encode handles the tokenization and pooling internally
    embeddings = model.encode(texts, show_progress_bar=True)
    
    return embeddings