import os
import uuid
import sys
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, SparseVectorParams, SparseIndexParams
)
from fastembed import SparseTextEmbedding

# Configure terminal for clean output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

class RAGEngine:
    def __init__(self):
        print("--- Initializing Professional Engine (Hybrid + Cache) ---")
        
        # 1. Models Initialization
        # Dense: For semantic meaning
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Sparse: For keyword matching (BM25)
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        # Reranker: For high-precision scoring
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # 2. Database Connection
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.collection_name = "dynamic_rag_pro"
        self.cache_collection = "llm_semantic_cache"
        
        self._setup_db()

    def _setup_db(self):
        """Creates collections with Hybrid and Cache configurations if they don't exist."""
        # Main Collection
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=384, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
                }
            )
            print(f"✅ Created Hybrid Collection: {self.collection_name}")
            # Create index for file_id filtering (required by Qdrant Cloud)
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.file_id",
                field_schema="keyword"
            )
            print("✅ Created index for metadata.file_id")

        # Cache Collection
        if not self.client.collection_exists(self.cache_collection):
            self.client.create_collection(
                collection_name=self.cache_collection,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            print(f"✅ Initialized Semantic Cache: {self.cache_collection}")

    def get_sparse_vector(self, text):
        """Helper: Converts text to BM25 sparse vector format."""
        embeddings = list(self.sparse_model.embed([text]))
        return models.SparseVector(
            indices=embeddings[0].indices.tolist(),
            values=embeddings[0].values.tolist()
        )

    def process_and_ingest(self, chunks, file_path):
        """Handles ID generation, Hybrid embedding, and batch uploading."""
        file_name = file_path.split("\\")[-1]
        file_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, file_name))
        
        print(f"Ingesting: {file_name} (ID: {file_id})")
        texts = [c.page_content for c in chunks]
        
        # Generate Dense Embeddings in batches
        dense_vectors = self.embed_model.encode(texts, batch_size=32, show_progress_bar=True)

        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, dense_vectors)):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": vector.tolist(),
                        "sparse": self.get_sparse_vector(chunk.page_content)
                    },
                    payload={
                        "content": chunk.page_content,
                        "metadata": {**chunk.metadata, "file_id": file_id}
                    }
                )
            )

        # Batch upload for efficiency
        print(f"Uploading {len(points)} chunks to Qdrant...")
        for i in range(0, len(points), 100):
            self.client.upsert(
                collection_name=self.collection_name, 
                points=points[i : i + 100]
            )
        
        print(f"Done! {file_name} is ready.")
        return file_id

    def check_cache(self, query, threshold=0.92):
        """Professional Feature: Semantic Cache Check using modern API."""
        query_vector = self.embed_model.encode(query).tolist()
        
        # Updated to query_points() to fix the AttributeError
        results = self.client.query_points(
            collection_name=self.cache_collection,
            query=query_vector,
            limit=1
        ).points
        
        if results and results[0].score >= threshold:
            # Check if an actual answer exists in the payload
            return results[0].payload.get('answer')
        return None

    def add_to_cache(self, query, answer):
        """Stores query and answer in the semantic cache."""
        query_vector = self.embed_model.encode(query).tolist()
        self.client.upsert(
            collection_name=self.cache_collection,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=query_vector,
                payload={"query": query, "answer": answer}
            )]
        )

    def search(self, query, file_id=None, top_k=10):
        """Hybrid Search (Dense + Sparse) with Cross-Encoder Reranking."""
        search_filter = None
        if file_id:
            search_filter = Filter(
                must=[FieldCondition(key="metadata.file_id", match=MatchValue(value=file_id))]
            )

        # 1. Hybrid Retrieval (RRF Fusion)
        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=self.embed_model.encode(query).tolist(),
                    using="dense",
                    filter=search_filter,
                    limit=20
                ),
                models.Prefetch(
                    query=self.get_sparse_vector(query),
                    using="sparse",
                    filter=search_filter,
                    limit=20
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k
        ).points

        if not response: 
            return []

        # 2. Precision Reranking
        pairs = [[query, r.payload['content']] for r in response]
        scores = self.reranker.predict(pairs)
        for i, res in enumerate(response):
            res.score = scores[i]
        
        # Sort by reranker score and return top 3 for Gemini
        return sorted(response, key=lambda x: x.score, reverse=True)[:3]