"""
Qdrant Vector Database Connector and Indexer (Topics 3.30, 3.31)
"""
import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models


class QdrantStore:
    def __init__(
        self,
        collection_name: str = "automotive_knowledge_base",
        location: str = ":memory:"
    ):
        self.collection_name = collection_name
        self.location = location
        self.client = QdrantClient(location=location)
        self._ensure_collection_exists()

    def _ensure_collection_exists(self, vector_dim: int = 384):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_dim,
                    distance=models.Distance.COSINE
                )
            )
            # Create payload indexes only for server instances (local in-memory indexes automatically)
            if self.location not in [":memory:", None]:
                for field in ["vehicle_model", "model_year", "variant", "region", "document_type", "status"]:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD
                    )

    def index_chunks(self, chunks: List[Dict[str, Any]], vectors: List[List[float]]):
        """Bulk indexes chunks and vectors into Qdrant."""
        points = []
        for chunk, vector in zip(chunks, vectors):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=chunk
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)