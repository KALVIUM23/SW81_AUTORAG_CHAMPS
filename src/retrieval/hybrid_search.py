"""
Metadata-Filtered Retrieval and Candidate Ranking Engine (Topics 3.33, 3.35)
"""
from typing import List, Dict, Any
from qdrant_client.http import models
from src.retrieval.embedder import AutomotiveEmbedder
from src.retrieval.vector_store import QdrantStore


class AutomotiveRetriever:
    def __init__(self, vector_store: QdrantStore, embedder: AutomotiveEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        vehicle_model: str,
        model_year: int,
        variant: str,
        region: str,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        query_vector = self.embedder.embed_query(query)

        # Construct strict metadata pre-filters
        filter_conditions = [
            models.FieldCondition(key="vehicle_model", match=models.MatchValue(value=vehicle_model)),
            models.FieldCondition(key="model_year", match=models.MatchValue(value=model_year)),
            models.FieldCondition(key="status", match=models.MatchValue(value="ACTIVE")),
        ]

        if region != "GLOBAL":
            filter_conditions.append(
                models.FieldCondition(key="region", match=models.MatchValue(value=region))
            )

        query_filter = models.Filter(must=filter_conditions)

        # Modern Qdrant Query API
        response = self.vector_store.client.query_points(
            collection_name=self.vector_store.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True
        )

        ranked_results = []
        for point in response.points:
            payload = dict(point.payload)
            payload["similarity_score"] = round(point.score, 4)
            ranked_results.append(payload)

        return ranked_results