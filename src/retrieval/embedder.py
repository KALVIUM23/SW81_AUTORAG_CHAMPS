"""
Local Embedding Engine using FastEmbed (Topics 3.25, 3.26, 3.28)
"""
from typing import List
import numpy as np
from fastembed import TextEmbedding


class AutomotiveEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.embedding_model = TextEmbedding(model_name=self.model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Batch generates vector embeddings as standard Python float lists."""
        if not texts:
            return []
        embeddings_generator = self.embedding_model.embed(texts)
        return [embedding.tolist() for embedding in embeddings_generator]

    def embed_query(self, query: str) -> List[float]:
        """Embeds a single query string for vector search."""
        return self.embed_texts([query])[0]