"""
API Dependency Injection Container (Topic 3.44)
"""
from functools import lru_cache
from src.retrieval.embedder import AutomotiveEmbedder
from src.retrieval.vector_store import QdrantStore
from src.retrieval.hybrid_search import AutomotiveRetriever
from src.rag.llm_client import DiagnosticLLMClient
from src.rag.orchestrator import AutomotiveRAGPipeline


class ServiceContainer:
    def __init__(self):
        self.embedder = AutomotiveEmbedder()
        self.vector_store = QdrantStore(location=":memory:")
        self.retriever = AutomotiveRetriever(self.vector_store, self.embedder)
        self.llm_client = DiagnosticLLMClient()
        self.pipeline = AutomotiveRAGPipeline(self.retriever, self.llm_client)


@lru_cache()
def get_services() -> ServiceContainer:
    return ServiceContainer()