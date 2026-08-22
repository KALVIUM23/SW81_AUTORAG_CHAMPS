from src.ingestion.chunker import AutomotiveChunker
from src.retrieval.embedder import AutomotiveEmbedder
from src.retrieval.vector_store import QdrantStore
from src.retrieval.hybrid_search import AutomotiveRetriever
from src.rag.llm_client import DiagnosticLLMClient
from src.rag.orchestrator import AutomotiveRAGPipeline

if __name__ == "__main__":
    print("1. Initializing full RAG Pipeline components...")
    embedder = AutomotiveEmbedder()
    store = QdrantStore(location=":memory:")
    retriever = AutomotiveRetriever(store, embedder)
    llm_client = DiagnosticLLMClient()
    pipeline = AutomotiveRAGPipeline(retriever, llm_client)

    # Ingest verified manual chunk
    doc_ind = {
        "document_id": "WSM-2025-MODX-IND-001",
        "document_name": "Apex Model X Powertrain Manual",
        "vehicle_model": "Model X",
        "model_year": 2025,
        "variant": "Hybrid",
        "region": "India",
        "document_type": "WSM",
        "version": "4.2",
        "status": "ACTIVE"
    }
    chunker = AutomotiveChunker()
    chunks = chunker.chunk_document(
        doc_ind,
        [{"page_number": 37, "raw_text": "For DTC P0420, inspect post-catalytic O2 sensor voltage. Target is <0.2V variance. Torque sensor to 45 Nm."}]
    )
    vectors = embedder.embed_texts([c["content"] for c in chunks])
    store.index_chunks(chunks, vectors)

    print("\n--- TEST CASE 1: Valid Grounded Query ---")
    res1 = pipeline.process_query(
        vehicle_model="Model X",
        model_year=2025,
        variant="Hybrid",
        region="India",
        query="How to diagnose P0420 and what is torque?"
    )
    print(f"Refusal State: {res1.is_refusal}")
    print(f"Summary: {res1.summary}")
    for step in res1.steps:
        print(f"Step {step.step_number}: {step.title} | Torque: {step.torque_or_electrical_spec}")

    print("\n--- TEST CASE 2: Guardrail Refusal Query (Out-of-Scope Region) ---")
    res2 = pipeline.process_query(
        vehicle_model="Model X",
        model_year=2025,
        variant="Hybrid",
        region="Europe",  # No European chunks indexed
        query="How to diagnose P0420?"
    )
    print(f"Refusal State: {res2.is_refusal}")
    print(f"Refusal Reason: {res2.refusal_reason}")