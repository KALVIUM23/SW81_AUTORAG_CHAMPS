from src.ingestion.chunker import AutomotiveChunker
from src.retrieval.embedder import AutomotiveEmbedder
from src.retrieval.vector_store import QdrantStore
from src.retrieval.hybrid_search import AutomotiveRetriever

if __name__ == "__main__":
    print("1. Initializing Embedder and Qdrant in-memory store...")
    embedder = AutomotiveEmbedder()
    store = QdrantStore(location=":memory:")
    retriever = AutomotiveRetriever(store, embedder)

    # Ingest 2 distinct chunks: one for India BS-VI, one for US EPA
    doc_ind = {
        "document_id": "WSM-2025-IND-001",
        "document_name": "Model X India Manual",
        "vehicle_model": "Model X",
        "model_year": 2025,
        "variant": "Hybrid",
        "region": "India",
        "document_type": "WSM",
        "version": "4.2",
        "status": "ACTIVE"
    }
    doc_us = {
        "document_id": "WSM-2025-USA-001",
        "document_name": "Model X USA Manual",
        "vehicle_model": "Model X",
        "model_year": 2025,
        "variant": "Hybrid",
        "region": "USA",
        "document_type": "WSM",
        "version": "4.2",
        "status": "ACTIVE"
    }

    chunker = AutomotiveChunker()
    chunks_ind = chunker.chunk_document(doc_ind, [{"page_number": 37, "raw_text": "India BS-VI: DTC P0420 sensor voltage fluctuation must be <0.2V peak-to-peak. Torque is 45 Nm."}])
    chunks_us = chunker.chunk_document(doc_us, [{"page_number": 42, "raw_text": "USA EPA Tier 3: DTC P0420 sensor voltage fluctuation must be <0.1V peak-to-peak. Torque is 40 Nm."}])

    all_chunks = chunks_ind + chunks_us
    print(f"2. Generating embeddings for {len(all_chunks)} chunks...")
    vectors = embedder.embed_texts([c["content"] for c in all_chunks])

    print("3. Indexing chunks into Qdrant...")
    store.index_chunks(all_chunks, vectors)

    print("\n4. Testing Metadata Filter: Querying 'India' region...")
    results = retriever.retrieve(
        query="How to diagnose P0420 voltage fluctuation?",
        vehicle_model="Model X",
        model_year=2025,
        variant="Hybrid",
        region="India",
        top_k=2
    )

    for i, res in enumerate(results, 1):
        print(f"\n[Result {i}] Score: {res['similarity_score']}")
        print(f"Doc ID: {res['document_id']} | Region: {res['region']}")
        print(f"Content: {res['content']}")