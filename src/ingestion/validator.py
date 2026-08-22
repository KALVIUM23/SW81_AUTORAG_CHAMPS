"""
Ingestion Validation and Corpus Integrity Reporter (Topic 3.24)
"""
from typing import List, Dict, Any


class IngestionValidator:
    @staticmethod
    def generate_report(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chunks:
            return {"status": "FAILED", "error": "Zero chunks produced."}

        total_chunks = len(chunks)
        total_tokens = sum(c["token_count"] for c in chunks)
        unique_docs = len(set(c["document_id"] for c in chunks))
        missing_metadata_count = sum(
            1 for c in chunks if not all([c.get("vehicle_model"), c.get("region"), c.get("version")])
        )

        return {
            "status": "HEALTHY" if missing_metadata_count == 0 else "WARNING",
            "total_documents": unique_docs,
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": round(total_tokens / total_chunks, 1),
            "missing_metadata_chunks": missing_metadata_count,
        }