"""
Token-Aware Automotive Semantic Chunker (Topics 3.21, 3.22, 3.23)
"""
import uuid
from typing import List, Dict, Any
from src.rag.token_budget import TokenManager


class AutomotiveChunker:
    def __init__(self, chunk_size_tokens: int = 512, overlap_tokens: int = 64):
        self.chunk_size = chunk_size_tokens
        self.overlap = overlap_tokens
        self.token_manager = TokenManager()

    def chunk_document(
        self,
        document_metadata: Dict[str, Any],
        pages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        chunks = []
        doc_id = document_metadata.get("document_id", str(uuid.uuid4()))
        doc_title = document_metadata.get("document_name", "Service Manual")

        for page in pages:
            page_num = page["page_number"]
            text = page["raw_text"]
            
            # Split text by paragraphs first to preserve semantic diagnostic steps
            paragraphs = text.split("\n\n")
            current_chunk_text = ""

            for paragraph in paragraphs:
                candidate = f"{current_chunk_text}\n\n{paragraph}".strip() if current_chunk_text else paragraph
                token_count = self.token_manager.count_tokens(candidate)

                if token_count <= self.chunk_size:
                    current_chunk_text = candidate
                else:
                    if current_chunk_text:
                        chunk_id = f"{doc_id}_P{page_num:03d}_C{len(chunks)+1:03d}"
                        chunks.append({
                            "chunk_id": chunk_id,
                            "document_id": doc_id,
                            "document_name": doc_title,
                            "page_number": page_num,
                            "vehicle_model": document_metadata.get("vehicle_model", "ALL"),
                            "model_year": document_metadata.get("model_year", 2025),
                            "variant": document_metadata.get("variant", "ALL"),
                            "region": document_metadata.get("region", "GLOBAL"),
                            "document_type": document_metadata.get("document_type", "WSM"),
                            "version": document_metadata.get("version", "1.0"),
                            "status": document_metadata.get("status", "ACTIVE"),
                            "content": current_chunk_text,
                            "token_count": self.token_manager.count_tokens(current_chunk_text)
                        })
                    current_chunk_text = paragraph

            if current_chunk_text:
                chunk_id = f"{doc_id}_P{page_num:03d}_C{len(chunks)+1:03d}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "document_name": doc_title,
                    "page_number": page_num,
                    "vehicle_model": document_metadata.get("vehicle_model", "ALL"),
                    "model_year": document_metadata.get("model_year", 2025),
                    "variant": document_metadata.get("variant", "ALL"),
                    "region": document_metadata.get("region", "GLOBAL"),
                    "document_type": document_metadata.get("document_type", "WSM"),
                    "version": document_metadata.get("version", "1.0"),
                    "status": document_metadata.get("status", "ACTIVE"),
                    "content": current_chunk_text,
                    "token_count": self.token_manager.count_tokens(current_chunk_text)
                })

        return chunks