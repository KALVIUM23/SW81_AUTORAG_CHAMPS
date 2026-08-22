"""
End-to-End Grounded RAG Orchestrator with Guardrails (Topics 3.37, 3.38, 3.39, 3.40, 3.41)
"""
from typing import Dict, Any, List
from src.retrieval.hybrid_search import AutomotiveRetriever
from src.rag.llm_client import DiagnosticLLMClient
from src.rag.schemas import DiagnosticResponse, Citation, SafetyAdvisory


class AutomotiveRAGPipeline:
    CONFIDENCE_THRESHOLD = 0.65

    def __init__(self, retriever: AutomotiveRetriever, llm_client: DiagnosticLLMClient):
        self.retriever = retriever
        self.llm_client = llm_client

    def _build_context_envelope(self, chunks: List[Dict[str, Any]]) -> str:
        context_blocks = []
        for i, chunk in enumerate(chunks, 1):
            block = (
                f"--- SOURCE CHUNK [{i}] ---\n"
                f"Document ID: {chunk.get('document_id')}\n"
                f"Document Title: {chunk.get('document_name')}\n"
                f"Version: {chunk.get('version')} | Status: {chunk.get('status')}\n"
                f"Page: {chunk.get('page_number')} | Region: {chunk.get('region')}\n"
                f"Model: {chunk.get('vehicle_model')} {chunk.get('model_year')} ({chunk.get('variant')})\n"
                f"Content:\n{chunk.get('content')}\n"
            )
            context_blocks.append(block)
        return "\n\n".join(context_blocks)

    def process_query(
        self,
        vehicle_model: str,
        model_year: int,
        variant: str,
        region: str,
        query: str,
        top_k: int = 4
    ) -> DiagnosticResponse:
        # 1. Retrieve applicable chunks using metadata pre-filtering
        retrieved_chunks = self.retriever.retrieve(
            query=query,
            vehicle_model=vehicle_model,
            model_year=model_year,
            variant=variant,
            region=region,
            top_k=top_k
        )

        # 2. Guardrail Check: Trigger Refusal if zero chunks or below confidence threshold
        if not retrieved_chunks or retrieved_chunks[0].get("similarity_score", 0.0) < self.CONFIDENCE_THRESHOLD:
            return DiagnosticResponse(
                vehicle_context=f"{model_year} {vehicle_model} {variant} [{region}]",
                dtc_code=None,
                safety_advisory=SafetyAdvisory(has_active_recall=False),
                summary="Insufficient or inapplicable documentation.",
                steps=[],
                citations=[],
                is_refusal=True,
                refusal_reason=(
                    f"No authorized service documentation met the confidence threshold "
                    f"for {model_year} {vehicle_model} ({variant}) in region '{region}'. "
                    f"Please verify your vehicle configuration or escalate to technical support."
                )
            )

        # 3. Augment prompt with grounded context
        context_envelope = self._build_context_envelope(retrieved_chunks)

        # 4. Generate structured response
        response, _ = self.llm_client.run_completion(
            vehicle_model=vehicle_model,
            model_year=model_year,
            variant=variant,
            region=region,
            query=query,
            retrieved_context=context_envelope
        )

        return response
