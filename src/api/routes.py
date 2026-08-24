"""
FastAPI Route Handlers for AutoRAG (Topics 3.44, 3.45)
"""
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from src.api.dependencies import get_services, ServiceContainer
from src.rag.schemas import DiagnosticResponse
from src.ingestion.loaders import DocumentLoader
from src.ingestion.cleaner import TextCleaner
from src.ingestion.chunker import AutomotiveChunker
from src.ingestion.validator import IngestionValidator

router = APIRouter(prefix="/api/v1", tags=["Diagnostic & Ingestion"])


class DiagnosticRequest(BaseModel):
    vehicle_model: str = Field(..., example="Model X")
    model_year: int = Field(..., example=2025)
    variant: str = Field(default="Hybrid", example="Hybrid")
    region: str = Field(default="India", example="India")
    query: str = Field(..., example="How to diagnose DTC P0420 catalytic efficiency?")


@router.post("/diagnose", response_model=DiagnosticResponse)
async def diagnose(
    request: DiagnosticRequest,
    services: ServiceContainer = Depends(get_services)
):
    """Executes grounded diagnostic RAG retrieval with safety guardrails."""
    try:
        response = services.pipeline.process_query(
            vehicle_model=request.vehicle_model,
            model_year=request.model_year,
            variant=request.variant,
            region=request.region,
            query=request.query
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic pipeline error: {str(e)}")


@router.post("/ingest")
async def ingest_manual(
    document_id: str = Form(...),
    document_name: str = Form(...),
    vehicle_model: str = Form(...),
    model_year: int = Form(...),
    variant: str = Form("ALL"),
    region: str = Form("GLOBAL"),
    version: str = Form("1.0"),
    file: UploadFile = File(...),
    services: ServiceContainer = Depends(get_services)
):
    """Uploads and indexes an automotive manual into Qdrant."""
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        loader = DocumentLoader()
        raw_pages = loader.load_file(temp_path)

        for page in raw_pages:
            page["raw_text"] = TextCleaner.clean_page_text(page["raw_text"])

        metadata = {
            "document_id": document_id,
            "document_name": document_name,
            "vehicle_model": vehicle_model,
            "model_year": model_year,
            "variant": variant,
            "region": region,
            "document_type": "WSM",
            "version": version,
            "status": "ACTIVE"
        }

        chunker = AutomotiveChunker()
        chunks = chunker.chunk_document(metadata, raw_pages)

        # Index into vector database
        vectors = services.embedder.embed_texts([c["content"] for c in chunks])
        services.vector_store.index_chunks(chunks, vectors)

        report = IngestionValidator.generate_report(chunks)
        return {"status": "SUCCESS", "report": report}

    finally:
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.get("/health")
async def health_check(services: ServiceContainer = Depends(get_services)):
    """Telemetry and vector database health endpoint."""
    collections = services.vector_store.client.get_collections().collections
    return {
        "status": "HEALTHY",
        "vector_collections": [c.name for c in collections],
        "model": services.llm_client.model
    }