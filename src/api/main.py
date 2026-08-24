"""
FastAPI Server Entrypoint with CORS and Middleware (Topic 3.44)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router

app = FastAPI(
    title="AutoRAG Diagnostic Engine API",
    description="Enterprise Multi-Modal RAG Platform for Automotive Service Bays",
    version="1.0.0"
)

# Workshop Bay Browser Client CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "service": "AutoRAG Diagnostic Backend",
        "docs_url": "/docs",
        "health_url": "/api/v1/health"
    }