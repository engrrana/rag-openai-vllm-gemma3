"""
FastAPI Application Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api import router
from app.api.schemas import HealthResponse
from app.services import rag_service



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    print("🚀 Starting RAG API...")
    rag_service.initialize()
    yield
    # Shutdown
    print("👋 Shutting down RAG API...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="Retrieval-Augmented Generation API for UET Lahore Graduate Admissions",
    version=settings.app_version,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router)


@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of the API and vector store.
    """
    return HealthResponse(
        status="healthy",
        vector_store_loaded=rag_service.is_initialized,
        total_documents=rag_service.total_documents
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Alternative health check endpoint"""
    return await health_check()
