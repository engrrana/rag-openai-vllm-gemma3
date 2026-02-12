"""
API Routes for RAG endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import List

from app.api.schemas import (
    QuestionRequest,
    RetrievalResponse,
    AnswerResponse
)
from app.services import rag_service
from app.utils.converters import document_to_schema

router = APIRouter(prefix="/api/v1", tags=["RAG"])


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_documents(request: QuestionRequest):
    """
    Retrieve relevant documents without LLM answer generation.
    
    This endpoint returns only the retrieved chunks with metadata,
    useful for debugging or building custom UIs.
    
    - **question**: The user's question
    
    Returns:
    - Retrieved documents with metadata
    - Total number of documents
    - Retrieval method used
    - Filter applied (if any)
    """
    try:
        # Get the structured query to see the filter
        structured_query = rag_service.self_query_retriever.query_constructor.invoke(request.question)
        filter_str = str(structured_query.filter) if structured_query.filter else None
        
        # Get relevant documents
        docs = rag_service.retrieve_documents(request.question)
        
        # Convert to schema
        retrieved_docs = [document_to_schema(doc) for doc in docs]
        
        return RetrievalResponse(
            question=request.question,
            retrieved_documents=retrieved_docs,
            total_documents=len(retrieved_docs),
            retrieval_method="self_query",
            filter_applied=filter_str
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval error: {str(e)}"
        )


@router.post("/answer", response_model=AnswerResponse)
async def get_answer(request: QuestionRequest):
    """
    Get LLM-generated answer with source documents.
    
    This endpoint processes the question through the full RAG pipeline:
    1. Retrieves relevant documents
    2. Generates answer using GPT-4o
    3. Returns answer with source citations
    
    - **question**: The user's question
    
    Returns:
    - Generated answer
    - Source documents used
    - Total number of sources
    """
    try:
        # Get answer from QA chain
        result = rag_service.get_answer(request.question)
        
        # Convert source documents to schema
        source_docs = [
            document_to_schema(doc) 
            for doc in result['source_documents']
        ]
        
        return AnswerResponse(
            question=request.question,
            answer=result['result'],
            source_documents=source_docs,
            total_sources=len(source_docs)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Answer generation error: {str(e)}"
        )
