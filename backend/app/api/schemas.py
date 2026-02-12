"""
Pydantic schemas for RAG API request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class QuestionRequest(BaseModel):
    """Request schema for asking a question"""
    question: str = Field(..., min_length=1, description="The question to ask the RAG system")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What programs does the Department of Computer Science offer?"
            }
        }


class DocumentMetadata(BaseModel):
    """Metadata for a retrieved document"""
    department_name: str
    section: str
    section_type: str
    full_context: str
    department_number: Optional[str] = None
    source: Optional[str] = None


class RetrievedDocument(BaseModel):
    """Schema for a single retrieved document"""
    content: str = Field(..., description="The text content of the document chunk")
    metadata: DocumentMetadata
    relevance_score: Optional[float] = Field(None, description="Similarity score (if available)")


class RetrievalResponse(BaseModel):
    """Response schema for retrieval endpoint"""
    question: str
    retrieved_documents: List[RetrievedDocument]
    total_documents: int
    retrieval_method: str = Field(..., description="Method used: 'self_query' or 'semantic_fallback'")
    filter_applied: Optional[str] = Field(None, description="The metadata filter that was applied (if any)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What programs does Computer Science offer?",
                "retrieved_documents": [
                    {
                        "content": "Program: Ph.D. Computer Science\nProgram: M.Sc. Computer Science",
                        "metadata": {
                            "department_name": "Department of Computer Science",
                            "section": "Offered Programs",
                            "section_type": "programs",
                            "full_context": "Department of Computer Science - Offered Programs"
                        }
                    }
                ],
                "total_documents": 1,
                "retrieval_method": "self_query",
                "filter_applied": "eq('department_name', 'Department of Computer Science')"
            }
        }


class AnswerResponse(BaseModel):
    """Response schema for answer endpoint (with LLM)"""
    question: str
    answer: str
    source_documents: List[RetrievedDocument]
    total_sources: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What programs does Computer Science offer?",
                "answer": "The Department of Computer Science offers Ph.D. Computer Science, M.Sc. Computer Science, M.Sc. Software Engineering, and M.Sc. Data Science programs.",
                "source_documents": [
                    {
                        "content": "Program: Ph.D. Computer Science\nProgram: M.Sc. Computer Science",
                        "metadata": {
                            "department_name": "Department of Computer Science",
                            "section": "Offered Programs",
                            "section_type": "programs",
                            "full_context": "Department of Computer Science - Offered Programs"
                        }
                    }
                ],
                "total_sources": 1
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    vector_store_loaded: bool
    total_documents: int
