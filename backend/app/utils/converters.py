"""
Helper functions for converting between LangChain and Pydantic models
"""
from typing import TYPE_CHECKING
from langchain.schema import Document

if TYPE_CHECKING:
    from app.api.schemas import RetrievedDocument, DocumentMetadata


def document_to_schema(doc: Document, score: float = None):
    """Convert LangChain Document to Pydantic RetrievedDocument schema"""
    from app.api.schemas import RetrievedDocument, DocumentMetadata
    
    return RetrievedDocument(
        content=doc.page_content,
        metadata=DocumentMetadata(**doc.metadata),
        relevance_score=score
    )
