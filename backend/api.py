"""
FastAPI endpoints for RAG system
==================================
Two endpoints:
1. /retrieve - Get relevant documents only (no LLM)
2. /ask - Get complete answer with LLM
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import re
from pathlib import Path

# LangChain
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Utilities
import numpy as np
from collections import Counter

# ==================== CONFIGURATION ====================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")


PROJECT_ROOT = Path(__file__).parent
DATA_FILE = str(PROJECT_ROOT / "data" / "processed" / "rag_optimized_data.txt")
CHROMA_DB_DIR = str(PROJECT_ROOT / "chroma_db_openai")

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 5

# ==================== INITIALIZE APP ====================

app = FastAPI(
    title="RAG System API",
    description="OpenAI-powered RAG system with retrieval and QA endpoints",
    version="1.0.0"
)

# ==================== REQUEST/RESPONSE MODELS ====================

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What programs does Computer Science offer?",
                "top_k": 5
            }
        }

class DocumentResponse(BaseModel):
    content: str
    metadata: Dict
    
class RetrievalResponse(BaseModel):
    question: str
    documents: List[DocumentResponse]
    count: int
    
class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict]
    document_count: int

# ==================== LOAD RAG SYSTEM ====================

print("Loading RAG system...")

# Parse data
class SemanticChunker:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.documents = []
    
    def load_and_parse(self) -> List[Document]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        departments = re.split(r'=== DEPARTMENT \d+ ===', content)[1:]
        
        for dept_content in departments:
            self._parse_department(dept_content)
        
        return self.documents
    
    def _parse_department(self, dept_content: str):
        lines = dept_content.strip().split('\n')
        
        dept_name = None
        dept_number = None
        
        for line in lines[:5]:
            if line.startswith('Department Name:'):
                dept_name = line.replace('Department Name:', '').strip()
            elif line.startswith('Department Number:'):
                dept_number = line.replace('Department Number:', '').strip()
        
        if not dept_name:
            return
        
        sections = re.split(r'--- SECTION: (.+?) ---', dept_content)
        
        for i in range(1, len(sections), 2):
            if i+1 >= len(sections):
                break
            
            section_name = sections[i].strip()
            section_content = sections[i+1].strip()
            
            if not section_content:
                continue
            
            metadata = {
                'department_number': dept_number,
                'department_name': dept_name,
                'section': section_name,
                'full_context': f"{dept_name} - {section_name}",
                'source': 'university_prospectus'
            }
            
            if 'Introduction' in section_name:
                metadata['section_type'] = 'introduction'
            elif 'Program' in section_name:
                metadata['section_type'] = 'programs'
            elif 'Eligibility' in section_name:
                metadata['section_type'] = 'eligibility'
            elif 'Faculty' in section_name:
                metadata['section_type'] = 'faculty'
            
            doc = Document(
                page_content=section_content,
                metadata=metadata
            )
            
            self.documents.append(doc)

# Load documents
chunker = SemanticChunker(DATA_FILE)
documents = chunker.load_and_parse()
print(f"✅ Loaded {len(documents)} documents")

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY
)
print(f"✅ Embeddings initialized: {EMBEDDING_MODEL}")

# Load or create vector store
if os.path.exists(CHROMA_DB_DIR):
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name="university_prospectus_openai"
    )
    print(f"✅ Loaded existing vector store")
else:
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name="university_prospectus_openai"
    )
    print(f"✅ Created new vector store")

# Setup retrievers
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = TOP_K

hybrid_retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

# Initialize LLM
llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)
print(f"✅ LLM initialized: {LLM_MODEL}")

# Setup SelfQueryRetriever
metadata_field_info = [
    AttributeInfo(
        name="department_name",
        description="The full name of the academic department or institute",
        type="string"
    ),
    AttributeInfo(
        name="section_type",
        description="Type of section: 'introduction', 'programs', 'eligibility', or 'faculty'",
        type="string"
    ),
]

self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="University prospectus with departments, programs, eligibility, and faculty",
    metadata_field_info=metadata_field_info,
    verbose=False
)

retriever = self_query_retriever
print(f"✅ SelfQueryRetriever configured")

# Setup QA chain
qa_prompt_template = """You are a helpful assistant answering questions about university departments and programs.

Use the following context to provide accurate, specific answers. If the information is not in the context, say so clearly.

Context:
{context}

Question: {question}

Answer (be specific and cite the department/program names):"""

QA_PROMPT = PromptTemplate(
    template=qa_prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT}
)

print("✅ QA chain ready")
print("=" * 60)
print("RAG System loaded successfully!")
print("=" * 60)

# ==================== API ENDPOINTS ====================

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "RAG System API",
        "version": "1.0.0",
        "endpoints": {
            "/retrieve": "Get relevant documents only",
            "/ask": "Get complete answer with LLM"
        }
    }

@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_documents(request: QueryRequest):
    """
    Retrieve relevant documents without LLM answer generation.
    
    This endpoint uses the SelfQueryRetriever to find relevant documents
    based on the question, with automatic metadata filtering.
    
    **Use this when you only need the source documents.**
    """
    try:
        # Get relevant documents
        docs = retriever.get_relevant_documents(request.question)
        
        # Limit to top_k
        docs = docs[:request.top_k]
        
        # Format response
        doc_responses = [
            DocumentResponse(
                content=doc.page_content,
                metadata=doc.metadata
            )
            for doc in docs
        ]
        
        return RetrievalResponse(
            question=request.question,
            documents=doc_responses,
            count=len(doc_responses)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QueryRequest):
    """
    Get a complete answer using LLM with retrieved context.
    
    This endpoint:
    1. Retrieves relevant documents using SelfQueryRetriever
    2. Sends them to GPT-4o-mini for answer generation
    3. Returns the answer with source documents
    
    **Use this when you need a natural language answer.**
    """
    try:
        # Get answer from QA chain
        result = qa_chain({"query": request.question})
        
        # Format sources
        sources = [
            {
                "content": doc.page_content[:200] + "...",  # Preview
                "department": doc.metadata.get("department_name", "N/A"),
                "section": doc.metadata.get("section", "N/A"),
                "section_type": doc.metadata.get("section_type", "N/A"),
                "full_context": doc.metadata.get("full_context", "N/A")
            }
            for doc in result['source_documents'][:request.top_k]
        ]
        
        return AnswerResponse(
            question=request.question,
            answer=result['result'],
            sources=sources,
            document_count=len(result['source_documents'])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QA error: {str(e)}")

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "components": {
            "vectorstore": "ready",
            "embeddings": EMBEDDING_MODEL,
            "llm": LLM_MODEL,
            "documents_loaded": len(documents),
            "retriever": "SelfQueryRetriever"
        }
    }

# ==================== RUN SERVER ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
