"""
RAG System Service - Handles all RAG operations
"""
import os
from typing import List
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from app.core.config import settings
from app.utils.chunker import SemanticChunker


class RAGService:
    """Singleton service for RAG operations"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.documents: List[Document] = []
        self.vectorstore = None
        self.self_query_retriever = None
        self.qa_chain = None
        self._initialized = False
    
    def initialize(self):
        """Initialize all RAG components"""
        if self._initialized:
            print("⚠️ RAG service already initialized")
            return
        
        print("🔄 Initializing RAG service...")
        
        # 1. Load and parse documents
        self._load_documents()
        
        # 2. Initialize embeddings
        embeddings = self._create_embeddings()
        
        # 3. Load or create vector store
        self._initialize_vectorstore(embeddings)
        
        # 4. Setup retrievers
        self._setup_retrievers()
        
        # 5. Setup QA chain
        self._setup_qa_chain()
        
        self._initialized = True
        print(f"✅ RAG service initialized with {len(self.documents)} documents")
    
    def _load_documents(self):
        """Load and parse documents from data file"""
        chunker = SemanticChunker(settings.data_file)
        self.documents = chunker.load_and_parse()
    
    def _create_embeddings(self) -> OpenAIEmbeddings:
        """Create OpenAI embeddings instance"""
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key
        )
    
    def _initialize_vectorstore(self, embeddings: OpenAIEmbeddings):
        """Load existing or create new vector store"""
        if os.path.exists(settings.chroma_db_path):
            print("📂 Loading existing ChromaDB...")
            self.vectorstore = Chroma(
                persist_directory=settings.chroma_db_path,
                embedding_function=embeddings
            )
        else:
            print("🔨 Building new ChromaDB...")
            self.vectorstore = Chroma.from_documents(
                documents=self.documents,
                embedding=embeddings,
                persist_directory=settings.chroma_db_path
            )
    
    def _setup_retrievers(self):
        """Setup self-query retriever with metadata filtering"""
        # Main LLM for query construction (matching notebook)
        main_llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )
        
        # Metadata field definitions (exact from notebook)
        metadata_field_info = [
            AttributeInfo(
                name="department_name",
                description="The official name found in the document (must start with 'Department of '). If a user mentions ANY department, you MUST create an 'eq' filter for this field and wrap it in an 'AND' with other filters.",
                type="string"
            ),
            AttributeInfo(
                name="section_type",
                description="Must be one of: 'introduction', 'programs', 'eligibility', 'faculty'. If question contains 'program'/'course'/'degree', filter by 'programs'. If contains 'eligibility'/'requirement'/'criteria'/'admission', filter by 'eligibility'. If contains 'faculty'/'professor'/'chairman'/'teacher', filter by 'faculty'. If contains 'introduction'/'about'/'overview', filter by 'introduction'. Combine with department_name using AND.",
                type="string"
            ),
        ]
        
        # Create self-query retriever (exact from notebook)
        self.self_query_retriever = SelfQueryRetriever.from_llm(
            llm=main_llm,
            vectorstore=self.vectorstore,
            document_contents="Prospectus for university departments. Each document is a single section (Intro, Programs, Eligibility, or Faculty).",
            metadata_field_info=metadata_field_info,
            verbose=True
        )
    
    def _setup_qa_chain(self):
        """Setup QA chain with custom prompt"""
        llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )
        
        qa_prompt_template = """You are a helpful assistant for UET Lahore's admissions.

Answer the question based on the context below. If the context contains relevant information, provide a complete answer. Only if the context is completely unrelated to the question, say: "I apologize, but I don't have information about [topic] in the prospectus. Please contact the admissions office at admission@uet.edu.pk."

Note: If asked about specific sessions (e.g., "this spring"), mention that session dates are not specified in the prospectus.

Context: {context}

Question: {question}

Answer:"""
        
        qa_prompt = PromptTemplate(
            template=qa_prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.self_query_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_prompt}
        )
    
    def retrieve_documents(self, question: str) -> List[Document]:
        """Retrieve relevant documents without LLM answer"""
        if not self._initialized:
            raise RuntimeError("RAG service not initialized. Call initialize() first.")
        
        return self.self_query_retriever.get_relevant_documents(question)
    
    def get_answer(self, question: str) -> dict:
        """Get LLM answer with source documents"""
        if not self._initialized:
            raise RuntimeError("RAG service not initialized. Call initialize() first.")
        
        return self.qa_chain({"query": question})
    
    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized"""
        return self._initialized
    
    @property
    def total_documents(self) -> int:
        """Get total number of documents"""
        return len(self.documents) if self.documents else 0


# Global service instance
rag_service = RAGService()
