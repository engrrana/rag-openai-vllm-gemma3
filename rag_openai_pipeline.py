"""
State-of-the-Art RAG System with OpenAI
========================================
Production-ready RAG system using:
- OpenAI embeddings (text-embedding-3-small)
- OpenAI GPT-4 for LLM-based auto-filtering
- SelfQueryRetriever for intelligent metadata filtering
- ChromaDB vector store
- Hybrid search (semantic + BM25)
"""

import os
import re
import time
from typing import List, Dict
from pathlib import Path

# LangChain
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Utilities
import numpy as np
from collections import Counter
import json

print("✅ Libraries imported successfully!")


# ==================== CONFIGURATION ====================

# OpenAI API Key (set this in your environment)
# Option 1: Set environment variable: export OPENAI_API_KEY='your-key'
# Option 2: Set here (not recommended for production)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Get from environment
if not OPENAI_API_KEY:
    print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
    print("   Set it with: export OPENAI_API_KEY='your-key-here'")
    # Uncomment to set directly (NOT recommended):
    # OPENAI_API_KEY = "sk-..."
    # os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# File paths
PROJECT_ROOT = Path(__file__).parent
DATA_FILE = str(PROJECT_ROOT / "data" / "processed" / "rag_optimized_data.txt")
CHROMA_DB_DIR = str(PROJECT_ROOT / "chroma_db_openai")
config_file = str(PROJECT_ROOT / "rag_config_openai.json")

# OpenAI Models
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dims, $0.02/1M tokens
# Alternative: "text-embedding-3-large" (3072 dims, better accuracy, $0.13/1M tokens)

LLM_MODEL = "gpt-4o-mini"  # Fast, cheap, smart
# Alternative: "gpt-4o" (more powerful, more expensive)

# Retrieval settings
TOP_K = 5
USE_HYBRID_SEARCH = True
USE_AUTO_FILTERING = True  # LLM-based metadata filtering

print(f"Configuration:")
print(f"  Data file: {DATA_FILE}")
print(f"  Embedding model: {EMBEDDING_MODEL}")
print(f"  LLM model: {LLM_MODEL}")
print(f"  Top-K retrieval: {TOP_K}")
print(f"  Hybrid search: {USE_HYBRID_SEARCH}")
print(f"  Auto-filtering: {USE_AUTO_FILTERING}")


# ==================== DATA LOADING & PARSING ====================

class SemanticChunker:
    """
    Parse optimized data into semantic chunks with rich metadata.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.documents = []
    
    def load_and_parse(self) -> List[Document]:
        """Parse the structured data into semantic chunks"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by department
        departments = re.split(r'=== DEPARTMENT \d+ ===', content)[1:]
        
        for dept_content in departments:
            self._parse_department(dept_content)
        
        print(f"✅ Parsed {len(self.documents)} semantic chunks")
        return self.documents
    
    def _parse_department(self, dept_content: str):
        """Parse a single department into chunks"""
        lines = dept_content.strip().split('\n')
        
        # Extract department metadata
        dept_name = None
        dept_number = None
        
        for line in lines[:5]:
            if line.startswith('Department Name:'):
                dept_name = line.replace('Department Name:', '').strip()
            elif line.startswith('Department Number:'):
                dept_number = line.replace('Department Number:', '').strip()
        
        if not dept_name:
            return
        
        # Split by sections
        sections = re.split(r'--- SECTION: (.+?) ---', dept_content)
        
        # Process section pairs
        for i in range(1, len(sections), 2):
            if i+1 >= len(sections):
                break
            
            section_name = sections[i].strip()
            section_content = sections[i+1].strip()
            
            if not section_content:
                continue
            
            # Create metadata
            metadata = {
                'department_number': dept_number,
                'department_name': dept_name,
                'section': section_name,
                'full_context': f"{dept_name} - {section_name}",
                'source': 'university_prospectus'
            }
            
            # Add section type for filtering
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


# Load and parse data
print("\n" + "="*60)
print("LOADING AND PARSING DATA")
print("="*60)

chunker = SemanticChunker(DATA_FILE)
documents = chunker.load_and_parse()

# Display statistics
print(f"\n📊 Chunk Statistics:")
print(f"  Total chunks: {len(documents)}")
print(f"  Avg chunk size: {np.mean([len(doc.page_content) for doc in documents]):.0f} chars")
print(f"  Min chunk size: {min([len(doc.page_content) for doc in documents])} chars")
print(f"  Max chunk size: {max([len(doc.page_content) for doc in documents])} chars")

# Show section distribution
section_types = [doc.metadata.get('section_type', 'other') for doc in documents]
section_counts = Counter(section_types)
print(f"\n📋 Section Distribution:")
for section, count in section_counts.items():
    print(f"  {section}: {count}")


# ==================== INITIALIZE OPENAI EMBEDDINGS ====================

print("\n" + "="*60)
print("INITIALIZING OPENAI EMBEDDINGS")
print("="*60)

print(f"Loading OpenAI embedding model: {EMBEDDING_MODEL}")
print("This uses the OpenAI API...\n")

start_time = time.time()

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY
)

load_time = time.time() - start_time
print(f"✅ Embedding model initialized in {load_time:.2f}s")

# Test embedding
test_text = "What programs does Computer Science offer?"
test_embedding = embeddings.embed_query(test_text)
print(f"  Embedding dimension: {len(test_embedding)}")
print(f"  Sample values: {test_embedding[:5]}")

print(f"\n💰 Cost Estimate:")
total_chars = sum(len(doc.page_content) for doc in documents)
total_tokens = total_chars / 4  # Rough estimate: 1 token ≈ 4 chars
cost = (total_tokens / 1_000_000) * 0.02  # $0.02 per 1M tokens
print(f"  Total tokens (estimated): {total_tokens:,.0f}")
print(f"  Embedding cost: ${cost:.4f}")


# ==================== BUILD VECTOR STORE ====================

print("\n" + "="*60)
print("BUILDING VECTOR STORE")
print("="*60)

force_rebuild = False  # Set to True to rebuild from scratch

if os.path.exists(CHROMA_DB_DIR) and not force_rebuild:
    print("Loading existing vector store...")
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name="university_prospectus_openai"
    )
    print(f"✅ Loaded {vectorstore._collection.count()} vectors from disk")
else:
    print("Creating new vector store from documents...")
    print("⚠️  This will make API calls to OpenAI for embeddings...")
    start_time = time.time()
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name="university_prospectus_openai"
    )
    
    build_time = time.time() - start_time
    print(f"✅ Vector store created with {len(documents)} vectors in {build_time:.2f}s")
    print(f"  Persisted to: {CHROMA_DB_DIR}")

collection_count = vectorstore._collection.count()
print(f"\n📊 Vector Store Info:")
print(f"  Collection name: university_prospectus_openai")
print(f"  Total vectors: {collection_count}")
print(f"  Embedding model: {EMBEDDING_MODEL}")


# ==================== SETUP RETRIEVERS ====================

print("\n" + "="*60)
print("SETTING UP RETRIEVERS")
print("="*60)

# Semantic retriever
semantic_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K}
)
print(f"✅ Semantic retriever created (top-{TOP_K})")

# BM25 retriever
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = TOP_K
print(f"✅ BM25 retriever created (top-{TOP_K})")

# Hybrid retriever
if USE_HYBRID_SEARCH:
    hybrid_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    print(f"✅ Hybrid retriever created (semantic + BM25)")
    base_retriever = hybrid_retriever
else:
    base_retriever = semantic_retriever


# ==================== SELF-QUERY RETRIEVER (LLM-BASED AUTO-FILTERING) ====================

if USE_AUTO_FILTERING:
    print("\n" + "="*60)
    print("SETTING UP LLM-BASED AUTO-FILTERING")
    print("="*60)
    
    # Initialize OpenAI LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,  # Deterministic for filtering
        openai_api_key=OPENAI_API_KEY
    )
    print(f"✅ LLM initialized: {LLM_MODEL}")
    
    # Define metadata schema for the LLM
    metadata_field_info = [
        AttributeInfo(
            name="department_name",
            description="The full name of the academic department or institute (e.g., 'Department of Computer Science', 'Institute of Data Science')",
            type="string"
        ),
        AttributeInfo(
            name="department_number",
            description="The numeric identifier of the department (1-27)",
            type="string"
        ),
        AttributeInfo(
            name="section_type",
            description=(
                "The type of information in this section. Options are: "
                "'introduction' for department history and overview, "
                "'programs' for offered degree programs (Ph.D., M.Sc., etc.), "
                "'eligibility' for admission requirements and criteria, "
                "'faculty' for professors, chairpersons, and staff information"
            ),
            type="string"
        ),
        AttributeInfo(
            name="section",
            description="The full section name (e.g., 'Offered Programs', 'Faculty Members', 'Eligibility Criteria')",
            type="string"
        ),
    ]
    
    # Create SelfQueryRetriever
    self_query_retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents="University prospectus containing information about academic departments, programs, eligibility criteria, and faculty members",
        metadata_field_info=metadata_field_info,
        verbose=True,  # Show what the LLM is doing
        enable_limit=True
    )
    
    print(f"✅ SelfQueryRetriever created with auto-filtering")
    print(f"   The LLM will automatically detect and apply metadata filters!")
    
    # Set as default retriever
    retriever = self_query_retriever
else:
    retriever = base_retriever

print(f"\n🎯 Active retriever: {'SelfQuery (LLM Auto-Filter)' if USE_AUTO_FILTERING else 'Hybrid' if USE_HYBRID_SEARCH else 'Semantic'}")


# ==================== QUERY TESTING ====================

print("\n" + "="*60)
print("TESTING QUERIES")
print("="*60)

def test_query(query: str, retriever_to_use=None, show_results: int = 3):
    """Test a query and display results"""
    if retriever_to_use is None:
        retriever_to_use = retriever
    
    print("\n" + "="*80)
    print(f"❓ QUERY: {query}")
    print("="*80)
    
    start_time = time.time()
    results = retriever_to_use.get_relevant_documents(query)
    query_time = time.time() - start_time
    
    print(f"\n⏱️ Query time: {query_time*1000:.2f}ms")
    print(f"📊 Retrieved {len(results)} documents\n")
    
    for i, doc in enumerate(results[:show_results], 1):
        print(f"📄 Result {i}:")
        print(f"   Department: {doc.metadata.get('department_name', 'N/A')}")
        print(f"   Section: {doc.metadata.get('section', 'N/A')}")
        print(f"   Section Type: {doc.metadata.get('section_type', 'N/A')}")
        print(f"   Content Preview: {doc.page_content[:200].strip()}...")
        print()
    
    return results


# Test queries
test_queries = [
    "What programs does the Department of Computer Science offer?",
    "Who is the chairman of Electrical Engineering?",
    "Eligibility criteria for M.Sc. Artificial Intelligence",
    "Tell me about the Data Science institute",
    "Which departments offer Ph.D. programs in engineering?",
]

print("\nRunning test queries with LLM-based auto-filtering...\n")

for query in test_queries[:3]:  # Test first 3
    results = test_query(query)
    time.sleep(0.5)  # Small delay between queries


# ==================== SETUP QA CHAIN ====================

print("\n" + "="*60)
print("SETTING UP QA CHAIN")
print("="*60)

# Custom prompt for better answers
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

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT}
)

print("✅ QA chain created with OpenAI GPT-4")


# ==================== END-TO-END QA TESTING ====================

print("\n" + "="*60)
print("END-TO-END QA TESTING")
print("="*60)

def ask_question(question: str):
    """Ask a question and get a complete answer"""
    print("\n" + "="*80)
    print(f"❓ QUESTION: {question}")
    print("="*80)
    
    start_time = time.time()
    result = qa_chain({"query": question})
    query_time = time.time() - start_time
    
    print(f"\n💡 ANSWER:")
    print(result['result'])
    
    print(f"\n📚 SOURCES ({len(result['source_documents'])} documents):")
    for i, doc in enumerate(result['source_documents'][:3], 1):
        print(f"  {i}. {doc.metadata.get('full_context', 'N/A')}")
    
    print(f"\n⏱️ Total time: {query_time:.2f}s")
    
    return result


# Test complete QA
qa_test_questions = [
    "What programs does Computer Science offer?",
    "Who is the chairman of the Electrical Engineering department?",
    "What are the eligibility requirements for M.Sc. in Artificial Intelligence?",
]

print("\nTesting complete QA pipeline...\n")

for question in qa_test_questions[:2]:  # Test first 2
    ask_question(question)
    time.sleep(1)


# ==================== SAVE CONFIGURATION ====================

print("\n" + "="*60)
print("SAVING CONFIGURATION")
print("="*60)

config_data = {
    "embedding_model": EMBEDDING_MODEL,
    "llm_model": LLM_MODEL,
    "total_chunks": len(documents),
    "vector_db": "ChromaDB",
    "retrieval_strategy": "SelfQuery (LLM Auto-Filter)" if USE_AUTO_FILTERING else "Hybrid" if USE_HYBRID_SEARCH else "Semantic",
    "top_k": TOP_K,
    "chunk_stats": {
        "avg_size": int(np.mean([len(doc.page_content) for doc in documents])),
        "min_size": min([len(doc.page_content) for doc in documents]),
        "max_size": max([len(doc.page_content) for doc in documents]),
    },
    "metadata_fields": ["department_name", "department_number", "section_type", "section"],
}

config_file = "d:/nlp/rag_config_openai.json"
with open(config_file, 'w') as f:
    json.dump(config_data, f, indent=2)

print(f"✅ Configuration saved to: {config_file}")
print(f"\n📊 Summary:")
print(f"  Embedding Model: {config_data['embedding_model']}")
print(f"  LLM Model: {config_data['llm_model']}")
print(f"  Total Chunks: {config_data['total_chunks']}")
print(f"  Retrieval Strategy: {config_data['retrieval_strategy']}")
print(f"  Vector DB: {config_data['vector_db']}")


# ==================== INTERACTIVE MODE ====================

print("\n" + "="*60)
print("READY FOR INTERACTIVE USE")
print("="*60)

print("""
Your RAG system is ready! You can now:

1. Ask questions:
   result = ask_question("Your question here")

2. Test retrieval only:
   docs = test_query("Your query here")

3. Query with custom retriever:
   docs = semantic_retriever.get_relevant_documents("query")
   docs = self_query_retriever.get_relevant_documents("query")

4. Access the vector store directly:
   docs = vectorstore.similarity_search("query", k=5)
   docs = vectorstore.similarity_search("query", k=5, filter={"section_type": "faculty"})

Example questions to try:
- "What programs does Computer Science offer?"
- "Who is the chairman of Electrical Engineering?"
- "What are the eligibility requirements for M.Sc. Data Science?"
- "Tell me about the Department of Mechanical Engineering"
""")

print("\n✅ System ready! Start asking questions.")
