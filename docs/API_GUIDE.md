# FastAPI RAG System

A production-ready REST API for the OpenAI RAG system with two endpoints.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

**Windows:**
```powershell
$env:OPENAI_API_KEY="sk-your-key-here"
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

### 3. Run the Server

```bash
python api.py
```

Or with uvicorn directly:
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: **http://localhost:8000**

## 📚 API Documentation

Once running, visit:
- **Interactive docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

## 🔌 Endpoints

### 1. `/retrieve` - Document Retrieval Only

Get relevant documents without LLM answer generation.

**Use case**: When you only need source documents for your own processing.

**Request:**
```bash
curl -X POST "http://localhost:8000/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What programs does Computer Science offer?",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "question": "What programs does Computer Science offer?",
  "documents": [
    {
      "content": "Ph.D. Computer Science\nM.Sc. Computer Science\n...",
      "metadata": {
        "department_name": "Department of Computer Science",
        "section": "Offered Programs",
        "section_type": "programs",
        "full_context": "Department of Computer Science - Offered Programs"
      }
    }
  ],
  "count": 5
}
```

### 2. `/ask` - Complete LLM Answer

Get a natural language answer using GPT-4o-mini with retrieved context.

**Use case**: When you need a human-readable answer to a question.

**Request:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What programs does Computer Science offer?",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "question": "What programs does Computer Science offer?",
  "answer": "The Department of Computer Science offers the following programs: Ph.D. in Computer Science, M.Sc. in Computer Science, and M.Sc. in Artificial Intelligence.",
  "sources": [
    {
      "content": "Ph.D. Computer Science\nM.Sc. Computer Science\n...",
      "department": "Department of Computer Science",
      "section": "Offered Programs",
      "section_type": "programs",
      "full_context": "Department of Computer Science - Offered Programs"
    }
  ],
  "document_count": 5
}
```

### 3. `/` - Root / Health Check

```bash
curl http://localhost:8000/
```

### 4. `/health` - Detailed Health Check

```bash
curl http://localhost:8000/health
```

## 🐍 Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Example 1: Get documents only
response = requests.post(
    f"{BASE_URL}/retrieve",
    json={
        "question": "Who is the chairman of Electrical Engineering?",
        "top_k": 3
    }
)
result = response.json()
print(f"Found {result['count']} documents")
for doc in result['documents']:
    print(f"- {doc['metadata']['full_context']}")

# Example 2: Get complete answer
response = requests.post(
    f"{BASE_URL}/ask",
    json={
        "question": "What are the eligibility requirements for M.Sc. Data Science?",
        "top_k": 5
    }
)
result = response.json()
print(f"Q: {result['question']}")
print(f"A: {result['answer']}")
print(f"Sources: {result['document_count']} documents")
```

## 📊 Request Parameters

Both endpoints accept:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `question` | string | Yes | - | The question to ask |
| `top_k` | integer | No | 5 | Number of documents to retrieve |

## 🔧 Configuration

Edit these variables in `api.py`:

```python
DATA_FILE = "d:/nlp/rag_optimized_data.txt"
CHROMA_DB_DIR = "d:/nlp/chroma_db_openai"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 5
```

## 💰 Cost Estimates

**Per request:**
- `/retrieve`: ~$0.00001 (embeddings only)
- `/ask`: ~$0.00008 (embeddings + LLM)

**1,000 requests:**
- `/retrieve`: ~$0.01
- `/ask`: ~$0.10

## 🚀 Production Deployment

### Using Gunicorn (Linux/Mac)

```bash
pip install gunicorn
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OPENAI_API_KEY=""

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t rag-api .
docker run -p 8000:8000 -e OPENAI_API_KEY="sk-..." rag-api
```

## 🧪 Testing

```bash
# Health check
curl http://localhost:8000/health

# Test retrieval
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Computer Science?"}'

# Test QA
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What programs are offered?"}'
```

## 📝 Notes

- The vector store is loaded once at startup for fast responses
- SelfQueryRetriever automatically applies metadata filters based on the question
- All responses include source attribution for transparency
- The API uses async endpoints for better concurrency

## 🛠️ Troubleshooting

**Error: "OPENAI_API_KEY not set"**
- Make sure you've set the environment variable before starting the server

**Error: "No module named 'langchain.retrievers'"**
- Run `pip install -r requirements.txt` again
- Make sure you're using Python 3.9+

**Slow first request:**
- The first request loads the vector store, which takes ~5 seconds
- Subsequent requests are fast (~1-2 seconds)
