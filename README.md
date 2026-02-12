# RAG Chatbot: OpenAI Embedding + vLLM (Gemma3)

A department-focused chatbot that answers questions about UET (University of Engineering and Technology) using Retrieval-Augmented Generation with OpenAI embeddings and vLLM for local inference.

## 🎯 Features

- **RAG Pipeline**: Retrieval-Augmented Generation for accurate answers
- **OpenAI Embeddings**: High-quality semantic search
- **vLLM Inference**: Local, high-throughput LLM inference (100+ req/s)
- **Guardrail Validation**: Ensures responses are department-related only
- **Citation Support**: All answers include source references
- **FastAPI Backend**: REST API with clean endpoints
- **Streamlit Frontend**: Simple, user-friendly chat interface

## 📋 Project Components

```
PDF (UET Prospectus)
    ↓
Chunk & Preprocess
    ↓
OpenAI Embeddings
    ↓
Vector Database (Chroma)
    ↓
FastAPI Server
    ├─ Guardrail Validation
    ├─ Vector Search
    └─ vLLM + Gemma3
    ↓
Streamlit Chat UI
```

## 👥 Team Members

| ID | Name | Role |
|---|---|---|
| 2025-MSAI-199 | **AZHAR SALEEM** | PDF reading, cleaning, and chunking |
| 2025-MSAI-102 | **SHOAIB ABID** | RAG / Retrieval logic |
| 2025-MSAI-101 | **AZEEFA TAHIR** | Backend API development (FastAPI) |
| 2024-MSAIE-18 | **SAAD ALI AMJAD** | GUI development, testing, and video |

## 🏗️ Architecture

See `/diagrams/` folder for detailed architecture diagrams:
- `architecture_diagram.svg` - System overview
- `data_flow_diagram.svg` - Processing flow
- `deployment_architecture.svg` - Infrastructure

## ⚙️ Tech Stack

- **Backend**: FastAPI, Python 3.9+
- **Embeddings**: OpenAI API (text-embedding-3-small)
- **Vector DB**: Chroma
- **LLM**: vLLM + Gemma3 (7B/13B/70B)
- **Frontend**: Streamlit
- **GPU**: NVIDIA (16GB+ VRAM)

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Setup Environment
```bash
cp .env.example .env
# Update .env with your OpenAI API key
```

### Start Services

**1. vLLM Server** (in separate terminal)
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --port 8001
```

**2. FastAPI Backend**
```bash
uvicorn backend.api:app --reload --port 8000
```

**3. Streamlit Frontend** (in another terminal)
```bash
streamlit run frontend/app.py
```

Then open http://localhost:8501

## 📁 Folder Structure

```
.
├── backend/
│   ├── api.py              # FastAPI endpoints
│   ├── embeddings.py       # OpenAI embeddings
│   ├── vector_db.py        # Vector database
│   └── guardrail.py        # Validation logic
├── frontend/
│   └── app.py              # Streamlit UI
├── data/
│   └── embeddings/         # Chroma DB storage
├── diagrams/
│   ├── architecture_diagram.svg
│   ├── data_flow_diagram.svg
│   ├── deployment_architecture.svg
│   └── guides/             # Detailed documentation
├── tests/
│   └── test_queries.json   # 20 test queries
├── requirements.txt
├── .env.example
└── README.md
```

## 📊 API Endpoints

### Chat Endpoint
```
POST /chat

Request:
{
  "message": "What programs does CS offer?"
}

Response:
{
  "answer": "The Computer Science department offers...",
  "citations": ["Page 23: BS CS", "Page 25: MS CS"],
  "confidence": 0.92,
  "processing_time": 1250
}
```

## 🧪 Testing

Test the system with 20 queries:
- 10 department-related questions ✅
- 5 tricky questions 🤔
- 5 out-of-scope questions ❌

Run tests:
```bash
# See tests/test_queries.json for test set
python tests/test_system.py
```

## 📈 Performance

| Metric | Value |
|--------|-------|
| Query Embedding | ~100ms |
| Vector Search | ~50ms |
| Response Generation | 500-2000ms |
| **Total Response** | **~1-2 seconds** |
| Throughput | 100+ req/s |

## 🔐 Validation

The chatbot automatically validates queries:
- ✅ Department-related → Answer with sources
- ❌ Non-department → "I only answer department information"

## 📝 Configuration

Edit `.env` file:
```
OPENAI_API_KEY=your_key_here
VLLM_BASE_URL=http://localhost:8001
VECTOR_DB_PATH=./data/embeddings
TOP_K=5
```

## 🎬 Video Presentation

6-10 minute video covering:
1. Team introduction & task division
2. Architecture explanation
3. Live demo (3 sample queries)
4. System requirements & deployment
5. Testing results

[Link to video](add_video_link_here)

## 📚 Documentation

For detailed documentation, see:
- `/diagrams/DIAGRAM_GUIDE.md` - Architecture details
- `/diagrams/QUICK_REFERENCE.md` - Quick reference
- `/diagrams/README_DIAGRAMS.md` - Full guide

## 🛠️ Troubleshooting

**GPU out of memory?**
- Use smaller Gemma3 model (7B instead of 13B)

**Slow responses?**
- Check vLLM is running: `curl http://localhost:8001/health`
- Increase GPU VRAM or reduce context size

**Low quality answers?**
- Increase `TOP_K` in .env
- Adjust chunking strategy
- Improve guardrail rules

## 📦 Requirements

### Hardware
- GPU: NVIDIA 16GB+ VRAM
- RAM: 32GB+ system
- Storage: 100GB+ SSD

### Software
- Python 3.9+
- CUDA 11.8+
- Docker (recommended)

## 📜 License

This is an academic project for NLP class.

## 📞 Contact

For questions, contact any team member above.

---

**Project**: NLP Class - RAG Chatbot
**Date**: 2026
**Status**: ✅ Complete
