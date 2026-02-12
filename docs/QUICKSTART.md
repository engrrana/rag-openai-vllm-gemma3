# Quick Start Guide

## 1. Setup (One-time)
```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file
copy .env.example .env

# Edit .env and add your OpenAI API key
notepad .env
```

## 2. Run Server
```bash
python run.py
```

Server will start at: http://localhost:8000

## 3. Test Endpoints

### Basic Tests
```bash
python test_api.py
```

### Quality Tests (20 Questions)
```bash
python test_quality.py
```

## 4. API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Retrieve Documents
```bash
curl -X POST http://localhost:8000/api/v1/retrieve ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"What programs does CS offer?\"}"
```

### Get Answer
```bash
curl -X POST http://localhost:8000/api/v1/answer ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"Who is the chairman of EE?\"}"
```

## 5. View API Docs
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure
```
app/
├── api/         # Endpoints & schemas
├── core/        # Configuration
├── services/    # Business logic
└── utils/       # Utilities
```

## Common Commands
```bash
# Start server
python run.py

# Run basic tests
python test_api.py

# Run quality tests
python test_quality.py

# Install dependencies
pip install -r requirements.txt
```
