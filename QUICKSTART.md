# Quick Start Guide

## Prerequisites
- Docker & Docker Compose installed
- OR: Python 3.11+ and Tesseract-OCR

## Option 1: Docker (Recommended)

### Start All Services
```bash
docker-compose up -d --build
```

### Check Status
```bash
docker-compose ps
```

### View Logs
```bash
docker-compose logs -f
```

### Access Services
- **FastAPI**: http://localhost:8000
- **FastAPI Docs (Swagger UI)**: http://localhost:8000/docs

## Option 2: Manual Setup

### 1. Start FastAPI Microservice
```bash
cd python-microservice
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn app.main:app --port 8000
```

## Quick Test

### Check Health
```bash
curl http://localhost:8000/health
```

### Process Document
```bash
curl -X POST http://localhost:8000/api/v1/ocr/extract \
  -F "file=@your_image.jpg" \
  -F "mode=precise"
```

## Stop Services
```bash
docker-compose down
```

## For Full Documentation
See [README.md](README.md) and [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)
