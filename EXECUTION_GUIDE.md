# Complete Execution & Usage Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Method 1: Docker Compose (Recommended)](#method-1-docker-compose-recommended)
4. [Method 2: Manual Execution](#method-2-manual-execution)
5. [Verification Steps](#verification-steps)
6. [API Usage Tutorial](#api-usage-tutorial)
7. [Running the UI (Streamlit)](#running-the-ui-streamlit)

---

## Prerequisites

### Required Software

#### For Docker Method (Recommended)
- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher

#### For Manual Method
- **Python**: Version 3.11 or higher
- System packages (Windows/Linux): Tesseract-OCR

### Required AI Models
Ensure this file is in your project directory:
- `yolov8n.pt` - YOLO model weights

Download YOLO model if missing:
```bash
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -o yolov8n.pt
```

---

## Project Structure

The project has been radically simplified into a stateless Python microservice. All Java/Spring Boot dependencies have been permanently removed.

```
c:\D\ai proj
├── python-microservice/      (FastAPI service & Core AI logic)
├── src/                      (Streamlit UI frontend)
├── yolov8n.pt                (YOLO Weights)
├── docker-compose.yml        (Local Deployment)
└── docker-compose.prod.yml   (Production Deployment)
```

Configuration is handled dynamically via Pydantic `BaseSettings` (`python-microservice/app/core/config.py`). There are no rigid `config.json` files. Thresholds and limits can be tuned using Docker Environment variables.

---

## Method 1: Docker Compose (Recommended)


### Step 1: Start the Service
```bash
docker-compose up -d --build
```
This builds and starts the FastAPI microservice on port `8000`.

### Step 2: Check Logs
```bash
docker-compose logs -f
```
Wait for: `Application startup complete`.

---

## Method 2: Manual Execution

### Step 1: Start FastAPI Microservice
```bash
cd python-microservice
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Wait for: `Application startup complete`.

---

## Verification Steps

### Check Health Endpoints
```bash
curl http://localhost:8000/health
```
**Expected response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": true
}
```

### Access API Documentation
Open your browser to: http://localhost:8000/docs
This provides the interactive Swagger UI to test all endpoints.

---

## API Usage Tutorial

### 1. Preprocess Image
```bash
curl -X POST http://localhost:8000/api/v1/preprocess \
  -F "file=@image.jpg" \
  -F "include_images=false"
```

### 2. OCR Extraction
```bash
curl -X POST http://localhost:8000/api/v1/ocr/extract \
  -F "file=@image.jpg" \
  -F "mode=precise"
```

### 3. NLP Analysis
```bash
curl -X POST http://localhost:8000/api/v1/nlp/analyze \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Your text to analyze here...\"}"
```

### 4. Object Detection
```bash
curl -X POST http://localhost:8000/api/v1/detect/objects \
  -F "file=@image.jpg" \
  -F "annotate=true" \
  -F "include_images=true"
```

---

## Running the UI (Streamlit)

A legacy Streamlit frontend is provided for easy testing.

1. Ensure Python dependencies are installed globally or in a local virtual environment (from `python-microservice/requirements.txt` and `requirements.txt`).
2. Start the FastAPI microservice in the background.
3. Run the Streamlit app:
```bash
streamlit run src/app.py
```
This will open the Interactive UI on `http://localhost:8501`.
