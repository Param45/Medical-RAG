# Medical Records RAG — Production Backend Dockerfile (Azure Container Deployment)
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install Linux system dependencies (Poppler for PDF rendering, build tools for local LLM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    curl \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python backend dependencies
COPY requirements-backend.txt .

# Install dependencies, fetching pre-built CPU wheels for llama-cpp-python to accelerate build
RUN pip install --no-cache-dir -U pip wheel && \
    pip install --no-cache-dir -r requirements-backend.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Copy application source code and pre-indexed data structures
COPY . .

# Ensure data directories exist with proper write permissions for ephemeral processing
RUN mkdir -p /app/data/raw /app/data/pages /app/data/ocr /app/data/reports /app/data/evidence /app/data/pageindex /app/models

# Expose backend service port
EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
