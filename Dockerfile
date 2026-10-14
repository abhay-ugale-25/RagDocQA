# Dockerfile:
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for ChromaDB and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to keep the image lightweight (~700MB smaller)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the CrossEncoder model into the image cache so container startup is instant
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY ./app ./app

RUN mkdir -p /app/data /app/vectorstore

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
