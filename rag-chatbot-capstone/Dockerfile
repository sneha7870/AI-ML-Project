FROM python:3.11-slim

WORKDIR /app

# System deps needed by faiss / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the vector index at image-build time so it's baked into the deployed
# image (no need to re-ingest on every cold start). If your documents change
# often, move this to a startup script instead.
RUN python ingest.py

EXPOSE 8000

CMD gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 4 --timeout 120 app:app
