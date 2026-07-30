# --- Build a lean, production-ready Flask image ---
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching on rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code + pre-trained model artifacts
COPY . .

# Render provides $PORT at runtime; gunicorn binds to it
EXPOSE 8000

CMD gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 2 --timeout 120 app:app
