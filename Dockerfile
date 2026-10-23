# Phishing detection API (src/api.py). Runtime data (blocklist, Tranco, caches, injection
# patterns, monitoring log) lives in ./data, mounted as a volume; API keys come from ./.env
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# OpenMP runtime: LightGBM (imported by Layer C inference) loads libgomp at import time
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# CPU build of torch (inference only, much smaller than the CUDA build), then the rest
COPY requirements.txt requirements-api.txt ./
RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-api.txt

COPY src/ src/
COPY models/url_classifier/ models/url_classifier/
COPY dataset/brands/ dataset/brands/

# Non-root user (uid 1000: the mounted ./data must be writable by it, e.g. `chown -R 1000 data`)
RUN useradd --uid 1000 --create-home app
USER app

EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=10s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
