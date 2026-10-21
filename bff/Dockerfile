# BFF image (bff/app.py): small, public-facing; holds API_KEY in its environment only
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .

RUN useradd --uid 1000 --create-home app
USER app

EXPOSE 8080
HEALTHCHECK --interval=60s --timeout=10s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/health')"
# --proxy-headers: client IP from the reverse proxy (trusted per FORWARDED_ALLOW_IPS)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2", "--proxy-headers"]
