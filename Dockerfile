FROM python:3.11-slim-bookworm AS builder

ARG TARGETPLATFORM

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip wheel setuptools --timeout 120 --retries 5

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt --timeout 120 --retries 5

FROM python:3.11-slim-bookworm AS production

LABEL maintainer="Audit Tool Team"
LABEL description="Audit Checkpoint Generator - RAG-powered verification checkpoint generation"
LABEL version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    BACKEND_PORT=8000 \
    FRONTEND_PORT=8501

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    libxml2 \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN mkdir -p /app/data/uploads \
    /app/data/vector_db \
    /app/.cache/huggingface \
    && chown -R appuser:appgroup /app

COPY --chown=appuser:appgroup backend/ /app/backend/
COPY --chown=appuser:appgroup frontend/ /app/frontend/
COPY --chown=appuser:appgroup requirements.txt /app/

USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${BACKEND_PORT}/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
