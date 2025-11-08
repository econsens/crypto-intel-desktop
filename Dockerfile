# ========= Crypto Intel Dockerfile (v1.1 series) =========

# --- Base image
FROM python:3.11-slim

# --- Version metadata (can be overridden at build time)
ARG APP_VERSION=1.1.0
LABEL org.opencontainers.image.title="crypto-mini" \
      org.opencontainers.image.description="Crypto Intel: RSS+sentiment+predictions+semantic memory" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.source="local"

# --- Environment
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_CACHE=/data/models \
    HF_HOME=/data/models \
    HF_HUB_CACHE=/data/models \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000

# --- System deps (curl for healthcheck, git for HF, ca-certs)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl git && \
    rm -rf /var/lib/apt/lists/*

# --- Workdir
WORKDIR /app

# --- Copy dependency list first (keeps Docker cache stable)
COPY requirements.txt /app/requirements.txt

# --- Torch CPU (pinned) + project deps
# NOTE: keep torch/vision/audio before the rest to speed up rebuilds
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 && \
    pip install --no-cache-dir -r /app/requirements.txt

# --- App files
# app.py is your main FastAPI app; ml_memory.py is the MiniLM/FAISS wrapper we added in Phase 2
COPY app.py /app/app.py
COPY ml_memory.py /app/ml_memory.py

# --- Data dirs (persisted via -v …:/data when you run)
# /data           -> database, predictions.json
# /data/models    -> HF/transformers model cache (FinBERT, MiniLM)
# /data/memory    -> semantic memory artifacts
RUN mkdir -p /data /data/models /data/memory

# --- Stamp image with version (handy for debugging inside the container)
RUN echo "${APP_VERSION}" > /app/VERSION

# --- Expose port
EXPOSE 8000

# --- Healthcheck hits the /health endpoint we added
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# --- Start app (single worker is fine for local use; raise workers if you like)
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

