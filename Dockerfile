# ============================================================================
# Medical VQA - Dockerfile
# Knowledge-Guided Explainable Transformer for Medical Visual QA
# ============================================================================

FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python alias
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/checkpoints /app/uploads /app/results /app/logs

# Default model path
ENV MODEL_PATH=/app/checkpoints/best_model
ENV DEVICE=cuda

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
