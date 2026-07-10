FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install the CPU-only build of PyTorch first. The default Linux wheel bundles
# CUDA and is roughly 2.5 GB; the CPU wheel is a few hundred MB, which fits on
# free hosting tiers and starts much faster. The requirements install below
# then sees torch already satisfied and does not pull the CUDA build.
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# Run as a non-root user. Some hosts (for example Hugging Face Spaces) require
# this, and it is good practice everywhere.
RUN useradd --create-home --uid 1000 appuser
COPY --chown=appuser:appuser . .
USER appuser

# Hosts inject the port through $PORT; default to 8000 when run locally.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
