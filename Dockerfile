FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Free-tier hosts give this container a fraction of one CPU and a hard 512MB
# memory cap. Extra BLAS/torch threads are pure overhead there, and each
# glibc malloc arena (one is created per contending thread by default) holds
# onto freed heap pages, inflating RSS on a box where every MB counts.
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2

WORKDIR /app

# PyTorch comes from the CPU-only index. The default Linux wheel on PyPI pulls
# the CUDA runtime, gigabytes of wheels that this CPU-only service never
# uses. The three steps:
#   1. Install torch from the CPU index at the version requirements.txt pins
#      (the spec is read from that file, so the two cannot disagree).
#   2. Install requirements.txt with the installed torch as a constraint. The
#      constraint carries the +cpu local version, which no PyPI wheel has, so
#      this step can keep the CPU build but never replace it; a pin that no
#      longer matches fails the build instead.
#   3. Fail the build if the result is not the CPU build or any CUDA, NVIDIA
#      or Triton distribution is present.
COPY requirements.txt .
RUN pip install "$(grep -E '^torch[~=<>!]' requirements.txt | tr -d '\r')" \
        --index-url https://download.pytorch.org/whl/cpu \
 && python -c "import importlib.metadata as m; print('torch==' + m.version('torch'))" \
        > /tmp/torch-pin.txt \
 && pip install -r requirements.txt -c /tmp/torch-pin.txt \
 && python -c "import importlib.metadata as m; v = m.version('torch'); gpu = sorted(str(d.metadata['Name']) for d in m.distributions() if (d.metadata['Name'] or '').lower().startswith(('nvidia-', 'cuda-', 'triton'))); assert v.endswith('+cpu') and not gpu, (v, gpu); print('torch', v, 'CPU build, no GPU packages')" \
 && rm /tmp/torch-pin.txt

# Run as a non-root user. Some hosts (for example Hugging Face Spaces) require
# this, and it is good practice everywhere.
RUN useradd --create-home --uid 1000 appuser
COPY --chown=appuser:appuser . .
USER appuser

# Hosts inject the port through $PORT; default to 8000 when run locally.
# The stream socket takes one small JSON config (the app refuses anything over
# 1,024 bytes), so uvicorn refuses frames over 4 KiB, queues at most 4 and
# turns off per-message compression.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --ws-max-size 4096 --ws-max-queue 4 --ws-per-message-deflate false"]
