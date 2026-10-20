# syntax=docker/dockerfile:1

# ---- builder: resolve dependencies into a throwaway virtualenv -----------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# single project-wide requirements.txt (see README.md) - simpler to maintain than splitting
# API/dev/training dependencies into separate files, at the cost of a larger image.
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---- runtime: copy only the venv + the handful of modules the API actually imports ----
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN useradd --create-home --no-log-init appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY src/config.py src/features.py src/inference.py src/schemas.py src/metrics.py \
     src/logging_utils.py src/main.py ./src/
COPY models/model.joblib ./models/model.joblib

USER appuser
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"

CMD ["uvicorn", "main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8001"]
