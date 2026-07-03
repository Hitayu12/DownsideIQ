# DownsideIQ — single image used by api / worker / dashboard (command differs).
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for numpy/scipy/xgboost/lightgbm wheels (libgomp for OpenMP).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Non-root runtime user.
RUN useradd -m appuser && mkdir -p /app/data /app/logs /app/models_store \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

# Default command is the API; compose overrides for worker/dashboard.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
