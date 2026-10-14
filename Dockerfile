# Phase 6: container for the prediction API (src/api).
#
# The image holds code and dependencies only. The model is mounted read-only
# at run time as a bundle exported from the MLflow registry
# (python -m src.api.bundle); no data, MLflow store or credentials are copied.
#
#   docker build -t readmission-api:phase6 .
#   docker run --rm -p 8000:8000 \
#     -v "$(pwd)/model_bundle:/model-bundle:ro" readmission-api:phase6

FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MLFLOW_DISABLE_TELEMETRY=true \
    DO_NOT_TRACK=true \
    READMISSION_MODEL_BUNDLE=/model-bundle

WORKDIR /app

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt && pip check

COPY src/ src/

RUN useradd --create-home --uid 10001 api
USER api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
