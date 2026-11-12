# Inference container for the Bengaluru rainfall API (src/serving).
#
# The model is baked into the image: build/model is an exported registry version
# (python -m src.serving.export_model), so one image = one model version.
# Build with scripts/docker_build.sh, which exports the model and sets the labels.

FROM python:3.12.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so code and model changes reuse this layer.
COPY requirements-serving.txt .
RUN pip install -r requirements-serving.txt

# Only the code the API imports (training code stays out of the image).
COPY src/__init__.py src/__init__.py
COPY src/models/__init__.py src/models/main_model.py src/models/cell_model_bundle.py src/models/
COPY src/serving/ src/serving/
COPY build/model/ model/

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
USER app

ARG GIT_COMMIT=unknown
ARG MODEL_NAME=unknown
ARG MODEL_VERSION=unknown
LABEL org.opencontainers.image.title="bengaluru-rainfall-api" \
      org.opencontainers.image.description="Monthly rainfall quantiles per IMD grid cell for Bengaluru" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      ml.model.name="${MODEL_NAME}" \
      ml.model.version="${MODEL_VERSION}"

ENV RAINFALL_MODEL_URI=/app/model \
    HOME=/tmp

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
