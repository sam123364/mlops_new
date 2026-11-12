#!/usr/bin/env bash
# Export a registered model version and build the inference image for it.
#
# Usage (from the repository root):
#   scripts/docker_build.sh                                   # production alias
#   scripts/docker_build.sh models:/bengaluru_rainfall_by_cell@challenger
#
# Tags: bengaluru-rainfall-api:<model name>-v<version> and bengaluru-rainfall-api:latest
set -euo pipefail

MODEL_URI="${1:-models:/bengaluru_rainfall_by_cell@production}"
IMAGE="bengaluru-rainfall-api"
PY="${PYTHON:-.venv/bin/python}"

export MLFLOW_DISABLE_AGENT_HINT=1
"$PY" -m src.serving.export_model --model-uri "$MODEL_URI" --out-dir build/model

read -r NAME VERSION < <("$PY" -c "import json; d = json.load(open('build/model/registry_info.json')); print(d['registered_name'], d['version'])")
COMMIT="$(git rev-parse --short HEAD)"
if ! git diff --quiet HEAD -- src requirements-serving.txt Dockerfile; then
  COMMIT="${COMMIT}-dirty"
fi

docker build \
  --build-arg GIT_COMMIT="$COMMIT" \
  --build-arg MODEL_NAME="$NAME" \
  --build-arg MODEL_VERSION="$VERSION" \
  -t "$IMAGE:${NAME}-v${VERSION}" \
  -t "$IMAGE:latest" \
  .

echo "Built $IMAGE:${NAME}-v${VERSION} (also tagged latest) from $MODEL_URI at commit $COMMIT"
