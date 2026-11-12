"""Export a registered model version to a local directory for the Docker image.

The container does not talk to the MLflow registry: the model is copied into the image
at build time, so each image is tied to exactly one model version. Registry details
(name, version, alias, run id) are written next to the model in ``registry_info.json``
so the API can still report them.

Usage:
    python -m src.serving.export_model                          # production alias -> build/model
    python -m src.serving.export_model --model-uri models:/bengaluru_rainfall_by_cell@challenger
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from src.serving.predictor import DEFAULT_MODEL_URI, DEFAULT_TRACKING_URI

REGISTRY_INFO = "registry_info.json"


def export(model_uri: str = DEFAULT_MODEL_URI, out_dir: Path = Path("build/model"),
           tracking_uri: str = DEFAULT_TRACKING_URI) -> dict:
    import mlflow
    from mlflow import MlflowClient

    if not model_uri.startswith("models:/") or "@" not in model_uri:
        raise ValueError("model_uri must look like models:/<name>@<alias>")
    mlflow.set_tracking_uri(tracking_uri)
    name, alias = model_uri.removeprefix("models:/").split("@", 1)
    mv = MlflowClient().get_model_version_by_alias(name, alias)

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}/{mv.version}", dst_path=str(out_dir))
    if not (out_dir / "MLmodel").exists():
        raise RuntimeError(f"export did not produce an MLmodel file in {out_dir}")

    info = {"registered_name": name, "version": str(mv.version), "alias": alias, "run_id": mv.run_id,
            "source_uri": model_uri, "exported_at": datetime.now(UTC).isoformat(timespec="seconds")}
    (out_dir / REGISTRY_INFO).write_text(json.dumps(info, indent=2) + "\n")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-uri", default=DEFAULT_MODEL_URI)
    parser.add_argument("--out-dir", type=Path, default=Path("build/model"))
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    args = parser.parse_args()
    info = export(args.model_uri, args.out_dir, args.tracking_uri)
    print(f"Exported {info['registered_name']} v{info['version']} ({info['alias']}) to {args.out_dir}")


if __name__ == "__main__":
    main()
