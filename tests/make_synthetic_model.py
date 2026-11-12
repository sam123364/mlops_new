"""Write a small synthetic MLflow pyfunc model to a directory, in the exported layout.

CI has no MLflow registry or DVC data, so the container smoke test builds the image
around this model instead. Same bundle as the API tests (``tests/test_serving.py``).

Usage:
    python -m tests.make_synthetic_model build/model
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import mlflow.pyfunc

from src.models.cell_model_bundle import CellRainfallPyfunc
from tests.test_serving import _meta, write_bundle


def main(out_dir: str = "build/model") -> None:
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    with tempfile.TemporaryDirectory() as tmp:
        bundle = write_bundle(Path(tmp) / "B2", _meta("B2"))
        mlflow.pyfunc.save_model(str(out), python_model=CellRainfallPyfunc(), artifacts={"bundle": str(bundle)},
                                 code_paths=["src"])
    info = {"registered_name": "synthetic_ci_model", "version": "0", "alias": "ci"}
    (out / "registry_info.json").write_text(json.dumps(info) + "\n")
    print(f"Wrote synthetic model to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
