"""Package the trained model: metadata + MLflow-format copy + registry + round-trip check.

Produces the three artifacts the rest of the project relies on:
  * models/model.joblib          - loaded directly by src/inference.py and the API
  * models/model_metadata.json   - metrics, params, schema, hashes, versions (for audits)
  * models/mlflow_model/         - MLflow "sklearn" flavour copy, servable via `mlflow models serve`
It also registers the winning run's model in the MLflow Model Registry under the
"champion" alias, so downstream consumers can resolve
`models:/heart-risk-classifier@champion` instead of hard-coding a run id.
"""
import hashlib
import json
import platform
import shutil
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.tracking import MlflowClient

from config import (
    BINARY_FEATURES,
    CLEAN_DATA_FILE,
    CONTINUOUS_FEATURES,
    FEATURE_COLUMNS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_DIR,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_COMPARISON_FILE,
    MODEL_FILE,
    MODEL_METADATA_FILE,
    NOMINAL_FEATURES,
    TARGET,
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def write_metadata(best_run: pd.Series, model_path: Path, data_path: Path) -> dict:
    metadata = {
        "model_family": best_run["model_family"],
        "best_params": best_run["best_params"],
        "metrics": {
            k: float(best_run[k])
            for k in best_run.index
            if k.startswith("test_") or k == "cv_roc_auc"
        },
        "features": {
            "continuous": CONTINUOUS_FEATURES,
            "binary": BINARY_FEATURES,
            "nominal": NOMINAL_FEATURES,
        },
        "target": TARGET,
        "trained_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "training_data_sha256": file_sha256(data_path),
        "model_sha256": file_sha256(model_path),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "mlflow_version": mlflow.__version__,
    }
    MODEL_METADATA_FILE.write_text(json.dumps(metadata, indent=2))
    return metadata


def export_mlflow_format(model, example: pd.DataFrame) -> None:
    if MLFLOW_MODEL_DIR.exists():
        shutil.rmtree(MLFLOW_MODEL_DIR)
    mlflow.sklearn.save_model(
        model,
        str(MLFLOW_MODEL_DIR),
        serialization_format="cloudpickle",
        input_example=example,
    )


def verify_round_trip(model_path: Path, X: pd.DataFrame) -> None:
    from_joblib = joblib.load(model_path).predict_proba(X)
    from_mlflow = mlflow.sklearn.load_model(str(MLFLOW_MODEL_DIR)).predict_proba(X)
    if not np.allclose(from_joblib, from_mlflow):
        raise AssertionError("joblib and MLflow copies of the model disagree after reload")


def register_best_run(alias: str = "champion") -> None:
    """Best-effort: find the run tagged `selected_as_best` and register+alias it.

    Registry access needs the same tracking store train.py wrote to; if that store
    is unavailable (e.g. packaging is re-run standalone without tracking.db) we log a
    warning instead of failing the whole packaging step, since the joblib/metadata/
    mlflow_model artifacts above are the primary deliverable.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            raise RuntimeError(f"No MLflow experiment named {MLFLOW_EXPERIMENT_NAME!r}")

        runs = client.search_runs(
            experiment.experiment_id,
            filter_string="tags.selected_as_best = 'true'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise RuntimeError("No run tagged selected_as_best=true found")

        run_id = runs[0].info.run_id
        model_uri = f"runs:/{run_id}/model"
        registered = mlflow.register_model(model_uri, MLFLOW_REGISTERED_MODEL_NAME)
        client.set_registered_model_alias(MLFLOW_REGISTERED_MODEL_NAME, alias, registered.version)
        print(
            f"Registered {MLFLOW_REGISTERED_MODEL_NAME} v{registered.version} "
            f"(alias '{alias}') from run {run_id}"
        )
    except Exception as exc:  # noqa: BLE001 - registry is a bonus step, never fail packaging for it
        print(f"Skipping MLflow registry step: {exc}")


def main() -> None:
    model = joblib.load(MODEL_FILE)
    best_run = pd.read_csv(MODEL_COMPARISON_FILE).iloc[0]  # already sorted by cv_roc_auc
    X = pd.read_csv(CLEAN_DATA_FILE)[FEATURE_COLUMNS].astype("float64")

    write_metadata(best_run, MODEL_FILE, CLEAN_DATA_FILE)
    print(f"Wrote {MODEL_METADATA_FILE.relative_to(MODEL_METADATA_FILE.parents[1])}")

    export_mlflow_format(model, X.head(2))
    print(f"Wrote {MLFLOW_MODEL_DIR.relative_to(MLFLOW_MODEL_DIR.parents[1])}/")

    verify_round_trip(MODEL_FILE, X)
    print("Round-trip check passed: joblib and MLflow copies give identical predictions.")

    register_best_run()


if __name__ == "__main__":
    main()
