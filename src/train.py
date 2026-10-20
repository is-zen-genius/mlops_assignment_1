"""Train, tune and compare three classifiers with full MLflow experiment tracking.

For each model family: grid-search over a small hyper-parameter grid with stratified
cross-validation, log every candidate's params/score as a nested run, then log the
best candidate's full metrics, plots and fitted pipeline on the parent run. The model
with the best cross-validated ROC-AUC across all three families is saved to
models/model.joblib for packaging (see package_model.py).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from config import (
    CLEAN_DATA_FILE,
    CV_FOLDS,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_COMPARISON_FILE,
    MODEL_DIR,
    MODEL_FILE,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
    TEST_SIZE,
)
from features import build_preprocessor

CANDIDATE_MODELS = {
    "logistic_regression": (
        # penalty is left at its sklearn default (l2); only strength and class balancing are tuned
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__class_weight": [None, "balanced"],
        },
    ),
    "random_forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_leaf": [1, 3],
        },
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.01, 0.1],
            "clf__max_depth": [2, 3],
        },
    ),
}


def load_training_data(path: Path = CLEAN_DATA_FILE):
    df = pd.read_csv(path)
    # float64 throughout (not int) so MLflow's inferred schema tolerates missing values at inference
    return df[FEATURE_COLUMNS].astype("float64"), df[TARGET]


def compute_metrics(estimator, X, y) -> dict:
    y_pred = estimator.predict(X)
    y_proba = estimator.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
    }


def save_diagnostic_plots(model_name: str, estimator, X_test, y_test) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    ConfusionMatrixDisplay.from_estimator(estimator, X_test, y_test, cmap="Blues")
    plt.title(f"Confusion matrix - {model_name}")
    cm_path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(cm_path)

    RocCurveDisplay.from_estimator(estimator, X_test, y_test)
    plt.title(f"ROC curve - {model_name}")
    roc_path = FIGURES_DIR / f"roc_curve_{model_name}.png"
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(roc_path)

    PrecisionRecallDisplay.from_estimator(estimator, X_test, y_test)
    plt.title(f"Precision-recall curve - {model_name}")
    pr_path = FIGURES_DIR / f"precision_recall_{model_name}.png"
    plt.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(pr_path)

    return paths


def log_top_candidates(search: GridSearchCV, model_name: str, top_n: int = 5) -> None:
    """Log the top-N grid-search candidates as nested child runs (full grid as a CSV artifact)."""
    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    for _, row in results.head(top_n).iterrows():
        with mlflow.start_run(run_name=f"candidate_rank{int(row['rank_test_score'])}", nested=True):
            mlflow.log_params({k.replace("clf__", ""): v for k, v in row["params"].items()})
            mlflow.log_metric("cv_roc_auc_mean", row["mean_test_score"])
            mlflow.log_metric("cv_roc_auc_std", row["std_test_score"])

    results_path = REPORTS_DIR / "cv_results" / f"cv_results_{model_name}.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    mlflow.log_artifact(str(results_path), artifact_path="cv_results")


def train_one_family(name, estimator, param_grid, X_train, y_train, X_test, y_test, cv):
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"model_family": name, "stage": "tuning"})
        mlflow.log_params({
            "model_family": name,
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "param_grid": json.dumps(param_grid),
        })

        pipeline = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
        search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)

        log_top_candidates(search, model_name=name)

        mlflow.log_params(
            {f"best_{k.replace('clf__', '')}": v for k, v in search.best_params_.items()}
        )
        test_metrics = compute_metrics(search.best_estimator_, X_test, y_test)
        mlflow.log_metric("cv_roc_auc", search.best_score_)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        for plot_path in save_diagnostic_plots(name, search.best_estimator_, X_test, y_test):
            mlflow.log_artifact(str(plot_path), artifact_path="plots")

        mlflow.sklearn.log_model(
            search.best_estimator_,
            artifact_path="model",
            input_example=X_test.head(2),
            serialization_format="cloudpickle",
        )

        summary = {
            "model_family": name,
            "best_params": search.best_params_,
            "cv_roc_auc": round(search.best_score_, 4),
            **{f"test_{k}": round(v, 4) for k, v in test_metrics.items()},
        }
        print(f"\n[{name}] cv_roc_auc={search.best_score_:.4f}  test={test_metrics}")
        return summary, search.best_estimator_, run.info.run_id


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X, y = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    summaries, fitted_models, run_ids = [], {}, {}
    for name, (estimator, grid) in CANDIDATE_MODELS.items():
        summary, fitted, run_id = train_one_family(
            name, estimator, grid, X_train, y_train, X_test, y_test, cv
        )
        summaries.append(summary)
        fitted_models[name] = fitted
        run_ids[name] = run_id

    comparison = pd.DataFrame(summaries).sort_values("cv_roc_auc", ascending=False)
    MODEL_COMPARISON_FILE.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(MODEL_COMPARISON_FILE, index=False)
    print("\nModel comparison (best candidate per family):\n", comparison.to_string(index=False))

    winner_name = comparison.iloc[0]["model_family"]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(fitted_models[winner_name], MODEL_FILE)

    with mlflow.start_run(run_id=run_ids[winner_name]):
        mlflow.set_tag("selected_as_best", "true")
        mlflow.log_artifact(str(MODEL_COMPARISON_FILE))

    print(f"\nSelected model: {winner_name} -> {MODEL_FILE} (run {run_ids[winner_name]})")


if __name__ == "__main__":
    main()
