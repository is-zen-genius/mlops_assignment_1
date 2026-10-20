import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from config import FEATURE_COLUMNS, TARGET
from features import build_preprocessor
from train import compute_metrics


def test_compute_metrics_returns_all_expected_keys_in_range(trained_model, sample_df):
    metrics = compute_metrics(trained_model, sample_df[FEATURE_COLUMNS], sample_df[TARGET])
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_predict_proba_rows_sum_to_one(trained_model, sample_df):
    proba = trained_model.predict_proba(sample_df[FEATURE_COLUMNS])
    assert proba.shape == (len(sample_df), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_training_is_reproducible_given_a_fixed_random_state(sample_df):
    def fit_once():
        pipeline = Pipeline([
            ("prep", build_preprocessor()),
            ("clf", DecisionTreeClassifier(max_depth=3, random_state=0)),
        ])
        return pipeline.fit(sample_df[FEATURE_COLUMNS], sample_df[TARGET])

    X = sample_df[FEATURE_COLUMNS]
    assert np.allclose(fit_once().predict_proba(X), fit_once().predict_proba(X))
