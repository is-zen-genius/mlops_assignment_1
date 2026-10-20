import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import FEATURE_COLUMNS, TARGET
from features import build_preprocessor


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """90 synthetic patients with the same schema as the real cleaned dataset."""
    rng = np.random.default_rng(7)
    n = 90
    df = pd.DataFrame({
        "age": rng.integers(29, 78, n),
        "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 410, n),
        "thalach": rng.integers(71, 202, n),
        "oldpeak": rng.random(n) * 5,
        "ca": rng.integers(0, 5, n),
        "sex": rng.integers(0, 2, n),
        "fbs": rng.integers(0, 2, n),
        "exang": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "restecg": rng.integers(0, 3, n),
        "slope": rng.integers(1, 4, n),
        "thal": rng.choice([3, 6, 7], n),
    }).astype("float64")
    df[TARGET] = ((df["oldpeak"] + df["ca"] + rng.normal(0, 1.0, n)) > 3.5).astype(int)
    return df[FEATURE_COLUMNS + [TARGET]]


@pytest.fixture
def trained_model(sample_df):
    pipeline = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    return pipeline.fit(sample_df[FEATURE_COLUMNS], sample_df[TARGET])
