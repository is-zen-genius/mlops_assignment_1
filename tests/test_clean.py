import numpy as np
import pandas as pd

from clean import CATEGORICAL_INT_COLUMNS, binarise_target, clean, impute_unknowns
from config import TARGET


def make_raw() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [63, 67, 41, 56, 57, 44], "sex": [1, 1, 0, 1, 0, 1],
        "cp": [1, 4, 2, 2, 4, 3], "trestbps": [145, 160, 130, 120, 140, 150],
        "chol": [233, 286, 204, 236, 354, 240], "fbs": [1, 0, 0, 0, 0, 1],
        "restecg": [2, 2, 2, 0, 0, 0], "thalach": [150, 108, 172, 178, 160, 165],
        "exang": [0, 1, 0, 0, 1, 0], "oldpeak": [2.3, 1.5, 1.4, 0.8, 0.6, 0.0],
        "slope": [3, 2, 1, 1, 1, 2], "ca": [0, 3, 0, 0, 0, 0],
        "thal": [6, 3, 3, 3, 3, 7], "num": [0, 2, 0, 0, 1, 4],
    }).astype("float64")


def test_binarise_target_maps_any_severity_to_one():
    out = binarise_target(make_raw())
    assert list(out[TARGET]) == [0, 1, 0, 0, 1, 1]
    assert "num" not in out.columns


def test_binarise_target_does_not_mutate_input():
    raw = make_raw()
    binarise_target(raw)
    assert "num" in raw.columns


def test_impute_unknowns_fills_with_mode():
    df = make_raw()
    df["ca"] = [1.0, 1.0, 1.0, 2.0, np.nan, 0.0]
    assert impute_unknowns(df)["ca"].iloc[4] == 1.0


def test_impute_unknowns_only_touches_requested_columns():
    df = make_raw()
    df["chol"] = [100, 200, 300, np.nan, 400, 500]
    out = impute_unknowns(df, columns=["ca", "thal"])
    assert out["chol"].isna().sum() == 1  # chol wasn't in the imputation list


def test_clean_leaves_no_missing_values_and_int_categoricals():
    df = make_raw()
    df.loc[0, "thal"] = np.nan
    df.loc[1, "ca"] = np.nan
    out = clean(df)
    assert out.isna().sum().sum() == 0
    assert all(out[c].dtype.kind == "i" for c in CATEGORICAL_INT_COLUMNS)


def test_clean_drops_exact_duplicates():
    df = pd.concat([make_raw(), make_raw().iloc[[0]]], ignore_index=True)
    assert len(clean(df)) == 6
