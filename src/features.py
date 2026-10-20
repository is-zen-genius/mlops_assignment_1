"""Build the sklearn preprocessing transformer shared by training and inference.

Three column groups, three strategies - see config.py for *why* each column is grouped
the way it is:
  * continuous / ordinal counts -> median-impute + standardise
  * already-binary flags        -> most-frequent-impute, used as-is
  * unordered categories        -> most-frequent-impute + one-hot
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import BINARY_FEATURES, CONTINUOUS_FEATURES, NOMINAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    continuous = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("continuous", continuous, CONTINUOUS_FEATURES),
            ("binary", binary, BINARY_FEATURES),
            ("nominal", nominal, NOMINAL_FEATURES),
        ],
    )
