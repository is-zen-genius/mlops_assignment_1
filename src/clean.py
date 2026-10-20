"""Clean the raw UCI export and write an analysis-ready CSV to data/processed/.

Steps: drop exact duplicate rows, collapse the 5-level diagnosis into a binary target,
impute the handful of missing `ca`/`thal` values, and cast categorical columns to int.
"""
import pandas as pd

from config import CLEAN_DATA_FILE, COLUMNS_WITH_UNKNOWNS, RAW_DATA_FILE, TARGET

CATEGORICAL_INT_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def load_raw(path=RAW_DATA_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def binarise_target(df: pd.DataFrame) -> pd.DataFrame:
    """`num` is 0 (no disease) .. 4 (severity); the assignment only needs presence/absence."""
    df = df.copy()
    df[TARGET] = (df["num"] > 0).astype(int)
    return df.drop(columns=["num"])


def impute_unknowns(df: pd.DataFrame, columns=COLUMNS_WITH_UNKNOWNS) -> pd.DataFrame:
    """UCI encodes missing values as '?' (already parsed to NaN); fill with the column mode."""
    df = df.copy()
    for col in columns:
        if df[col].isna().any():
            mode_value = df[col].mode(dropna=True).iloc[0]
            df[col] = df[col].fillna(mode_value)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = binarise_target(df)
    df = impute_unknowns(df)
    df[CATEGORICAL_INT_COLUMNS] = df[CATEGORICAL_INT_COLUMNS].astype(int)
    return df.reset_index(drop=True)


def main() -> None:
    raw = load_raw()
    missing_before = raw.isna().sum()
    print("Missing values before cleaning:\n", missing_before[missing_before > 0])

    cleaned = clean(raw)
    CLEAN_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN_DATA_FILE, index=False)

    print(f"Missing values after cleaning: {cleaned.isna().sum().sum()}")
    print(f"Saved {cleaned.shape[0]} rows x {cleaned.shape[1]} cols -> {CLEAN_DATA_FILE}")
    print("Class balance:\n", cleaned[TARGET].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
