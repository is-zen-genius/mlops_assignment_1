import numpy as np

from config import BINARY_FEATURES, CONTINUOUS_FEATURES, FEATURE_COLUMNS
from features import build_preprocessor


def test_output_has_no_nans_and_matches_row_count(sample_df):
    X = sample_df[FEATURE_COLUMNS]
    transformed = build_preprocessor().fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert not np.isnan(transformed).any()


def test_continuous_block_is_standardised(sample_df):
    transformed = build_preprocessor().fit_transform(sample_df[FEATURE_COLUMNS])
    continuous_block = transformed[:, : len(CONTINUOUS_FEATURES)]
    assert np.allclose(continuous_block.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(continuous_block.std(axis=0), 1, atol=1e-6)


def test_binary_block_passes_through_unchanged(sample_df):
    prep = build_preprocessor().fit(sample_df[FEATURE_COLUMNS])
    transformed = prep.transform(sample_df[FEATURE_COLUMNS])
    start = len(CONTINUOUS_FEATURES)
    binary_block = transformed[:, start : start + len(BINARY_FEATURES)]
    np.testing.assert_array_equal(binary_block, sample_df[BINARY_FEATURES].to_numpy())


def test_nominal_block_is_one_hot(sample_df):
    transformed = build_preprocessor().fit_transform(sample_df[FEATURE_COLUMNS])
    # more output columns than input features once the nominal block is one-hot expanded
    assert transformed.shape[1] > len(FEATURE_COLUMNS)
    nominal_block = transformed[:, len(CONTINUOUS_FEATURES) + len(BINARY_FEATURES) :]
    assert set(np.unique(nominal_block)) <= {0.0, 1.0}


def test_unseen_nominal_category_does_not_raise(sample_df):
    prep = build_preprocessor().fit(sample_df[FEATURE_COLUMNS])
    row = sample_df[FEATURE_COLUMNS].head(1).copy()
    row["cp"] = 99.0
    assert prep.transform(row).shape[0] == 1


def test_missing_values_are_imputed_at_transform_time(sample_df):
    prep = build_preprocessor().fit(sample_df[FEATURE_COLUMNS])
    row = sample_df[FEATURE_COLUMNS].head(1).copy()
    row["chol"] = np.nan
    row["thal"] = np.nan
    assert not np.isnan(prep.transform(row)).any()
