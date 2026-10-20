"""Shared constants: filesystem paths, column roles and run settings.

Centralising these here (instead of repeating path arithmetic in every script) is the one
piece of "framework" code in this project - everything else is a plain, linear script.
"""
from pathlib import Path

# --------------------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_FILE = DATA_DIR / "raw" / "heart_disease_raw.csv"
CLEAN_DATA_FILE = DATA_DIR / "processed" / "heart_disease_clean.csv"
SAMPLE_PATIENT_FILE = DATA_DIR / "sample_patient.json"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODEL_DIR / "model.joblib"
MODEL_METADATA_FILE = MODEL_DIR / "model_metadata.json"
MLFLOW_MODEL_DIR = MODEL_DIR / "mlflow_model"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_COMPARISON_FILE = REPORTS_DIR / "model_comparison.csv"

# --------------------------------------------------------------------------------------
# Dataset schema (UCI Cleveland heart disease dataset)
# --------------------------------------------------------------------------------------
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

# Columns that need a missing-value fill + imputation strategy during cleaning.
COLUMNS_WITH_UNKNOWNS = ["ca", "thal"]

TARGET = "target"

# Continuous + ordinal counts: scaled. "ca" (vessels coloured, 0-4) is numeric/ordinal here
# rather than one-hot encoded, because more vessels coloured is monotonically worse, not a
# category - this keeps the feature space ~15 columns instead of ~22.
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

# Already binary (0/1): passed through unchanged, no encoding needed.
BINARY_FEATURES = ["sex", "fbs", "exang"]

# True unordered categories: one-hot encoded.
NOMINAL_FEATURES = ["cp", "restecg", "slope", "thal"]

FEATURE_COLUMNS = CONTINUOUS_FEATURES + BINARY_FEATURES + NOMINAL_FEATURES

# --------------------------------------------------------------------------------------
# Experiment / training settings
# --------------------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'tracking.db'}"
MLFLOW_EXPERIMENT_NAME = "heart-disease-risk"
MLFLOW_REGISTERED_MODEL_NAME = "heart-risk-classifier"
