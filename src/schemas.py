"""Pydantic request/response models for the API (kept separate from routing/app wiring)."""
from typing import Literal

from pydantic import BaseModel, Field

SAMPLE_PATIENT = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
}


class PatientRecord(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: int = Field(..., ge=50, le=260, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: int = Field(..., ge=50, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of the peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels coloured (0-4)")
    thal: Literal[3, 6, 7] = Field(..., description="3 normal, 6 fixed defect, 7 reversible defect")

    model_config = {"json_schema_extra": {"example": SAMPLE_PATIENT}}


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    probability_disease: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
