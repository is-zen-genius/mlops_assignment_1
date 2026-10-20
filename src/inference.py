"""Turn raw patient record(s) into a prediction + confidence score."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from config import FEATURE_COLUMNS, MODEL_FILE, SAMPLE_PATIENT_FILE

DISEASE_THRESHOLD = 0.5


def load_model(path: Path = MODEL_FILE):
    return joblib.load(path)


def predict(records: list[dict], model=None) -> list[dict]:
    """records: list of dicts, each with the 13 feature names from config.FEATURE_COLUMNS."""
    model = model if model is not None else load_model()
    frame = pd.DataFrame(records)[FEATURE_COLUMNS].astype("float64")
    disease_probability = model.predict_proba(frame)[:, 1]

    outputs = []
    for p_disease in disease_probability:
        has_disease = int(p_disease >= DISEASE_THRESHOLD)
        confidence = p_disease if has_disease else 1 - p_disease
        outputs.append({
            "prediction": has_disease,
            "label": "disease" if has_disease else "no disease",
            "confidence": round(float(confidence), 4),
            "probability_disease": round(float(p_disease), 4),
        })
    return outputs


if __name__ == "__main__":
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE_PATIENT_FILE
    patient = json.loads(input_path.read_text())
    print(json.dumps(predict([patient])[0], indent=2))
