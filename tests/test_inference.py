import pytest

from config import FEATURE_COLUMNS, MODEL_FILE, SAMPLE_PATIENT_FILE
from inference import predict


def test_predict_output_schema(sample_df, trained_model):
    records = sample_df[FEATURE_COLUMNS].head(3).to_dict("records")
    outputs = predict(records, model=trained_model)
    assert len(outputs) == 3
    for item in outputs:
        assert set(item) == {"prediction", "label", "confidence", "probability_disease"}
        assert item["prediction"] in (0, 1)
        assert 0.5 <= item["confidence"] <= 1.0
        assert item["label"] == ("disease" if item["prediction"] else "no disease")


def test_predict_raises_on_missing_feature(sample_df, trained_model):
    record = sample_df[FEATURE_COLUMNS].head(1).to_dict("records")[0]
    del record["age"]
    with pytest.raises(KeyError):
        predict([record], model=trained_model)


@pytest.mark.skipif(not MODEL_FILE.exists(), reason="trained model artifact not present")
@pytest.mark.skipif(not SAMPLE_PATIENT_FILE.exists(), reason="sample patient file not present")
def test_saved_model_predicts_on_the_shipped_sample_patient():
    import json

    sample = json.loads(SAMPLE_PATIENT_FILE.read_text())
    assert predict([sample])[0]["prediction"] in (0, 1)
