"""Prometheus metric definitions, kept in one place so app.py and tests share the same objects."""
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

HTTP_REQUESTS_TOTAL = Counter(
    "heart_api_http_requests_total",
    "Total HTTP requests received",
    ["method", "path", "status"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "heart_api_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)

PREDICTIONS_TOTAL = Counter(
    "heart_api_predictions_total",
    "Predictions served, split by predicted label",
    ["label"],
)

PREDICTION_CONFIDENCE = Histogram(
    "heart_api_prediction_confidence",
    "Confidence score of served predictions",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)


def render_latest() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
