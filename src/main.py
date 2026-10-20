"""FastAPI service exposing /predict, /health and /metrics for the heart-risk model."""
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from inference import load_model, predict
from logging_utils import configure_logging
from metrics import (
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_TOTAL,
    PREDICTION_CONFIDENCE,
    PREDICTIONS_TOTAL,
    render_latest,
)
from schemas import HealthResponse, PatientRecord, PredictionResponse

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logger = configure_logging(level=getattr(logging, LOG_LEVEL, logging.INFO))
UNLOGGED_PATHS = {"/health", "/metrics"}  # liveness/readiness probes and scrapes would flood logs

model_state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    model_state["model"] = load_model()
    logger.info("model loaded", extra={"event": "startup"})
    yield
    model_state.clear()


app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts the presence of heart disease from patient health measurements.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_observability(request: Request, call_next):
    """Times every request, records Prometheus metrics, and emits one JSON log line."""
    started_at = time.perf_counter()
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        duration = time.perf_counter() - started_at
        route = request.scope.get("route")
        path_template = route.path if route is not None else request.url.path

        if path_template != "/metrics":
            HTTP_REQUESTS_TOTAL.labels(request.method, path_template, str(status_code)).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(path_template).observe(duration)

        if path_template not in UNLOGGED_PATHS:
            logger.info(
                "request handled",
                extra={
                    "event": "request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": status_code,
                    "duration_ms": round(duration * 1000, 2),
                },
            )


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in model_state}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    body, content_type = render_latest()
    return Response(content=body, media_type=content_type)


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(patient: PatientRecord) -> dict:
    result = predict([patient.model_dump()], model=model_state["model"])[0]
    PREDICTIONS_TOTAL.labels(label=result["label"]).inc()
    PREDICTION_CONFIDENCE.observe(result["confidence"])
    logger.info(
        "prediction served",
        extra={
            "event": "prediction",
            "prediction": result["prediction"],
            "confidence": result["confidence"],
        },
    )
    return result
