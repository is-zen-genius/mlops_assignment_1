"""Structured JSON logging for the API, so logs are directly queryable/greppable as JSON."""
import logging
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """Renders each log record as a single JSON line: timestamp + message fields + extras."""

    RESERVED = set(logging.LogRecord(__name__, 0, "", 0, "", (), None).__dict__) | {"message"}

    def format(self, record: logging.LogRecord) -> str:
        import json

        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extras = {k: v for k, v in record.__dict__.items() if k not in self.RESERVED}
        payload.update(extras)
        return json.dumps(payload)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger("heart_api")
    logger.setLevel(level)
    logger.handlers = [handler]
    logger.propagate = False
    return logger
