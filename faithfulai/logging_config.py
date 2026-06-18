"""Structured JSON logging for FaithfulAI."""

import json
import logging
import sys
import uuid
from contextvars import ContextVar

# Per-evaluation trace ID
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def new_trace_id() -> str:
    """Generate and set a new trace ID for this evaluation."""
    tid = uuid.uuid4().hex[:12]
    trace_id_var.set(tid)
    return tid


class JSONFormatter(logging.Formatter):
    """Emit structured JSON log lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "trace_id": trace_id_var.get(""),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logging(level: str = "INFO"):
    """Configure faithfulai logger with JSON output."""
    logger = logging.getLogger("faithfulai")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    return logger
