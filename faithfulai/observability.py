"""Datadog tracing + metrics. Graceful degradation if ddtrace not installed."""

from __future__ import annotations

import time
import logging
from contextlib import asynccontextmanager
from functools import wraps

logger = logging.getLogger("faithfulai")

# ── Try importing ddtrace ────────────────────────────────────────────
try:
    from ddtrace import tracer
    from ddtrace import patch_all
    patch_all()
    DD_AVAILABLE = True
    logger.info("Datadog tracing enabled")
except ImportError:
    DD_AVAILABLE = False
    tracer = None

# ── Try importing datadog statsd ─────────────────────────────────────
try:
    from datadog import DogStatsd
    statsd = DogStatsd(namespace="faithfulai")
    METRICS_AVAILABLE = True
except ImportError:
    statsd = None
    METRICS_AVAILABLE = False


# ── Trace context manager ────────────────────────────────────────────
@asynccontextmanager
async def trace_span(name: str, **tags):
    """Wrap an async block in a Datadog trace span.
    
    Usage:
        async with trace_span("faithfulai.claim_extraction", claim_count=5):
            result = await extract_claims(...)
    """
    if DD_AVAILABLE and tracer:
        with tracer.trace(name, service="faithfulai") as span:
            for k, v in tags.items():
                span.set_tag(k, v)
            start = time.perf_counter()
            try:
                yield span
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                span.set_metric("duration_ms", elapsed_ms)
    else:
        yield None


# ── Metrics helpers ──────────────────────────────────────────────────
def emit_counter(name: str, value: int = 1, tags: list[str] | None = None):
    """Increment a counter metric."""
    if METRICS_AVAILABLE and statsd:
        statsd.increment(name, value, tags=tags)


def emit_gauge(name: str, value: float, tags: list[str] | None = None):
    """Set a gauge metric."""
    if METRICS_AVAILABLE and statsd:
        statsd.gauge(name, value, tags=tags)


def emit_histogram(name: str, value: float, tags: list[str] | None = None):
    """Record a histogram metric."""
    if METRICS_AVAILABLE and statsd:
        statsd.histogram(name, value, tags=tags)
