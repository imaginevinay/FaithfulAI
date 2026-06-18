"""Detectors package."""

from faithfulai.detectors.claim_extractor import extract_claims
from faithfulai.detectors.nli_scorer import score_claims, on_nli_notification
from faithfulai.detectors.span_extractor import extract_hallucinated_spans
from faithfulai.detectors.drift_detector import detect_drift
from faithfulai.detectors.hf_nli import HuggingFaceNLI

__all__ = [
    "extract_claims",
    "score_claims",
    "on_nli_notification",
    "extract_hallucinated_spans",
    "detect_drift",
    "HuggingFaceNLI",
]
