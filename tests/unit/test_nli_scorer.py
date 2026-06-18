"""Tests for NLI scoring."""

import pytest
from unittest.mock import AsyncMock

from faithfulai.models import Claim, ClaimVerdict, NLILabel
from faithfulai.detectors.nli_scorer import score_claims, on_nli_notification


@pytest.mark.asyncio
async def test_nli_scoring_with_groq_only(mock_llm, sample_claims, sample_sources):
    """Test scoring with Groq (no HuggingFace)."""
    mock_llm.complete_json.return_value = {
        "label": "contradiction",
        "confidence": 0.95,
        "supporting_source_idx": None,
        "explanation": "Date mismatch",
    }
    
    verdicts, backend = await score_claims(
        sample_claims,
        sample_sources,
        mock_llm,
        hf=None,
    )
    
    assert len(verdicts) == len(sample_claims)
    assert backend == "groq"
    for verdict in verdicts:
        assert verdict.label in [NLILabel.ENTAILMENT, NLILabel.NEUTRAL, NLILabel.CONTRADICTION]


@pytest.mark.asyncio
async def test_nli_notification_callback(mock_llm, sample_claims, sample_sources):
    """Test notification callback when HF NLI fails."""
    notifications = []
    
    def callback(msg):
        notifications.append(msg)
    
    on_nli_notification(callback)
    
    # Mock HF with inference failure
    mock_hf = AsyncMock()
    mock_hf.classify.return_value = None  # Inference failure
    
    # Mock Groq fallback
    mock_llm.complete_json.return_value = {
        "label": "neutral",
        "confidence": 0.5,
        "supporting_source_idx": None,
        "explanation": "Fallback from HF",
    }
    
    verdicts, backend = await score_claims(
        sample_claims,
        sample_sources,
        mock_llm,
        hf=mock_hf,
    )
    
    # Should have notified about fallback
    assert any("failed" in str(n).lower() or "inference" in str(n).lower() for n in notifications)


@pytest.mark.asyncio
async def test_confidence_bounds(mock_llm, sample_claims, sample_sources):
    """Test confidence values are within bounds."""
    mock_llm.complete_json.return_value = {
        "label": "entailment",
        "confidence": 0.85,
        "supporting_source_idx": 0,
        "explanation": "Supported",
    }
    
    verdicts, _ = await score_claims(
        sample_claims,
        sample_sources,
        mock_llm,
        hf=None,
    )
    
    for verdict in verdicts:
        assert 0.0 <= verdict.confidence <= 1.0
