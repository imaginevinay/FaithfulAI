"""Tests for data models."""

import pytest
from faithfulai.models import (
    EvalRequest,
    Flag,
    NLILabel,
    EvalResult,
    HallucinatedSpan,
    ClaimVerdict,
    Claim,
)


def test_eval_request_validation():
    """Test EvalRequest validates correctly."""
    request = EvalRequest(
        query="What is X?",
        rag_answer="X is Y",
        sources=["Source 1"],
    )
    assert request.query == "What is X?"
    assert request.rag_answer == "X is Y"
    assert request.sources == ["Source 1"]


def test_eval_request_missing_fields():
    """Test EvalRequest rejects missing fields."""
    with pytest.raises(Exception):
        EvalRequest(query="What is X?")


def test_flag_enum_values():
    """Test Flag enum has correct values."""
    assert Flag.FAITHFUL.value == "FAITHFUL"
    assert Flag.HALLUCINATION.value == "HALLUCINATION"
    assert Flag.DRIFT.value == "DRIFT"
    assert Flag.UNSUPPORTED.value == "UNSUPPORTED"


def test_nli_label_enum_values():
    """Test NLILabel enum has correct values."""
    assert NLILabel.ENTAILMENT.value == "entailment"
    assert NLILabel.CONTRADICTION.value == "contradiction"
    assert NLILabel.NEUTRAL.value == "neutral"


def test_hallucinated_span_defaults():
    """Test HallucinatedSpan defaults."""
    span = HallucinatedSpan(text="test")
    assert span.text == "test"
    assert span.start == -1
    assert span.end == -1
    assert span.reason == ""


def test_claim_verdict_confidence_validation():
    """Test ClaimVerdict confidence bounds."""
    claim = Claim(text="test claim")
    
    # Valid confidence
    verdict = ClaimVerdict(
        claim=claim,
        label=NLILabel.ENTAILMENT,
        confidence=0.5,
    )
    assert verdict.confidence == 0.5
    
    # Invalid confidence > 1.0
    with pytest.raises(Exception):
        ClaimVerdict(
            claim=claim,
            label=NLILabel.ENTAILMENT,
            confidence=1.5,
        )
    
    # Invalid confidence < 0.0
    with pytest.raises(Exception):
        ClaimVerdict(
            claim=claim,
            label=NLILabel.ENTAILMENT,
            confidence=-0.5,
        )


def test_eval_result_serialization():
    """Test EvalResult serializes to JSON."""
    result = EvalResult(
        score=0.9,
        flags=[Flag.FAITHFUL],
        explanation="All claims supported",
    )
    
    # Should be JSON serializable
    json_dict = result.model_dump()
    assert json_dict["score"] == 0.9
    assert "FAITHFUL" in json_dict["flags"]
