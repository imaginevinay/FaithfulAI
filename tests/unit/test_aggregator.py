"""Tests for aggregator logic."""

import pytest

from faithfulai.models import ClaimVerdict, Claim, NLILabel, HallucinatedSpan, Flag
from faithfulai.pipeline import FaithfulAIPipeline
from faithfulai.config import FaithfulAIConfig


def test_aggregator_all_entailment():
    """Test aggregation with all entailment verdicts."""
    config = FaithfulAIConfig(llm_api_key="test-key")
    pipeline = FaithfulAIPipeline(config)
    
    claims = [
        Claim(text="Claim 1"),
        Claim(text="Claim 2"),
        Claim(text="Claim 3"),
    ]
    verdicts = [
        ClaimVerdict(claim=c, label=NLILabel.ENTAILMENT, confidence=0.9, explanation="Supported")
        for c in claims
    ]
    
    result = pipeline._aggregate(
        verdicts=verdicts,
        hallucinated_spans=[],
        drift_score=0.0,
        drift_explanation="On topic",
        nli_backend="groq",
    )
    
    assert result.score > 0.8
    assert Flag.FAITHFUL in result.flags
    assert Flag.HALLUCINATION not in result.flags


def test_aggregator_with_contradictions():
    """Test aggregation with contradictions."""
    config = FaithfulAIConfig(llm_api_key="test-key")
    pipeline = FaithfulAIPipeline(config)
    
    claims = [
        Claim(text="Claim 1"),
        Claim(text="Claim 2"),
    ]
    verdicts = [
        ClaimVerdict(
            claim=claims[0],
            label=NLILabel.ENTAILMENT,
            confidence=0.9,
            explanation="Supported",
        ),
        ClaimVerdict(
            claim=claims[1],
            label=NLILabel.CONTRADICTION,
            confidence=0.95,
            explanation="Contradicted",
        ),
    ]
    
    result = pipeline._aggregate(
        verdicts=verdicts,
        hallucinated_spans=[HallucinatedSpan(text="wrong", start=0, end=5)],
        drift_score=0.1,
        drift_explanation="Mostly on topic",
        nli_backend="groq",
    )
    
    assert result.score < 0.7
    assert Flag.HALLUCINATION in result.flags


def test_aggregator_drift_detection():
    """Test aggregation with high drift score."""
    config = FaithfulAIConfig(llm_api_key="test-key", drift_threshold=0.4)
    pipeline = FaithfulAIPipeline(config)
    
    claims = [Claim(text="Claim")]
    verdicts = [
        ClaimVerdict(
            claim=claims[0],
            label=NLILabel.ENTAILMENT,
            confidence=0.9,
            explanation="Supported",
        ),
    ]
    
    result = pipeline._aggregate(
        verdicts=verdicts,
        hallucinated_spans=[],
        drift_score=0.8,  # High drift
        drift_explanation="Off topic",
        nli_backend="groq",
    )
    
    assert Flag.DRIFT in result.flags


def test_aggregator_unsupported_threshold():
    """Test aggregation with unsupported claims."""
    config = FaithfulAIConfig(llm_api_key="test-key", unsupported_threshold=0.3)
    pipeline = FaithfulAIPipeline(config)
    
    claims = [Claim(text="C1"), Claim(text="C2"), Claim(text="C3")]
    verdicts = [
        ClaimVerdict(claim=claims[0], label=NLILabel.ENTAILMENT, confidence=0.9, explanation="Ok"),
        ClaimVerdict(claim=claims[1], label=NLILabel.NEUTRAL, confidence=0.5, explanation="Unsure"),
        ClaimVerdict(claim=claims[2], label=NLILabel.NEUTRAL, confidence=0.5, explanation="Unsure"),
    ]
    
    result = pipeline._aggregate(
        verdicts=verdicts,
        hallucinated_spans=[],
        drift_score=0.0,
        drift_explanation="On topic",
        nli_backend="groq",
    )
    
    # 2 out of 3 are unsupported (0.67) which is > 0.3 threshold
    assert Flag.UNSUPPORTED in result.flags


def test_aggregator_explanation_format():
    """Test that explanation starts with [NLI: backend]."""
    config = FaithfulAIConfig(llm_api_key="test-key")
    pipeline = FaithfulAIPipeline(config)
    
    claims = [Claim(text="Test")]
    verdicts = [
        ClaimVerdict(claim=claims[0], label=NLILabel.ENTAILMENT, confidence=0.9, explanation="Ok"),
    ]
    
    result = pipeline._aggregate(
        verdicts=verdicts,
        hallucinated_spans=[],
        drift_score=0.0,
        drift_explanation="On topic",
        nli_backend="huggingface+groq",
    )
    
    assert result.explanation.startswith("[NLI: huggingface+groq]")
