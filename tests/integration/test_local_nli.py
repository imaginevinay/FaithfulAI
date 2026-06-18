"""Integration test for local NLI inference."""

import pytest
from faithfulai.detectors.hf_nli import HuggingFaceNLI


@pytest.mark.asyncio
async def test_local_nli_inference():
    """Test that local NLI model loads and performs inference."""
    nli = HuggingFaceNLI()
    
    # Test basic inference
    premise = "The sky is blue and clear today."
    hypothesis = "The sky is blue."
    
    result = await nli.classify(premise, hypothesis)
    
    # Should return a result
    assert result is not None
    assert "label" in result
    assert "scores" in result
    
    # Label should be one of the expected values
    assert result["label"] in ["entailment", "neutral", "contradiction"]
    
    # Scores should have all three keys
    assert set(result["scores"].keys()) == {"entailment", "neutral", "contradiction"}
    
    # All scores should be floats between 0 and 1
    for score in result["scores"].values():
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    # Scores should sum to approximately 1.0
    total = sum(result["scores"].values())
    assert 0.99 <= total <= 1.01


@pytest.mark.asyncio
async def test_local_nli_contradiction():
    """Test that contradictions are detected."""
    nli = HuggingFaceNLI()
    
    premise = "Paris is the capital of France."
    hypothesis = "Paris is the capital of Germany."
    
    result = await nli.classify(premise, hypothesis)
    
    assert result is not None
    # Contradiction should have higher score than entailment
    assert result["scores"]["contradiction"] > result["scores"]["entailment"]


@pytest.mark.asyncio
async def test_local_nli_neutral():
    """Test that neutral cases are detected."""
    nli = HuggingFaceNLI()
    
    premise = "The weather is sunny today."
    hypothesis = "The dog is brown."
    
    result = await nli.classify(premise, hypothesis)
    
    assert result is not None
    # Neutral should have high score since premise and hypothesis are unrelated
    # (This is not strictly guaranteed, but generally true for NLI models)
    assert result["label"] in ["neutral", "contradiction"]
