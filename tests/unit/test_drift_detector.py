"""Tests for drift detection."""

import pytest

from faithfulai.detectors.drift_detector import detect_drift


@pytest.mark.asyncio
async def test_drift_detection_high_relevance(mock_llm):
    """Test low drift for high relevance answer."""
    mock_llm.complete_json.return_value = {
        "relevance_score": 0.9,
        "explanation": "Answer directly addresses the query",
    }
    
    drift_score, explanation = await detect_drift(
        mock_llm,
        "What happened in 2023?",
        "In 2023, many things happened including economic changes.",
    )
    
    assert drift_score == 0.1
    assert "directly" in explanation.lower() or "addresses" in explanation.lower()


@pytest.mark.asyncio
async def test_drift_detection_low_relevance(mock_llm):
    """Test high drift for low relevance answer."""
    mock_llm.complete_json.return_value = {
        "relevance_score": 0.1,
        "explanation": "Answer is mostly off-topic",
    }
    
    drift_score, explanation = await detect_drift(
        mock_llm,
        "What is machine learning?",
        "Bananas are yellow and grow in tropical regions.",
    )
    
    assert drift_score == 0.9


@pytest.mark.asyncio
async def test_drift_detection_error_handling(mock_llm):
    """Test error handling returns safe default."""
    mock_llm.complete_json.side_effect = Exception("API error")
    
    drift_score, explanation = await detect_drift(
        mock_llm,
        "Query?",
        "Answer",
    )
    
    # Should return safe default
    assert drift_score == 0.5
    assert "error" in explanation.lower()
