"""Integration tests for the complete pipeline."""

import pytest

from faithfulai import FaithfulAIConfig, FaithfulAIPipeline, Flag


@pytest.mark.asyncio
async def test_pipeline_faithful_case(mock_llm):
    """Test pipeline with fully faithful answer."""
    # Mock claims
    mock_llm.complete_json.side_effect = [
        # Claims extraction
        {
            "claims": [
                {"text": "Tesla delivered 1.81M vehicles", "source_span": "1.81 million"},
                {"text": "increase of 38%", "source_span": "38%"},
            ]
        },
        # NLI: first claim
        {
            "label": "entailment",
            "confidence": 0.95,
            "supporting_source_idx": 0,
            "explanation": "Matches source",
        },
        # NLI: second claim
        {
            "label": "entailment",
            "confidence": 0.93,
            "supporting_source_idx": 0,
            "explanation": "Matches source",
        },
        # Drift detection
        {
            "relevance_score": 0.95,
            "explanation": "Answer directly addresses query",
        },
    ]
    
    config = FaithfulAIConfig(llm_api_key="test-key")
    pipeline = FaithfulAIPipeline(config)
    
    # Monkey-patch to use mock
    pipeline._llm = mock_llm
    
    result = await pipeline.evaluate(
        query="How many vehicles did Tesla deliver in 2023?",
        rag_answer="Tesla delivered 1.81 million vehicles in 2023, a 38% increase.",
        sources=["Tesla delivered 1.81 million vehicles in 2023, up 38% from 2022."],
    )
    # print(result)
    assert result.score > 0.7
    assert Flag.DRIFT in result.flags
    assert len(result.details.hallucinated_spans) == 0


@pytest.mark.asyncio
async def test_pipeline_hallucination_case(mock_llm):
    """Test pipeline detecting hallucination."""
    # Mock responses for hallucination scenario
    mock_llm.complete_json.side_effect = [
        # Claims extraction
        {
            "claims": [
                {"text": "BCG launched Deckster", "source_span": "BCG launched Deckster", },
                {"text": "in 2021", "source_span": "in 2021"},
                {"text": "50% reduction", "source_span": "50%"},
            ]
        },
        # NLI verdicts
        {
            "label": "entailment",
            "confidence": 0.9,
            "supporting_source_idx": 0,
            "explanation": "Claim supported",
        },
        {
            "label": "contradiction",
            "confidence": 0.95,
            "supporting_source_idx": None,
            "explanation": "Date should be 2022, not 2021",
        },
        {
            "label": "contradiction",
            "confidence": 0.92,
            "supporting_source_idx": None,
            "explanation": "Percentage should be 30%, not 50%",
        },
        # Drift detection
        {
            "relevance_score": 0.85,
            "explanation": "Mostly relevant",
        },
    ]
    
    config = FaithfulAIConfig(llm_api_key="test-key")
    pipeline = FaithfulAIPipeline(config)
    pipeline._llm = mock_llm
    
    result = await pipeline.evaluate(
        query="When was Deckster launched?",
        rag_answer="BCG launched Deckster in 2021, reducing time by 50%.",
        sources=["BCG launched Deckster in 2022, cutting deck creation time by 30%."],
    )
    
    assert result.score < 0.7
    assert Flag.HALLUCINATION in result.flags
    assert len(result.details.hallucinated_spans) > 0


@pytest.mark.asyncio
async def test_pipeline_no_claims(mock_llm):
    """Test pipeline when no claims extracted."""
    mock_llm.complete_json.return_value = {"claims": []}
    
    config = FaithfulAIConfig(llm_api_key="test-key")
    pipeline = FaithfulAIPipeline(config)
    pipeline._llm = mock_llm
    
    result = await pipeline.evaluate(
        query="Is this good?",
        rag_answer="Yes, it is good.",
        sources=["Something"],
    )
    
    assert result.score == 1.0
    assert Flag.FAITHFUL in result.flags
