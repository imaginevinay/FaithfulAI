"""Tests for claim extraction."""

import pytest
from unittest.mock import AsyncMock

from faithfulai.detectors.claim_extractor import extract_claims
from faithfulai.models import Claim


@pytest.mark.asyncio
async def test_extract_claims_success(mock_llm):
    """Test successful claim extraction."""
    mock_llm.complete_json.return_value = {
        "claims": [
            {"text": "BCG launched Deckster", "source_span": "BCG launched Deckster"},
            {"text": "in 2022", "source_span": "in 2022"},
        ]
    }
    
    answer = "BCG launched Deckster in 2022, cutting deck creation time by 30%."
    claims = await extract_claims(mock_llm, "When was Deckster launched?", answer)
    
    assert len(claims) == 2
    assert claims[0].text == "BCG launched Deckster"
    assert claims[1].text == "in 2022"
    # Check positions found
    assert claims[0].start >= 0
    assert claims[1].start >= 0


@pytest.mark.asyncio
async def test_extract_claims_empty(mock_llm):
    """Test extraction with no claims."""
    mock_llm.complete_json.return_value = {"claims": []}
    
    claims = await extract_claims(mock_llm, "Query?", "Answer")
    assert claims == []


@pytest.mark.asyncio
async def test_extract_claims_malformed(mock_llm):
    """Test extraction with malformed response."""
    mock_llm.complete_json.side_effect = Exception("JSON error")
    
    claims = await extract_claims(mock_llm, "Query?", "Answer")
    assert claims == []


@pytest.mark.asyncio
async def test_extract_claims_case_insensitive(mock_llm):
    """Test case-insensitive span matching."""
    mock_llm.complete_json.return_value = {
        "claims": [
            {"text": "BCG launched", "source_span": "bcg launched"},
        ]
    }
    
    answer = "BCG launched Deckster"
    claims = await extract_claims(mock_llm, "Query?", answer)
    
    assert len(claims) == 1
    assert claims[0].start >= 0  # Should find despite case difference
