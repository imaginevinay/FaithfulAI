"""Shared test fixtures."""

import pytest
from unittest.mock import AsyncMock

from faithfulai import FaithfulAIConfig
from faithfulai.llm_providers.base import BaseLLMProvider
from faithfulai.models import Claim, NLILabel


@pytest.fixture
def mock_llm():
    """Mock LLM provider that returns configurable JSON responses."""
    provider = AsyncMock(spec=BaseLLMProvider)
    return provider


@pytest.fixture
def sample_claims():
    """Sample claims for testing."""
    return [
        Claim(text="BCG launched Deckster", source_span="BCG launched Deckster", start=0, end=20),
        Claim(text="launched in 2021", source_span="in 2021", start=28, end=35),
        Claim(text="reducing time by 50%", source_span="50%", start=53, end=56),
    ]


@pytest.fixture
def sample_sources():
    """Sample source documents."""
    return ["BCG launched Deckster in 2022, cutting deck creation time by 30% across 200 pilot engagements."]


@pytest.fixture
def config():
    """Test configuration."""
    return FaithfulAIConfig(llm_api_key="test-key")
