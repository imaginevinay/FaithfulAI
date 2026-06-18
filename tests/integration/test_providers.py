"""Tests for LLM providers."""

import pytest


@pytest.mark.asyncio
async def test_groq_provider():
    """Test Groq provider."""
    pytest.skip("Requires real API key")


@pytest.mark.asyncio
async def test_openai_provider():
    """Test OpenAI provider."""
    pytest.skip("Requires real API key")


@pytest.mark.asyncio
async def test_anthropic_provider():
    """Test Anthropic provider."""
    pytest.skip("Requires real API key")
