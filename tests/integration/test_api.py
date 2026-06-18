"""Integration tests for API."""

import pytest


@pytest.mark.asyncio
async def test_api_evaluate_valid():
    """Test /evaluate endpoint with valid request."""
    pytest.skip("Requires FastAPI server setup")


@pytest.mark.asyncio
async def test_api_health():
    """Test /health endpoint."""
    pytest.skip("Requires FastAPI server setup")


@pytest.mark.asyncio
async def test_api_invalid_request():
    """Test /evaluate with invalid request."""
    pytest.skip("Requires FastAPI server setup")
