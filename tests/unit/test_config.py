"""Tests for configuration."""

import os
import pytest

from faithfulai.config import FaithfulAIConfig, LLMProvider


def test_get_api_key_from_config():
    """Test getting API key from config."""
    config = FaithfulAIConfig(llm_api_key="test-key-123")
    assert config.get_api_key() == "test-key-123"


def test_get_api_key_from_env(monkeypatch):
    """Test getting API key from environment."""
    monkeypatch.setenv("GROQ_API_KEY", "env-key-456")
    config = FaithfulAIConfig()
    assert config.get_api_key() == "env-key-456"


def test_get_api_key_config_prefers_over_env(monkeypatch):
    """Test config key takes precedence over env."""
    monkeypatch.setenv("GROQ_API_KEY", "env-key")
    config = FaithfulAIConfig(llm_api_key="config-key")
    assert config.get_api_key() == "config-key"


def test_get_api_key_missing_raises_error(monkeypatch):
    """Test missing API key raises error."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    
    config = FaithfulAIConfig()
    with pytest.raises(ValueError):
        config.get_api_key()


def test_get_base_url_per_provider():
    """Test correct base URLs per provider."""
    groq_config = FaithfulAIConfig(
        llm_provider=LLMProvider.GROQ,
        llm_api_key="key",
    )
    assert groq_config.get_base_url() == "https://api.groq.com/openai/v1"
    
    openai_config = FaithfulAIConfig(
        llm_provider=LLMProvider.OPENAI,
        llm_api_key="key",
    )
    assert openai_config.get_base_url() == "https://api.openai.com/v1"
    
    anthropic_config = FaithfulAIConfig(
        llm_provider=LLMProvider.ANTHROPIC,
        llm_api_key="key",
    )
    assert anthropic_config.get_base_url() is None


def test_get_hf_token_from_config():
    """Test getting HuggingFace token from config."""
    config = FaithfulAIConfig(hf_api_token="hf-token-123")
    assert config.get_hf_token() == "hf-token-123"


def test_get_hf_token_from_env(monkeypatch):
    """Test getting HuggingFace token from environment."""
    monkeypatch.setenv("HF_API_TOKEN", "hf-env-token")
    config = FaithfulAIConfig()
    assert config.get_hf_token() == "hf-env-token"


def test_get_hf_token_none_when_not_set(monkeypatch):
    """Test HuggingFace token returns None when not set."""
    monkeypatch.delenv("HF_API_TOKEN", raising=False)
    config = FaithfulAIConfig()
    assert config.get_hf_token() is None
