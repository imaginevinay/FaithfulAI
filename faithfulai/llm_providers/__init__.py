"""Factory for creating LLM providers."""

from __future__ import annotations

from faithfulai.config import FaithfulAIConfig, LLMProvider
from faithfulai.llm_providers.base import BaseLLMProvider
from faithfulai.llm_providers.openai_compat import OpenAICompatibleProvider
from faithfulai.llm_providers.anthropic_provider import AnthropicProvider


def create_provider(config: FaithfulAIConfig) -> BaseLLMProvider:
    """Create an LLM provider based on configuration.
    
    Args:
        config: FaithFulAI configuration
        
    Returns:
        An initialized LLM provider
    """
    api_key = config.get_api_key()

    if config.llm_provider == LLMProvider.ANTHROPIC:
        return AnthropicProvider(
            api_key=api_key,
            model=config.llm_model,
            temperature=config.llm_temperature,
        )
    else:  # GROQ or OPENAI
        return OpenAICompatibleProvider(
            api_key=api_key,
            base_url=config.get_base_url(),
            model=config.llm_model,
            temperature=config.llm_temperature,
        )
