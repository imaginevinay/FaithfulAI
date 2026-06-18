"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a text completion."""
        pass

    @abstractmethod
    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Generate a JSON completion."""
        pass
