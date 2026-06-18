"""Anthropic provider implementation."""

from __future__ import annotations

import json
import logging
import httpx

from faithfulai.llm_providers.base import BaseLLMProvider

logger = logging.getLogger("faithfulai")


class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic API."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.0):
        """Initialize the provider.
        
        Args:
            api_key: API key for authentication
            model: Model name to use
            temperature: Temperature for generation (0.0 to 1.0)
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a text completion."""
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": self.temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]

    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Generate a JSON completion."""
        enhanced_system = system_prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON."
        
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": self.temperature,
            "system": enhanced_system,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()
            content = data["content"][0]["text"]
            
            # Strip markdown fences if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            return json.loads(content)
