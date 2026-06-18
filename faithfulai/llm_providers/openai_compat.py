"""OpenAI-compatible provider for Groq and OpenAI."""

from __future__ import annotations

import json
import logging
import httpx

from faithfulai.llm_providers.base import BaseLLMProvider

logger = logging.getLogger("faithfulai")


class OpenAICompatibleProvider(BaseLLMProvider):
    """Provider for OpenAI-compatible APIs (Groq, OpenAI)."""

    def __init__(self, api_key: str, base_url: str, model: str, temperature: float = 0.0):
        """Initialize the provider.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for the API (e.g., https://api.groq.com/openai/v1)
            model: Model name to use
            temperature: Temperature for generation (0.0 to 2.0)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a text completion."""
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Generate a JSON completion."""
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            # Strip markdown fences if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            return json.loads(content)
