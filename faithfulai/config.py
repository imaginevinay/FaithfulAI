"""Configuration for FaithfulAI."""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from dotenv import load_dotenv


# Load .env file at module import time
load_dotenv()


class LLMProvider(str, Enum):
    GROQ = "groq"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class FaithfulAIConfig(BaseModel):
    """Central config — instantiate once, pass everywhere."""

    # LLM provider (Groq / OpenAI / Anthropic)
    llm_provider: LLMProvider = LLMProvider.GROQ
    llm_model: str = "llama-3.3-70b-versatile"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None

    # HuggingFace NLI
    hf_api_token: Optional[str] = None
    hf_nli_model: str = "cross-encoder/nli-deberta-v3-base"

    # Thresholds
    hallucination_threshold: float = Field(0.5)
    drift_threshold: float = Field(0.4)
    unsupported_threshold: float = Field(0.3)

    # LLM temperature
    llm_temperature: float = 0.0

    def get_hf_token(self) -> str | None:
        """Get HuggingFace API token from config or environment."""
        if self.hf_api_token:
            return self.hf_api_token
        return os.environ.get("HF_API_TOKEN")

    def get_api_key(self) -> str:
        """Get LLM API key from config or environment."""
        if self.llm_api_key:
            return self.llm_api_key
        env_map = {
            LLMProvider.GROQ: "GROQ_API_KEY",
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            LLMProvider.OPENAI: "OPENAI_API_KEY",
        }
        key = os.environ.get(env_map[self.llm_provider], "")
        if not key:
            raise ValueError(f"No API key. Set {env_map[self.llm_provider]}.")
        return key

    def get_base_url(self) -> str | None:
        """Get LLM base URL from config or environment."""
        if self.llm_base_url:
            return self.llm_base_url
        return {
            LLMProvider.GROQ: "https://api.groq.com/openai/v1",
            LLMProvider.OPENAI: "https://api.openai.com/v1",
            LLMProvider.ANTHROPIC: None,
        }[self.llm_provider]
