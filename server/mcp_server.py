"""MCP server for Faithful evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
import os

from dotenv import load_dotenv

from faithfulai.config import FaithfulAIConfig, LLMProvider
from faithfulai.pipeline import FaithfulAIPipeline

# Load environment variables
load_dotenv()

logger = logging.getLogger("faithful_ai")


def create_config_from_env() -> FaithfulAIConfig:
    """Create config from environment variables."""
    provider_str = os.environ.get("LLM_PROVIDER", "groq").lower()
    provider = LLMProvider(provider_str) if provider_str in ["groq", "openai", "anthropic"] else LLMProvider.GROQ
    
    return FaithfulAIConfig(
        llm_provider=provider,
        llm_model=os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile"),
        llm_api_key=None,  # Will be read from env
        hf_api_token=None,  # Will be read from env
    )


async def evaluate_faithfulness(query: str, rag_answer: str, sources: list[str]) -> dict:
    """Evaluate faithfulness of a RAG answer.
    
    Args:
        query: User's original question
        rag_answer: RAG-generated answer
        sources: Source documents
        
    Returns:
        JSON-serialized EvalResult
    """
    config = create_config_from_env()
    pipeline = FaithfulAIPipeline(config)
    
    result = await pipeline.evaluate(query, rag_answer, sources)
    return result.model_dump()


def main():
    """MCP server entry point."""
    # This is a placeholder for MCP server implementation
    # In a real implementation, would use the mcp SDK
    print("FaithfulAI MCP server initialized")
    print(f"Available function: evaluate_faithfulness(query, rag_answer, sources)")
    print("Ready to accept MCP calls")


if __name__ == "__main__":
    main()
