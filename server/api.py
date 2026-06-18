"""FastAPI server for FaithfulAI."""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from dotenv import load_dotenv

from faithfulai.config import FaithfulAIConfig, LLMProvider
from faithfulai.models import EvalRequest, EvalResult
from faithfulai.pipeline import FaithfulAIPipeline
from faithfulai.logging_config import setup_logging

# Load environment
load_dotenv()

# Setup logging
setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("faithfulai")

app = FastAPI(title="FaithfulAI", version="0.1.0")

# Lazy-init pipeline
_pipeline: FaithfulAIPipeline | None = None


def get_pipeline() -> FaithfulAIPipeline:
    """Get or create pipeline instance."""
    global _pipeline
    if _pipeline is None:
        provider_str = os.environ.get("LLM_PROVIDER", "groq").lower()
        provider = LLMProvider(provider_str) if provider_str in ["groq", "openai", "anthropic"] else LLMProvider.GROQ
        
        config = FaithfulAIConfig(
            llm_provider=provider,
            llm_model=os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile"),
        )
        _pipeline = FaithfulAIPipeline(config)
        logger.info("Pipeline initialized")
    
    return _pipeline


@app.post("/evaluate")
async def evaluate(request: EvalRequest) -> EvalResult:
    """Evaluate a RAG answer for hallucination.
    
    Args:
        request: Evaluation request with query, answer, and sources
        
    Returns:
        Evaluation result with score, flags, and explanations
    """
    pipeline = get_pipeline()
    result = await pipeline.evaluate(request.query, request.rag_answer, request.sources)
    return result


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Shutting down FaithfulAI API")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
