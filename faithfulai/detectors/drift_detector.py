"""Drift detection for answer relevance."""

from __future__ import annotations

import logging

from faithfulai.llm_providers.base import BaseLLMProvider

logger = logging.getLogger("faithfulai")


DRIFT_SYSTEM_PROMPT = """You are a relevance evaluator. Given a user QUERY and an AI ANSWER,
rate how well the answer addresses the query.

Score from 0.0 to 1.0:
- 1.0 = Answer directly and completely addresses the query
- 0.7 = Answer mostly addresses the query, minor tangents
- 0.4 = Answer partially relevant but has significant drift
- 0.1 = Answer is mostly off-topic
- 0.0 = Answer has nothing to do with the query

Respond ONLY with JSON:
{
  "relevance_score": 0.0 to 1.0,
  "explanation": "brief reason"
}
"""


async def detect_drift(
    llm: BaseLLMProvider,
    query: str,
    answer: str,
) -> tuple[float, str]:
    """Detect relevance drift between query and answer.
    
    Args:
        llm: LLM provider
        query: Original user query
        answer: RAG-generated answer
        
    Returns:
        Tuple of (drift_score, explanation)
    """
    user_prompt = f"QUERY: {query}\n\nANSWER: {answer}"
    
    try:
        result = await llm.complete_json(DRIFT_SYSTEM_PROMPT, user_prompt)
        relevance_score = result.get("relevance_score", 0.5)
        
        # Drift is inverse of relevance
        drift_score = round(1.0 - relevance_score, 3)
        explanation = result.get("explanation", "")
        
        return drift_score, explanation
    
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        # Safe default: assume some drift
        return 0.5, f"Error during drift detection: {str(e)}"
