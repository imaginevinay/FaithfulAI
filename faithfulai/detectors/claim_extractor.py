"""Claim extraction from RAG answers."""

from __future__ import annotations

import logging

from faithfulai.models import Claim
from faithfulai.llm_providers.base import BaseLLMProvider

logger = logging.getLogger("faithfulai")


CLAIM_EXTRACTION_PROMPT = """You are a claim extraction engine. Given a question and an AI-generated answer,
break the answer into individual atomic factual claims.

Rules:
- Each claim must be a single, independently verifiable statement.
- Preserve the EXACT wording from the answer — do not paraphrase.
- Include numbers, dates, names, percentages exactly as they appear.
- Skip filler phrases like "Sure, here is..." or "Based on the context...".
- Return between 1-20 claims.

Respond ONLY with a JSON object in this format:
{
  "claims": [
    {"text": "exact claim text from answer", "source_span": "the phrase in the answer this came from"}
  ]
}
"""


async def extract_claims(
    llm: BaseLLMProvider, query: str, answer: str
) -> list[Claim]:
    """Extract atomic claims from a RAG answer.
    
    Args:
        llm: LLM provider
        query: Original user query
        answer: RAG-generated answer
        
    Returns:
        List of extracted claims with positions
    """
    user_prompt = f"Question: {query}\n\nAnswer to decompose:\n{answer}"
    
    try:
        result = await llm.complete_json(CLAIM_EXTRACTION_PROMPT, user_prompt)
    except Exception as e:
        logger.error(f"Claim extraction failed: {e}")
        return []
    
    claims = []
    for claim_data in result.get("claims", []):
        text = claim_data.get("text", "")
        source_span = claim_data.get("source_span", "")
        
        if not text:
            continue
            
        # Find source_span position in answer
        start = -1
        end = -1
        if source_span:
            pos = answer.lower().find(source_span.lower())
            if pos >= 0:
                start = pos
                end = pos + len(source_span)
        
        claims.append(
            Claim(
                text=text,
                source_span=source_span,
                start=start,
                end=end,
            )
        )
    
    return claims
