"""NLI scoring with HuggingFace and Groq fallback."""

from __future__ import annotations

import logging
from typing import Callable, Optional

from faithfulai.models import Claim, ClaimVerdict, NLILabel
from faithfulai.llm_providers.base import BaseLLMProvider
from faithfulai.detectors.hf_nli import HuggingFaceNLI

logger = logging.getLogger("faithfulai")


_notification_callback: Optional[Callable[[str], None]] = None


def on_nli_notification(callback: Callable[[str], None]) -> None:
    """Register a callback for NLI notifications (e.g., HF fallback)."""
    global _notification_callback
    _notification_callback = callback


def _notify(message: str) -> None:
    """Fire the notification callback and log."""
    logger.warning(message)
    if _notification_callback:
        _notification_callback(message)


NLI_SYSTEM_PROMPT = """You are a Natural Language Inference (NLI) engine for detecting hallucinations in RAG answers.

Given a CLAIM and one or more SOURCE documents, classify the claim as:

- "entailment": The claim is fully supported by at least one source.
- "contradiction": The claim directly conflicts with information in the sources.
  A claim that changes a NUMBER, DATE, NAME, or PERCENTAGE from what the source says is a CONTRADICTION.
- "neutral": The claim cannot be verified or refuted from the sources alone.

Be precise and strict. Even small factual changes (2022→2021, 30%→50%, 200→500) are contradictions.

Respond ONLY with a JSON object:
{
  "label": "entailment" | "contradiction" | "neutral",
  "confidence": 0.0 to 1.0,
  "supporting_source_idx": null or index (0-based) of best supporting source,
  "explanation": "brief reason"
}
"""


async def score_claims(
    claims: list[Claim],
    sources: list[str],
    llm: BaseLLMProvider,
    hf: Optional[HuggingFaceNLI] = None,
) -> tuple[list[ClaimVerdict], str]:
    """Score claims using local HuggingFace NLI with Groq fallback.
    
    Args:
        claims: List of claims to evaluate
        sources: List of source documents
        llm: LLM provider for fallback
        hf: HuggingFace NLI client (local transformers-based)
        
    Returns:
        Tuple of (list of verdicts, backend string)
    """
    verdicts: list[ClaimVerdict] = []
    used_hf = False
    used_groq = False
    hf_available = hf is not None
    
    # Format sources for the prompt
    sources_text = "\n\n".join(
        f"[Source {i}]: {source}" for i, source in enumerate(sources)
    )
    
    for claim in claims:
        if hf_available:
            logger.info("Using local HuggingFace NLI for claim scoring")
            # Try HuggingFace first (local inference, no rate limits)
            try:
                # Combine sources as premise
                combined_premise = " ".join(sources)
                result = await hf.classify(combined_premise, claim.text)
                
                if result:
                    used_hf = True
                    label_str = result.get("label", "neutral")
                    # Map HF labels to our enum
                    label_map = {
                        "entailment": NLILabel.ENTAILMENT,
                        "neutral": NLILabel.NEUTRAL,
                        "contradiction": NLILabel.CONTRADICTION,
                    }
                    label = label_map.get(label_str, NLILabel.NEUTRAL)
                    scores = result.get("scores", {})
                    confidence = max(scores.values()) if scores else 0.5
                    
                    verdicts.append(
                        ClaimVerdict(
                            claim=claim,
                            label=label,
                            confidence=confidence,
                            supporting_source_idx=None,
                            explanation=f"DeBERTa NLI: {label_str}",
                        )
                    )
                    continue
                else:
                    # HF inference failed for this claim
                    logger.warning("Local HuggingFace NLI failed, switching to Groq for remaining claims")
                    hf_available = False
                    remaining = len(claims) - len(verdicts)
                    _notify(
                        f"HuggingFace NLI inference failed. Switching to Groq for remaining {remaining} claims."
                    )
            except Exception as e:
                logger.error(f"HF NLI inference failed: {e}")
                hf_available = False
        
        # Fall back to Groq
        used_groq = True
        user_prompt = f"CLAIM: {claim.text}\n\nSOURCES:\n{sources_text}"
        
        try:
            result = await llm.complete_json(NLI_SYSTEM_PROMPT, user_prompt)
            label_str = result.get("label", "neutral").lower()
            label_map = {
                "entailment": NLILabel.ENTAILMENT,
                "neutral": NLILabel.NEUTRAL,
                "contradiction": NLILabel.CONTRADICTION,
            }
            label = label_map.get(label_str, NLILabel.NEUTRAL)
            
            verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    label=label,
                    confidence=result.get("confidence", 0.5),
                    supporting_source_idx=result.get("supporting_source_idx"),
                    explanation=result.get("explanation", ""),
                )
            )
        except Exception as e:
            logger.error(f"Groq NLI scoring failed: {e}")
            # Default to neutral on error
            verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    label=NLILabel.NEUTRAL,
                    confidence=0.0,
                    explanation="Error during NLI scoring",
                )
            )
    
    # Determine backend string
    if used_hf and used_groq:
        backend = "huggingface+groq"
    elif used_hf:
        backend = "huggingface"
    else:
        backend = "groq"
    
    return verdicts, backend
