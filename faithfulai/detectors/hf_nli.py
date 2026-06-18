"""HuggingFace NLI client using local transformers inference."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger("faithfulai")


class NLIRateLimited(Exception):
    """Rate limit exceeded."""
    pass


class NLINetworkError(Exception):
    """Network connectivity error (DNS, connection refused, timeout)."""
    pass


class NLIServiceUnavailable(Exception):
    """Service unavailable or temporarily down."""
    pass


class LocalTransformerNLI:
    """Local HuggingFace NLI inference using transformers library.
    
    Uses cross-encoder/nli-deberta-v3-base for local inference.
    Avoids hosted API issues (DNS resolution, connection failures, rate limits).
    """
    
    def __init__(self, api_token: Optional[str] = None, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        """Initialize local NLI model.
        
        Args:
            api_token: Optional HF token for accessing gated models. Not required for public models.
            model_name: Model to use for NLI (default: cross-encoder/nli-deberta-v3-base)
        """
        self.model_name = model_name
        self.api_token = api_token
        self._pipeline = None
        self._load_lock = asyncio.Lock()
        logger.info("Local Transformer NLI client initialized", extra={"model": self.model_name})
    
    async def _ensure_model_loaded(self) -> None:
        """Lazy-load the model on first use to avoid blocking at init time."""
        if self._pipeline is not None:
            return
        
        async with self._load_lock:
            # Double-check after acquiring lock
            if self._pipeline is not None:
                return
            
            try:
                from transformers import pipeline
                logger.info(f"Loading local NLI model: {self.model_name}")
                # Run model loading in thread pool to avoid blocking event loop
                loop = asyncio.get_event_loop()
                self._pipeline = await loop.run_in_executor(
                    None,
                    lambda: pipeline(
                        'text-classification',
                        model=self.model_name,
                        return_all_scores=True,
                        function_to_apply='softmax',
                        device=-1,  # CPU only; set to 0 for GPU
                        token=self.api_token,
                    )
                )
                logger.info(f"Local NLI model loaded: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load local NLI model: {type(e).__name__}: {e}", exc_info=True)
                raise NLIServiceUnavailable(f"Failed to load NLI model: {e}") from e
    
    async def classify(
        self, premise: str, hypothesis: str
    ) -> dict | None:
        """Classify premise/hypothesis pair using local model.
        
        Args:
            premise: Source document text (premise)
            hypothesis: Claim to evaluate (hypothesis)
            
        Returns:
            Dict with "label" and "scores", or None if inference fails
            
        Raises:
            NLIServiceUnavailable: If model cannot be loaded
        """
        try:
            await self._ensure_model_loaded()
            
            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._pipeline(f"{premise} </s> {hypothesis}")
            )
            
            # result is a list of dicts: [{"label": "entailment", "score": 0.95}] for cross-encoder models
            # or [{"label": "LABEL_0", "score": 0.95}, ...] for standard classifier models
            if not isinstance(result, list) or len(result) == 0:
                logger.error(f"Unexpected model output format: {result}")
                return None
            
            # Handle cross-encoder output format: single result with label
            if len(result) == 1 and "label" in result[0]:
                label_str = result[0]["label"]
                score = result[0]["score"]
                
                # Normalize label string
                label_map = {
                    "entailment": "entailment",
                    "contradiction": "contradiction",
                    "neutral": "neutral",
                }
                predicted_label = label_map.get(label_str.lower(), "neutral")
                
                return {
                    "label": predicted_label,
                    "scores": {
                        "entailment": score if predicted_label == "entailment" else 0.0,
                        "neutral": score if predicted_label == "neutral" else 0.0,
                        "contradiction": score if predicted_label == "contradiction" else 0.0,
                    }
                }
            
            # Handle standard classifier output format: three results for each label
            if len(result) >= 3:
                scores = {item.get("label", f"LABEL_{i}"): item.get("score", 0.0) for i, item in enumerate(result)}
                
                # Map label indices to names
                label_map = {
                    "LABEL_0": "entailment",
                    "LABEL_1": "neutral",
                    "LABEL_2": "contradiction",
                }
                
                # Find the label with highest score
                max_label = max(scores.keys(), key=lambda k: scores[k])
                predicted_label = label_map.get(max_label, "neutral")
                
                return {
                    "label": predicted_label,
                    "scores": {
                        "entailment": scores.get("LABEL_0", 0.0),
                        "neutral": scores.get("LABEL_1", 0.0),
                        "contradiction": scores.get("LABEL_2", 0.0),
                    }
                }
            
            logger.error(f"Unexpected model output format: {result}")
            return None
        
        except NLIServiceUnavailable:
            raise
        except Exception as e:
            logger.error(f"Local NLI inference failed: {type(e).__name__}: {e}", exc_info=True)
            return None


# Backward compatibility alias
class HuggingFaceNLI(LocalTransformerNLI):
    """Backward-compatible alias for LocalTransformerNLI.
    
    Now uses local inference instead of hosted API.
    """
    pass
