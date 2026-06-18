"""Data models for FaithfulAI hallucination detection."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Input ────────────────────────────────────────────────────────────
class EvalRequest(BaseModel):
    """Input to the hallucination detection pipeline."""

    query: str = Field(..., description="The user's original question")
    rag_answer: str = Field(..., description="The RAG-generated answer to evaluate")
    sources: list[str] = Field(
        ..., description="Source documents / citations the RAG system used"
    )


# ── Internal intermediates ───────────────────────────────────────────
class Claim(BaseModel):
    """An atomic factual claim extracted from the RAG answer."""

    text: str = Field(..., description="The claim text")
    source_span: str = Field(
        "", description="Substring in rag_answer this claim came from"
    )
    start: int = Field(-1, description="Start char offset in rag_answer")
    end: int = Field(-1, description="End char offset in rag_answer")


class NLILabel(str, Enum):
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


class ClaimVerdict(BaseModel):
    """NLI verdict for a single claim against all sources."""

    claim: Claim
    label: NLILabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    supporting_source_idx: Optional[int] = Field(
        None, description="Index of the source that best supports this claim"
    )
    explanation: str = ""


# ── Flags ────────────────────────────────────────────────────────────
class Flag(str, Enum):
    FAITHFUL = "FAITHFUL"
    HALLUCINATION = "HALLUCINATION"
    DRIFT = "DRIFT"
    UNSUPPORTED = "UNSUPPORTED"


class HallucinatedSpan(BaseModel):
    """A span in the RAG answer that is hallucinated."""

    text: str
    start: int = -1
    end: int = -1
    reason: str = ""


# ── Output ───────────────────────────────────────────────────────────
class EvalDetails(BaseModel):
    """Detailed breakdown of the evaluation."""

    faithfulness_score: float = Field(..., ge=0.0, le=1.0)
    hallucinated_spans: list[HallucinatedSpan] = []
    drift_score: float = Field(0.0, ge=0.0, le=1.0)
    claim_verdicts: list[ClaimVerdict] = []
    test_cases_passed: int = 0
    test_cases_total: int = 0


class EvalResult(BaseModel):
    """Final output of the FaithfulAI pipeline."""

    score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall faithfulness score"
    )
    evidence: list[str] = Field(
        default_factory=list, description="Supporting spans from sources"
    )
    flags: list[Flag] = Field(default_factory=list, description="Detection flags")
    explanation: str = Field("", description="Human-readable explanation")
    details: EvalDetails = Field(
        default_factory=lambda: EvalDetails(faithfulness_score=0.0)
    )
