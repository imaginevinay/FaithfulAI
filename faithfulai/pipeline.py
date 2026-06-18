"""Pipeline orchestrator for FaithfulAI evaluation."""

from __future__ import annotations

import time
import logging

from faithfulai.config import FaithfulAIConfig
from faithfulai.models import (
    EvalResult,
    EvalDetails,
    Flag,
    NLILabel,
    ClaimVerdict,
)
from faithfulai.llm_providers import create_provider
from faithfulai.llm_providers.base import BaseLLMProvider
from faithfulai.detectors import (
    extract_claims,
    score_claims,
    extract_hallucinated_spans,
    detect_drift,
    HuggingFaceNLI,
)
from faithfulai.observability import trace_span, emit_counter, emit_gauge, emit_histogram
from faithfulai.logging_config import new_trace_id

logger = logging.getLogger("faithfulai")


class FaithfulAIPipeline:
    """Main evaluation pipeline."""

    def __init__(self, config: FaithfulAIConfig):
        """Initialize the pipeline.
        
        Args:
            config: FaithfulAI configuration
        """
        self.config = config
        self._llm: BaseLLMProvider | None = None
        self._hf: HuggingFaceNLI | None = None

    @property
    def llm(self) -> BaseLLMProvider:
        """Lazy-init LLM provider."""
        if self._llm is None:
            self._llm = create_provider(self.config)
        return self._llm

    @property
    def hf(self) -> HuggingFaceNLI | None:
        """Lazy-init HuggingFace NLI if token available."""
        if self._hf is None:
            hf_token = self.config.get_hf_token()
            if hf_token:
                self._hf = HuggingFaceNLI(hf_token, self.config.hf_nli_model)
            else:
                logger.info("No HF_API_TOKEN configured; skipping HuggingFace NLI")
        return self._hf

    async def evaluate(
        self, query: str, rag_answer: str, sources: list[str]
    ) -> EvalResult:
        """Evaluate a RAG answer for hallucination.
        
        Args:
            query: User's original question
            rag_answer: RAG-generated answer
            sources: Source documents
            
        Returns:
            Evaluation result with score, flags, and explanations
        """
        trace_id = new_trace_id()
        logger.info(
            "pipeline.evaluate called",
            extra={
                "query_len": len(query),
                "source_count": len(sources),
                "trace_id": trace_id,
            },
        )

        async with trace_span(
            "faithfulai.evaluate",
            query_length=len(query),
            source_count=len(sources),
        ):
            start = time.perf_counter()

            # Step 1: Extract claims
            async with trace_span(
                "faithfulai.claim_extraction",
                llm_provider=self.config.llm_provider.value,
            ):
                claims = await extract_claims(self.llm, query, rag_answer)
                logger.info(f"claims extracted: {len(claims)}")
            
            emit_histogram("claims.count", len(claims))

            # Handle no claims case
            if not claims:
                logger.info("No claims extracted, answer deemed faithful")
                result = EvalResult(
                    score=1.0,
                    flags=[Flag.FAITHFUL],
                    explanation="No claims to evaluate",
                    details=EvalDetails(
                        faithfulness_score=1.0,
                        drift_score=0.0,
                        claim_verdicts=[],
                    ),
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                emit_counter("evaluation.count")
                emit_histogram("evaluation.latency", elapsed_ms)
                return result

            # Step 2: Score claims with NLI
            async with trace_span("faithfulai.nli_scoring"):
                verdicts, nli_backend = await score_claims(
                    claims, sources, self.llm, self.hf
                )
                supported = sum(1 for v in verdicts if v.label == NLILabel.ENTAILMENT)
                contradicted = sum(
                    1 for v in verdicts if v.label == NLILabel.CONTRADICTION
                )
                neutral = sum(1 for v in verdicts if v.label == NLILabel.NEUTRAL)
                logger.info(
                    f"NLI scoring complete: backend={nli_backend}, supported={supported}, contradicted={contradicted}, neutral={neutral}"
                )

            emit_counter("nli.backend", tags=[f"backend:{nli_backend}"])
            emit_histogram("nli.supported", supported)
            emit_histogram("nli.contradicted", contradicted)
            emit_histogram("nli.neutral", neutral)

            # Step 3: Extract hallucinated spans
            async with trace_span("faithfulai.span_extraction"):
                hallucinated_spans = extract_hallucinated_spans(rag_answer, verdicts)
                logger.info(f"hallucinated spans extracted: {len(hallucinated_spans)}")

            emit_histogram("hallucinated_spans.count", len(hallucinated_spans))
            if hallucinated_spans:
                emit_counter("hallucination.detected")

            # Step 4: Detect drift
            async with trace_span("faithfulai.drift_detection"):
                drift_score, drift_explanation = await detect_drift(
                    self.llm, query, rag_answer
                )
                logger.info(f"drift score: {drift_score}")

            emit_gauge("drift.score", drift_score)

            # Step 5: Aggregate results
            async with trace_span("faithfulai.aggregation"):
                result = self._aggregate(
                    verdicts,
                    hallucinated_spans,
                    drift_score,
                    drift_explanation,
                    nli_backend,
                )
                logger.info(
                    f"evaluation complete: score={result.score}, flags={[f.value for f in result.flags]}"
                )

            # Emit final metrics
            elapsed_ms = (time.perf_counter() - start) * 1000
            emit_counter("evaluation.count")
            emit_histogram("evaluation.latency", elapsed_ms)
            emit_gauge("faithfulness.score", result.details.faithfulness_score)
            emit_gauge("overall.score", result.score)

            return result

    def _aggregate(
        self,
        verdicts: list[ClaimVerdict],
        hallucinated_spans,
        drift_score: float,
        drift_explanation: str,
        nli_backend: str,
    ) -> EvalResult:
        """Aggregate evaluation results.
        
        Args:
            verdicts: NLI verdicts for all claims
            hallucinated_spans: Extracted hallucinated spans
            drift_score: Relevance drift score
            drift_explanation: Explanation for drift score
            nli_backend: Backend used for NLI ("huggingface", "groq", or "huggingface+groq")
            
        Returns:
            Final evaluation result
        """
        total = len(verdicts)
        supported = sum(1 for v in verdicts if v.label == NLILabel.ENTAILMENT)
        contradicted = sum(1 for v in verdicts if v.label == NLILabel.CONTRADICTION)
        unsupported = sum(1 for v in verdicts if v.label == NLILabel.NEUTRAL)

        # Calculate faithfulness score
        faithfulness_score = round(supported / total, 3) if total > 0 else 1.0
        
        # Overall score: 80% faithfulness, 20% relevance
        overall_score = round(
            faithfulness_score * 0.8 + (1.0 - drift_score) * 0.2, 3
        )

        # Determine flags
        flags = []
        if contradicted > 0:
            flags.append(Flag.HALLUCINATION)
        if unsupported > 0 and unsupported / total > self.config.unsupported_threshold:
            flags.append(Flag.UNSUPPORTED)
        if drift_score > self.config.drift_threshold:
            flags.append(Flag.DRIFT)
        if not flags:
            flags.append(Flag.FAITHFUL)

        # Build explanation
        explanation = f"[NLI: {nli_backend}] "
        if Flag.HALLUCINATION in flags:
            explanation += f"Found {contradicted} contradicted claim(s). "
        if Flag.UNSUPPORTED in flags:
            explanation += f"Found {unsupported} unsupported claim(s). "
        if Flag.DRIFT in flags:
            explanation += f"Answer drift detected (score: {drift_score}). "
        if Flag.FAITHFUL in flags:
            explanation += "Answer is faithful to sources. "
        
        explanation += f"Faithfulness: {faithfulness_score}, Drift: {drift_score}"

        return EvalResult(
            score=overall_score,
            evidence=[v.claim.source_span for v in verdicts if v.label == NLILabel.ENTAILMENT],
            flags=flags,
            explanation=explanation,
            details=EvalDetails(
                faithfulness_score=faithfulness_score,
                hallucinated_spans=hallucinated_spans,
                drift_score=drift_score,
                claim_verdicts=verdicts,
                test_cases_passed=supported,
                test_cases_total=total,
            ),
        )
