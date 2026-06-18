"""Span extraction for hallucinated content."""

from __future__ import annotations

import logging
import re

from faithfulai.models import ClaimVerdict, HallucinatedSpan, NLILabel

logger = logging.getLogger("faithfulai")


def extract_hallucinated_spans(
    answer: str,
    verdicts: list[ClaimVerdict],
) -> list[HallucinatedSpan]:
    """Extract spans from answer that correspond to hallucinated claims.
    
    Args:
        answer: The RAG-generated answer text
        verdicts: NLI verdicts for all claims
        
    Returns:
        List of hallucinated spans with positions
    """
    spans: list[HallucinatedSpan] = []
    
    for verdict in verdicts:
        # Skip entailment — those are faithful
        if verdict.label == NLILabel.ENTAILMENT:
            continue
        
        claim = verdict.claim
        span_text = claim.source_span
        
        if not span_text:
            span_text = claim.text
        
        # Try exact match first
        start, end = _find_span(answer, span_text)
        
        if start < 0:
            # Try case-insensitive
            start, end = _find_span_ci(answer, span_text)
        
        if start < 0:
            # Try extracting key tokens
            start, end = _extract_key_tokens(answer, span_text)
        
        if start >= 0:
            spans.append(
                HallucinatedSpan(
                    text=span_text,
                    start=start,
                    end=end,
                    reason=f"{verdict.label.value}: {verdict.explanation}",
                )
            )
    
    # Merge overlapping/adjacent spans
    merged = _merge_spans(spans)
    return merged


def _find_span(text: str, span: str) -> tuple[int, int]:
    """Find span in text (exact match)."""
    pos = text.find(span)
    if pos >= 0:
        return pos, pos + len(span)
    return -1, -1


def _find_span_ci(text: str, span: str) -> tuple[int, int]:
    """Find span in text (case-insensitive)."""
    match = re.search(re.escape(span), text, re.IGNORECASE)
    if match:
        return match.start(), match.end()
    return -1, -1


def _extract_key_tokens(text: str, span: str) -> tuple[int, int]:
    """Extract key tokens from span and find in text."""
    # Patterns for numbers, dates, currency, etc.
    patterns = [
        r"\d+%",           # percentages
        r"\d{4}",          # years
        r"\$[\d,.]+",      # currency
        r"\d[\d,.]+",      # numbers
    ]
    
    tokens = []
    for pattern in patterns:
        matches = re.findall(pattern, span)
        tokens.extend(matches)
    
    if not tokens:
        return -1, -1
    
    # Try to find first token in text
    for token in tokens:
        pos = text.find(token)
        if pos >= 0:
            # Return just the token position
            return pos, pos + len(token)
    
    return -1, -1


def _merge_spans(spans: list[HallucinatedSpan]) -> list[HallucinatedSpan]:
    """Merge overlapping and adjacent spans.
    
    Args:
        spans: List of spans to merge
        
    Returns:
        List of merged spans
    """
    if not spans:
        return []
    
    # Sort by start position
    sorted_spans = sorted(spans, key=lambda s: s.start)
    
    merged: list[HallucinatedSpan] = []
    current = sorted_spans[0]
    
    for next_span in sorted_spans[1:]:
        # Check if overlapping or adjacent (within 1 char)
        if next_span.start <= current.end + 1:
            # Merge: extend end position and combine text
            merged_text = current.text
            if next_span.start > current.end:
                merged_text += current.text[current.end:next_span.start] + next_span.text
            else:
                merged_text += next_span.text
            
            current = HallucinatedSpan(
                text=merged_text,
                start=current.start,
                end=max(current.end, next_span.end),
                reason=f"{current.reason}; {next_span.reason}",
            )
        else:
            # No overlap, save current and start new
            merged.append(current)
            current = next_span
    
    merged.append(current)
    return merged
