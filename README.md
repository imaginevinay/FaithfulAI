# FaithfulAI — RAG Hallucination Detection

**FaithfulAI** evaluates whether a RAG-generated answer faithfully represents its source documents. It detects hallucinated spans, answer drift, and unsupported generalizations, returning a faithfulness score with pinpointed problem spans.

## Features

- **Hallucination Detection**: Identifies claims in RAG answers that contradict or aren't supported by source documents
- **Drift Detection**: Scores answer relevance to the original query
- **Local NLI Inference**: Uses `cross-encoder/nli-deberta-v3-base` locally (no API calls), with Groq fallback
- **Observability**: Datadog tracing + metrics (optional, gracefully degraded if not installed)
- **Structured Logging**: JSON-formatted logs with trace IDs
- **Minimal cost**: Uses Groq free tier for LLM calls only. NLI inference is entirely local (no API costs)
- **Works offline**: Local NLI model means no internet required for core hallucination detection

## Installation

### Prerequisites
- Python 3.10+
- A Groq API key (free tier)
- ~2GB disk space for model weights (downloaded once, cached)
- (Optional) HuggingFace API token (only needed for gated models)

### Setup

```bash
# Clone/download FaithfulAI
cd faithfulai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with all dependencies
pip install -e ".[all]"
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your_groq_api_key_here
HF_API_TOKEN=hf_your_huggingface_token_here
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
LOG_LEVEL=INFO
```

**Required:**
- `GROQ_API_KEY` (or `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` if using different provider)

**Optional:**
- `HF_API_TOKEN` — HuggingFace API token for gated models (not required for public `cross-encoder/nli-deberta-v3-base`)
- `LLM_PROVIDER` — `groq` | `openai` | `anthropic` (default: `groq`)
- `LLM_MODEL` — LLM model name (default: `llama-3.3-70b-versatile`)
- `LOG_LEVEL` — `DEBUG` | `INFO` | `WARNING` | `ERROR` (default: `INFO`)

All environment variables are loaded from `.env` at startup via `python-dotenv`.

## Quick Start

### Example Script

```bash
python example.py
```

This runs 3 evaluation examples:
- Faithful case: Tesla 2023 revenue
- Hallucination case: BCG Deckster launch (date + numbers)
- Faithful case: AWS 2023 revenue

### Python API

```python
import asyncio
from faithfulai import FaithfulAIConfig, FaithfulAIPipeline

async def main():
    config = FaithfulAIConfig()  # Reads from env
    pipeline = FaithfulAIPipeline(config)
    
    result = await pipeline.evaluate(
        query="How many vehicles did Tesla deliver in 2023?",
        rag_answer="Tesla delivered 1.81M vehicles, up 38% YoY.",
        sources=["Tesla delivered 1.81 million vehicles in 2023, up 38% from 2022."],
    )
    
    print(f"Score: {result.score}")
    print(f"Flags: {[f.value for f in result.flags]}")
    print(f"Hallucinated spans: {result.details.hallucinated_spans}")

asyncio.run(main())
```

### FastAPI Server

```bash
uvicorn server.api:app --port 8100
```

Then POST to `/evaluate`:

```bash
curl -X POST http://localhost:8100/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How many vehicles did Tesla deliver in 2023?",
    "rag_answer": "Tesla delivered 1.81M vehicles, up 38% YoY.",
    "sources": ["Tesla delivered 1.81 million vehicles in 2023, up 38% from 2022."]
  }'
```

### MCP Server

```bash
python -m server.mcp_server
```

## Tests

Run all tests:

```bash
pytest tests/
```

Run unit tests only:

```bash
pytest tests/unit/
```

Run integration tests:

```bash
pytest tests/integration/
```

Run evaluation suite:

```bash
python -m eval.run_eval
```

This runs against 10 test cases (5 faithful, 5 hallucinations) and reports accuracy.

## Architecture

### Detectors

1. **Claim Extraction** (`detectors/claim_extractor.py`)
   - Breaks RAG answer into atomic factual claims
   - Uses LLM to identify claims and their source spans

2. **NLI Scoring** (`detectors/nli_scorer.py`)
   - Evaluates each claim against sources using local `cross-encoder/nli-deberta-v3-base` model
   - Falls back to Groq if local inference fails
   - Labels: entailment, contradiction, neutral
   - No API calls, no rate limits, fully deterministic

3. **Span Extraction** (`detectors/span_extractor.py`)
   - Locates hallucinated spans in the original answer
   - Merges overlapping spans

4. **Drift Detection** (`detectors/drift_detector.py`)
   - Scores answer relevance to the original query

### Pipeline

The `FaithfulAIPipeline` orchestrates all detectors:

```
Query + RAG Answer + Sources
  ↓
Extract Claims (LLM)
  ↓
Score Claims (NLI: HF → Groq fallback)
  ↓
Extract Hallucinated Spans (Python)
  ↓
Detect Drift (LLM)
  ↓
Aggregate Results
  ↓
EvalResult (score, flags, spans, explanation)
```

### Output Format

```json
{
  "score": 0.85,
  "flags": ["FAITHFUL"],
  "evidence": ["1.81 million vehicles"],
  "explanation": "[NLI: groq] Answer is faithful to sources. Faithfulness: 1.0, Drift: 0.05",
  "details": {
    "faithfulness_score": 1.0,
    "drift_score": 0.05,
    "hallucinated_spans": [],
    "claim_verdicts": [
      {
        "claim": {"text": "Tesla delivered 1.81M vehicles", ...},
        "label": "entailment",
        "confidence": 0.95,
        "explanation": "Supported"
      }
    ]
  }
}
```

## Configuration Classes

### FaithfulAIConfig

```python
config = FaithfulAIConfig(
    llm_provider="groq",              # or "openai", "anthropic"
    llm_model="llama-3.3-70b-versatile",
    llm_api_key="gsk_...",            # Optional, reads from GROQ_API_KEY if omitted
    hf_api_token="hf_...",            # Optional, reads from HF_API_TOKEN if omitted
    hallucination_threshold=0.5,
    drift_threshold=0.4,
    unsupported_threshold=0.3,
    llm_temperature=0.0,
)
```

All API keys and tokens default to reading from environment variables.

## Observability

### Datadog Tracing

If `ddtrace` is installed, traces are automatically collected:

```bash
pip install ddtrace datadog
```

Traces appear under service `faithfulai` with spans for:
- `faithfulai.evaluate`
- `faithfulai.claim_extraction`
- `faithfulai.nli_scoring`
- `faithfulai.span_extraction`
- `faithfulai.drift_detection`
- `faithfulai.aggregation`

### Structured Logging

All logs are JSON-formatted with trace IDs:

```json
{
  "timestamp": "2024-01-15 14:32:10,123",
  "level": "INFO",
  "logger": "faithfulai",
  "event": "pipeline.evaluate called",
  "trace_id": "a1b2c3d4e5f6"
}
```

## Metrics

When Datadog is enabled, the following metrics are emitted:

- `evaluation.count` (counter)
- `evaluation.latency` (histogram, milliseconds)
- `faithfulness.score` (gauge)
- `drift.score` (gauge)
- `hallucination.detected` (counter)
- `claims.count` (histogram)
- `hallucinated_spans.count` (histogram)
- `nli.supported`, `nli.contradicted`, `nli.neutral` (histograms)
- `nli.backend` (counter with tags)

## Cost

**$0 per evaluation**:
- Groq: 100K tokens/day free (LLM calls only)
- Local NLI: No API costs (model runs locally)
- No GPU required (CPU inference, slower but works)
- Model weights: ~500MB, downloaded once and cached

## Compatibility

- **Python**: 3.10+
- **Async**: Full async/await support
- **LLM Providers**: Groq, OpenAI, Anthropic (zero code changes)
- **Graceful Degradation**: 
  - Skip `ddtrace` install → works identically without tracing
  - Local NLI fails → automatically falls back to Groq for remaining claims
  - Works completely offline after first model download

## Project Structure

```
faithfulai/
├── pyproject.toml              # Dependencies + packaging
├── README.md                   # This file
├── example.py                  # Example usage
├── .env.example                # Environment template
├── faithfulai/
│   ├── __init__.py
│   ├── models.py              # Pydantic models
│   ├── config.py              # Configuration + env loading
│   ├── pipeline.py            # Main orchestrator
│   ├── observability.py       # Datadog tracing
│   ├── logging_config.py      # JSON logging
│   ├── llm_providers/         # LLM integrations
│   │   ├── base.py
│   │   ├── openai_compat.py
│   │   └── anthropic_provider.py
│   └── detectors/             # Hallucination detection
│       ├── claim_extractor.py
│       ├── hf_nli.py
│       ├── nli_scorer.py
│       ├── span_extractor.py
│       └── drift_detector.py
├── server/                     # Delivery surfaces
│   ├── api.py                 # FastAPI
│   └── mcp_server.py          # MCP server
├── tests/
│   ├── conftest.py            # Pytest fixtures
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
└── eval/
    ├── faithfulai_eval_*.json # 10 evaluation cases
    └── run_eval.py            # Evaluation runner
```

## License

MIT
