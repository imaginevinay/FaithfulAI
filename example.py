"""Example usage of FaithfulAI."""

import asyncio
import json
import os

from dotenv import load_dotenv

from faithfulai import FaithfulAIConfig, FaithfulAIPipeline
from faithfulai.logging_config import setup_logging

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()


async def main():
    """Run example evaluations."""
    
    # Initialize config and pipeline
    config = FaithfulAIConfig()
    pipeline = FaithfulAIPipeline(config)
    
    # Test cases
    test_cases = [
        {
            "name": "FAITHFUL case - Tesla revenue",
            "query": "How many vehicles did Tesla deliver in 2023?",
            "answer": "Tesla delivered 1.81 million vehicles in 2023, representing a 38% year-over-year increase.",
            "sources": ["Tesla delivered 1.81 million vehicles in 2023, up 38% from 2022."],
        },
        {
            "name": "HALLUCINATION case - Deckster launch",
            "query": "When was BCG Deckster launched and what was its impact?",
            "answer": "BCG launched Deckster in 2021, reducing deck creation time by 50% across 500 engagements.",
            "sources": ["BCG launched Deckster in 2022, cutting deck creation time by 30% across 200 pilot engagements."],
        },
        {
            "name": "FAITHFUL case - AWS",
            "query": "What was AWS revenue in 2023?",
            "answer": "AWS generated $90.8 billion in revenue during 2023, which was a 13% increase compared to the prior year.",
            "sources": ["Amazon Web Services (AWS) reported $90.8 billion in revenue for 2023, up 13% year-over-year."],
        },
    ]
    
    print("=" * 80)
    print("FaithfulAI RAG Hallucination Detection - Example Run")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[Test {i}/{len(test_cases)}] {test_case['name']}")
        print("-" * 80)
        
        try:
            result = await pipeline.evaluate(
                query=test_case["query"],
                rag_answer=test_case["answer"],
                sources=test_case["sources"],
            )
            
            print(f"Query:  {test_case['query']}")
            print(f"Answer: {test_case['answer']}")
            print(f"\nResults:")
            print(f"  Overall Score:      {result.score:.3f}")
            print(f"  Faithfulness:       {result.details.faithfulness_score:.3f}")
            print(f"  Drift Score:        {result.details.drift_score:.3f}")
            print(f"  Flags:              {[f.value for f in result.flags]}")
            
            if result.details.hallucinated_spans:
                print(f"  Hallucinated Spans: {len(result.details.hallucinated_spans)}")
                for span in result.details.hallucinated_spans:
                    print(f"    - '{span.text}': {span.reason}")
            
            print(f"  Explanation:        {result.explanation}")
            
        except Exception as e:
            print(f"Error evaluating test case: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Example run complete")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
