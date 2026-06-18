"""Evaluation suite runner for FaithfulAI."""

import asyncio
import json
import os
from pathlib import Path

from faithfulai import FaithfulAIConfig, FaithfulAIPipeline, Flag
from faithfulai.logging_config import setup_logging

setup_logging()


async def run_eval():
    """Run evaluation suite against 10 test cases."""
    eval_dir = Path(__file__).parent
    eval_files = sorted(eval_dir.glob("eval_*.json"))
    
    if not eval_files:
        print(f"No eval files found in {eval_dir}")
        return
    
    config = FaithfulAIConfig()
    pipeline = FaithfulAIPipeline(config)
    
    results = {
        "total": len(eval_files),
        "passed": 0,
        "failed": 0,
        "details": [],
    }
    
    print("=" * 80)
    print("FaithfulAI Evaluation Suite")
    print(f"Running {len(eval_files)} test cases...")
    print("=" * 80)
    
    for eval_file in eval_files:
        with open(eval_file) as f:
            test_case = json.load(f)
        
        test_name = eval_file.stem
        ground_truth = test_case.get("ground_truth_label")
        
        print(f"\n[{test_name}] {test_case.get('question', 'N/A')}")
        print(f"  Expected: {ground_truth}")
        
        try:
            result = await pipeline.evaluate(
                query=test_case["question"],
                rag_answer=test_case["ai_answer"],
                sources=test_case.get("sources", [test_case["document_text"]]),
            )
            
            # Determine if test passed
            passed = False
            if ground_truth == "FAITHFUL":
                passed = Flag.FAITHFUL in result.flags
            elif ground_truth == "HALLUCINATION":
                passed = Flag.HALLUCINATION in result.flags
            
            flag_str = ", ".join(f.value for f in result.flags)
            print(f"  Got:      {flag_str}")
            print(f"  Score:    {result.score:.3f}")
            print(f"  Status:   {'✓ PASS' if passed else '✗ FAIL'}")
            
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
            
            results["details"].append({
                "test": test_name,
                "expected": ground_truth,
                "got": flag_str,
                "score": result.score,
                "passed": passed,
            })
        
        except Exception as e:
            print(f"  Error: {e}")
            results["failed"] += 1
            results["details"].append({
                "test": test_name,
                "expected": ground_truth,
                "got": "ERROR",
                "error": str(e),
                "passed": False,
            })
    
    # Summary
    accuracy = results["passed"] / results["total"] if results["total"] > 0 else 0
    print("\n" + "=" * 80)
    print(f"Results: {results['passed']}/{results['total']} passed")
    print(f"Accuracy: {accuracy*100:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_eval())
