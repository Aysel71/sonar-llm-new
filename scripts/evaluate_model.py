#!/usr/bin/env python3
"""
Evaluation script for trained SONAR-LLM models
Supports both NIAH and QA tasks with RULER metrics
"""

import os
import sys
import json
import argparse
import torch
from typing import List, Dict, Any
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from text_cleaning import clean_text_flag_style


def string_match_all(predictions: List[str], references: List[str]) -> float:
    """RULER metric for NIAH"""
    correct = sum(1 for pred, ref in zip(predictions, references) 
                  if ref.lower() in pred.lower())
    return round((correct / len(predictions)) * 100, 2) if predictions else 0.0


def string_match_part(predictions: List[str], references: List[str]) -> float:
    """RULER metric for QA"""
    correct = sum(1 for pred, ref in zip(predictions, references)
                  if ref.lower() in pred.lower())
    return round((correct / len(predictions)) * 100, 2) if predictions else 0.0


def load_results(results_path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON"""
    with open(results_path, 'r') as f:
        return json.load(f)


def print_evaluation_report(
    task_name: str,
    results: Dict[str, Any],
    metric_name: str = "string_match_all"
):
    """
    Print formatted evaluation report
    
    Args:
        task_name: Name of task (NIAH, QA)
        results: Evaluation results dictionary
        metric_name: Name of RULER metric used
    """
    print(f"\n{'='*80}")
    print(f"{task_name.upper()} EVALUATION REPORT")
    print(f"{'='*80}")
    print(f"RULER Metric: {metric_name}")
    print(f"{'='*80}\n")
    
    # Overall metrics
    print("Overall Metrics:")
    print(f"  Loss:              {results.get('loss', 0):.4f}")
    print(f"  Accuracy:          {results.get('accuracy', 0):.2f}%")
    print(f"  Samples Evaluated: {results.get('num_evaluated', 0)}")
    print()
    
    # Examples
    examples = results.get('examples', [])
    if examples:
        correct = sum(1 for ex in examples if ex.get('correct', False))
        incorrect = len(examples) - correct
        
        print(f"Example Predictions ({len(examples)} shown):")
        print(f"  Correct: {correct}, Incorrect: {incorrect}")
        print()
        
        # Show examples
        for i, ex in enumerate(examples):
            status = "✓ CORRECT" if ex.get('correct', False) else "✗ INCORRECT"
            print(f"{status} Example {i+1}:")
            
            if 'key' in ex and ex['key']:
                print(f"  Key: {ex['key']}")
            if 'question' in ex:
                print(f"  Question: {ex.get('question', 'N/A')}")
            
            print(f"  Expected: {ex.get('expected_value', 'N/A')}")
            print(f"  Prediction: {ex.get('prediction', 'N/A')[:100]}...")
            print()
    
    print("="*80 + "\n")


def compare_tasks(niah_results: Dict, qa_results: Dict):
    """Compare NIAH vs QA results"""
    print(f"\n{'='*80}")
    print("TASK COMPARISON")
    print(f"{'='*80}\n")
    
    print(f"{'Metric':<20} | {'NIAH':<15} | {'QA':<15} | {'Diff':<15}")
    print("-"*80)
    
    niah_acc = niah_results.get('accuracy', 0)
    qa_acc = qa_results.get('accuracy', 0)
    niah_loss = niah_results.get('loss', 0)
    qa_loss = qa_results.get('loss', 0)
    
    print(f"{'Accuracy (%)':<20} | {niah_acc:>15.2f} | {qa_acc:>15.2f} | {qa_acc - niah_acc:>+15.2f}")
    print(f"{'Loss':<20} | {niah_loss:>15.4f} | {qa_loss:>15.4f} | {qa_loss - niah_loss:>+15.4f}")
    print(f"{'Samples':<20} | {niah_results.get('num_evaluated', 0):>15} | {qa_results.get('num_evaluated', 0):>15} | {'':>15}")
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80 + "\n")
    
    if qa_acc > niah_acc:
        diff = qa_acc - niah_acc
        print(f"✓ QA task performed BETTER by {diff:.2f} percentage points")
        print("  Reason: QA has limited question types (easier to memorize)")
    elif niah_acc > qa_acc:
        diff = niah_acc - qa_acc
        print(f"✓ NIAH task performed BETTER by {diff:.2f} percentage points")
        print("  Surprising result - NIAH is typically harder")
    else:
        print(f"⚖ Both tasks achieved similar accuracy: {qa_acc:.2f}%")
    
    if max(niah_acc, qa_acc) < 30:
        print("\n⚠ Warning: Both tasks show low accuracy (<30%)")
        print("Recommendations:")
        print("  • Increase learning rate (try 1e-4)")
        print("  • Train longer (5-10 epochs)")
        print("  • Simplify task (shorter context, fewer unique values)")
        print("  • Consider architectural changes")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained SONAR-LLM models")
    parser.add_argument("--niah_results", type=str,
                       default="../models/niah_model/final_results.json",
                       help="Path to NIAH results")
    parser.add_argument("--qa_results", type=str,
                       default="../models/qa_model/final_results.json",
                       help="Path to QA results")
    parser.add_argument("--compare", action="store_true",
                       help="Compare NIAH vs QA")
    
    args = parser.parse_args()
    
    # Load results
    niah_results = None
    qa_results = None
    
    if os.path.exists(args.niah_results):
        niah_results = load_results(args.niah_results)
        print_evaluation_report("NIAH", niah_results, "string_match_all")
    else:
        print(f"⚠ NIAH results not found: {args.niah_results}")
    
    if os.path.exists(args.qa_results):
        qa_results = load_results(args.qa_results)
        print_evaluation_report("QA", qa_results, "string_match_part")
    else:
        print(f"⚠ QA results not found: {args.qa_results}")
    
    # Compare if both available
    if args.compare and niah_results and qa_results:
        compare_tasks(niah_results, qa_results)


if __name__ == "__main__":
    main()

