#!/usr/bin/env python3
"""
Evaluate YOLO11m model on Taiwan Traffic validation dataset.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluator import ModelEvaluator
from src.config import TAIWAN_CONFIG, RESULTS_DIR


def main():
    """Main evaluation function for Taiwan Traffic dataset."""
    print("\n" + "=" * 60)
    print("TAIWAN TRAFFIC DATASET EVALUATION")
    print("=" * 60)
    
    # Create evaluator
    evaluator = ModelEvaluator(TAIWAN_CONFIG)
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    output_path = RESULTS_DIR / "taiwan_traffic_evaluation.json"
    evaluator.save_results(str(output_path))
    
    print("\n✓ Taiwan Traffic evaluation complete!\n")


if __name__ == "__main__":
    main()
