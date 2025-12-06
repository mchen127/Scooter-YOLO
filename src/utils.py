"""
Utility functions for evaluation and result processing.
"""
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def save_results_json(results: Dict[str, Any], output_path: Path) -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Dictionary containing evaluation metrics
        output_path: Path to save the JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp
    results["timestamp"] = datetime.now().isoformat()
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {output_path}")


def format_metrics(metrics: Dict[str, Any]) -> str:
    """
    Format metrics dictionary into a readable string.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        
    Returns:
        Formatted string representation of metrics
    """
    lines = ["=" * 60]
    lines.append("EVALUATION METRICS")
    lines.append("=" * 60)
    
    # Overall metrics
    if "metrics/mAP50(B)" in metrics:
        lines.append(f"mAP@0.5:     {metrics['metrics/mAP50(B)']:.4f}")
    if "metrics/mAP50-95(B)" in metrics:
        lines.append(f"mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.4f}")
    if "metrics/precision(B)" in metrics:
        lines.append(f"Precision:   {metrics['metrics/precision(B)']:.4f}")
    if "metrics/recall(B)" in metrics:
        lines.append(f"Recall:      {metrics['metrics/recall(B)']:.4f}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def print_summary(config: Dict[str, Any], results: Dict[str, Any]) -> None:
    """
    Print evaluation summary.
    
    Args:
        config: Configuration dictionary
        results: Results dictionary from evaluation
    """
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY: {config['name']}")
    print("=" * 60)
    print(f"Model:       {Path(config['model_path']).name}")
    print(f"Dataset:     {config['data_yaml']}")
    print(f"Image Size:  {config['imgsz']}")
    print(f"Batch Size:  {config['batch']}")
    print(f"Device:      {config['device']}")
    print("=" * 60)
    
    if results and "results_dict" in results:
        print(format_metrics(results["results_dict"]))
    
    # Print traffic subset mAP for fair comparison
    if results and "traffic_subset_mAP" in results:
        subset_info = results["traffic_subset_mAP"]
        print("\n" + "=" * 60)
        print("TRAFFIC CLASSES SUBSET mAP (Fair Comparison)")
        print("=" * 60)
        print(f"Classes: {', '.join(subset_info['classes'])}")
        print(f"mAP@0.5:0.95: {subset_info['mAP50-95']:.4f}")
        print("=" * 60)
