import argparse
import json
from ultralytics import YOLO
import pandas as pd
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.base_config import config


def evaluate_model(
    model_path,
    taiwan_yaml,
    coco_subset_yaml,
    coco_full_yaml=None,
    device="0",
    output_path=None,
):
    if "ewc" not in model_path:
        return
        
    print(f"Evaluating model: {model_path}")
    model = YOLO(model_path)

    results = {}

    # 1. Evaluate on Taiwan Traffic (Target)
    print("\n=== Evaluating on Taiwan Traffic ===")
    metrics_taiwan = model.val(data=taiwan_yaml, split="test", device=device)
    results["Taiwan mAP50-95"] = metrics_taiwan.box.map
    results["Taiwan mAP50"] = metrics_taiwan.box.map50

    # 2. Evaluate on COCO Traffic Subset (Source - Relevant)
    print("\n=== Evaluating on COCO Traffic Subset ===")
    metrics_coco_sub = model.val(data=coco_subset_yaml, split="val", device=device)
    results["COCO Subset mAP50-95"] = metrics_coco_sub.box.map
    results["COCO Subset mAP50"] = metrics_coco_sub.box.map50

    # 3. Evaluate on Full COCO (Source - General) - Optional
    if coco_full_yaml:
        print("\n=== Evaluating on Full COCO ===")
        try:
            metrics_coco = model.val(data=coco_full_yaml, split="val", device=device)
            results["COCO Full mAP50-95"] = metrics_coco.box.map
            results["COCO Full mAP50"] = metrics_coco.box.map50
        except Exception as e:
            print(f"Skipping Full COCO eval: {e}")
            results["COCO Full mAP50-95"] = 0.0
            results["COCO Full mAP50"] = 0.0

    # Print Summary
    print("\n" + "=" * 40)
    print(f"RESULTS FOR {Path(model_path).name}")
    print("=" * 40)
    df = pd.DataFrame([results])
    print(df.to_string(index=False))
    print("=" * 40)

    # Save results to JSON if output path is provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing results if file exists
        existing_data = []
        if output_path.exists():
            try:
                with open(output_path, "r") as f:
                    existing_data = json.load(f)
                # Handle old format (single dict) by converting to list
                if isinstance(existing_data, dict):
                    existing_data = [existing_data]
            except (json.JSONDecodeError, Exception):
                existing_data = []

        # Create new result entry
        new_entry = {"model_path": str(model_path), "results": results}

        # Append new results
        existing_data.append(new_entry)

        with open(output_path, "w") as f:
            json.dump(existing_data, f, indent=4)

        print(
            f"\nResults saved to: {output_path} ({len(existing_data)} model(s) total)"
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--taiwan_data", type=str, default=config.data)
    parser.add_argument(
        "--coco_subset", type=str, default=config.coco_traffic_subset_yaml
    )
    parser.add_argument(
        "--coco_full",
        type=str,
        default="coco.yaml",
        help="Path to full coco.yaml if available",
    )
    parser.add_argument("--device", type=str, default=config.device)
    parser.add_argument(
        "--output",
        type=str,
        default="results/all_results.json",
        help="Path to save results JSON file (optional)",
    )

    args = parser.parse_args()

    evaluate_model(
        args.model,
        args.taiwan_data,
        args.coco_subset,
        args.coco_full,
        args.device,
        args.output,
    )
