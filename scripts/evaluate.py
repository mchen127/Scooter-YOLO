import argparse
from ultralytics import YOLO
import pandas as pd
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.base_config import config


def evaluate_model(
    model_path, taiwan_yaml, coco_subset_yaml, coco_full_yaml=None, device="0"
):
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
        except Exception as e:
            print(f"Skipping Full COCO eval: {e}")
            results["COCO Full mAP50-95"] = 0.0

    # Print Summary
    print("\n" + "=" * 40)
    print(f"RESULTS FOR {Path(model_path).name}")
    print("=" * 40)
    df = pd.DataFrame([results])
    print(df.to_string(index=False))
    print("=" * 40)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument(
        "--taiwan_data", type=str, default=config.data
    )
    parser.add_argument(
        "--coco_subset", type=str, default=config.coco_traffic_subset_yaml
    )
    parser.add_argument(
        "--coco_full",
        type=str,
        default=None,
        help="Path to full coco.yaml if available",
    )
    parser.add_argument("--device", type=str, default=config.device)

    args = parser.parse_args()

    evaluate_model(
        args.model, args.taiwan_data, args.coco_subset, args.coco_full, args.device
    )
