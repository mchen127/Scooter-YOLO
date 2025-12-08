import sys
from pathlib import Path
import wandb
from ultralytics import YOLO

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.base_config import config

# Constants
COCO_EVAL_YAML = config.coco_traffic_subset_yaml


def on_fit_epoch_end(trainer):
    """
    Callback to evaluate on COCO subset after each epoch.
    """
    print(
        f"[Callback] Evaluating on COCO Traffic Subset (Epoch {trainer.epoch + 1})..."
    )

    # Get the model from trainer and create YOLO wrapper
    try:
        # Load the model from the last checkpoint
        # We cannot pass trainer.model directly as it causes an OSError (filename too long)
        model_path = Path(trainer.save_dir) / "weights" / "last.pt"
        if model_path.exists():
            val_model = YOLO(model_path)
        else:
            print(
                f"Warning: Checkpoint not found at {model_path}. Skipping COCO evaluation."
            )
            return
        coco_metrics = val_model.val(
            data=COCO_EVAL_YAML, split="val", verbose=False, save=False
        )

        # Update trainer metrics to consolidate logging
        map50_95 = coco_metrics.box.map
        map50 = coco_metrics.box.map50

        if hasattr(trainer, "metrics"):
            trainer.metrics.update(
                {
                    "metrics/cocosubset_mAP50-95": map50_95,
                    "metrics/cocosubset_mAP50": map50,
                }
            )
        else:
            # Fallback for safety (though trainer.metrics should exist)
            print(
                "Warning: trainer.metrics not found. Metrics might not be logged properly."
            )

        print(f"COCO Subset mAP50-95: {map50_95:.4f}")

    except Exception as e:
        print(f"Error during COCO evaluation callback: {e}")


def on_train_start(trainer):
    """
    Callback to evaluate on Taiwan dataset BEFORE training starts (Epoch 0).
    """
    print(
        "[Callback] Evaluating Pretrained Model on Taiwan Traffic Dataset (Epoch 0)..."
    )

    try:
        # Load the initial model to evaluate on Taiwan dataset before training
        initial_model = YOLO(trainer.args.model)
        metrics = initial_model.val(
            data=trainer.args.data, split="val", verbose=False, save=False
        )
        results = metrics.results_dict
        print("Initial Taiwan Evaluation Results:", results)

        # Also evaluate on COCO subset at epoch 0
        print(
            "[Callback] Evaluating Pretrained Model on COCO Traffic Subset (Epoch 0)..."
        )
        coco_metrics = initial_model.val(
            data=COCO_EVAL_YAML, split="val", verbose=False, save=False
        )
        coco_results = coco_metrics.results_dict
        print(
            f"Initial COCO Subset mAP50-95: {coco_results['metrics/mAP50-95(B)']:.4f}"
        )
        print(f"Initial COCO Subset mAP50: {coco_results['metrics/mAP50(B)']:.4f}")
        if wandb.run is not None:
            wandb.log(
                {
                    "metrics/mAP50-95(B)": results["metrics/mAP50-95(B)"],
                    "metrics/mAP50(B)": results["metrics/mAP50(B)"],
                    "metrics/precision(B)": results["metrics/precision(B)"],
                    "metrics/recall(B)": results["metrics/recall(B)"],
                    "metrics/cocosubset_mAP50-95": coco_results["metrics/mAP50-95(B)"],
                    "metrics/cocosubset_mAP50": coco_results["metrics/mAP50(B)"],
                }
            )

    except Exception as e:
        print(f"Error during initial evaluation: {e}")
