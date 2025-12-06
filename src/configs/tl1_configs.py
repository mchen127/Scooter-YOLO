"""
Experiment configuration templates for TL-1 baseline.

This module provides configuration templates for learning rate ablation.
Epochs are handled by early stopping (patience=5) - no need for epoch ablation.
"""

from pathlib import Path

# Base configuration
BASE_CONFIG = {
    "model_path": "models/yolo11m.pt",
    "data_yaml": "datasets/taiwan-traffic/data.yaml",
    "project": "runs/tl1_baseline",
    "device": "0",
    "batch": 64,
    "imgsz": 640,
    "patience": 5,  # Early stopping - stops when validation plateaus
    "save": True,
    "plots": True,
}

# Learning rate ablation configurations
# Note: epochs=50 is the maximum, early stopping (patience=5) will kick in earlier
LR_ABLATION = {
    "tl1_lr_1e5": {
        **BASE_CONFIG,
        "name": "tl1_lr_1e5",
        "lr0": 1e-5,
        "epochs": 50,  # Max epochs, early stopping will find optimal point
        "description": "Conservative LR - very stable, may converge slowly"
    },
    "tl1_lr_1e4": {
        **BASE_CONFIG,
        "name": "tl1_lr_1e4",
        "lr0": 1e-4,
        "epochs": 50,
        "description": "Moderate LR - recommended baseline"
    },
    "tl1_lr_1e3": {
        **BASE_CONFIG,
        "name": "tl1_lr_1e3",
        "lr0": 1e-3,
        "epochs": 50,
        "description": "Aggressive LR - fast convergence, watch for instability"
    },
}

# Full ablation = LR ablation only (no epoch ablation needed with early stopping)
FULL_ABLATION = LR_ABLATION


def get_config(experiment_name: str) -> dict:
    """Get configuration for a specific experiment."""
    if experiment_name in FULL_ABLATION:
        return FULL_ABLATION[experiment_name]
    raise ValueError(f"Unknown experiment: {experiment_name}")


def list_experiments() -> list:
    """List all available experiment names."""
    return list(FULL_ABLATION.keys())
