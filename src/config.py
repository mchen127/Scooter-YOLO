"""
Configuration file for dataset paths and evaluation parameters.
"""
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
YOLO11M_PATH = MODELS_DIR / "yolo11m.pt"

# Dataset paths
DATASETS_DIR = PROJECT_ROOT / "datasets"
COCO_DIR = DATASETS_DIR / "coco"
TAIWAN_TRAFFIC_DIR = DATASETS_DIR / "taiwan-traffic"

# Results directory
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# COCO dataset configuration
COCO_CONFIG = {
    "name": "COCO",
    "data_yaml": "coco.yaml",  # Use built-in COCO config from ultralytics
    "model_path": str(YOLO11M_PATH),
    "task": "detect",
    "split": "val",
    "imgsz": 640,
    "batch": 64,
    "device": "0",  # GPU 0, use "cpu" for CPU
    "save_json": True,
    "save_hybrid": False,
    "conf": 0.001,
    "iou": 0.6,
    "max_det": 300,
    "half": True,  # Use FP16 for faster inference
    "verbose": True,
}

# Taiwan Traffic dataset configuration
TAIWAN_CONFIG = {
    "name": "Taiwan-Traffic",
    "data_yaml": str(TAIWAN_TRAFFIC_DIR / "data.yaml"),
    "model_path": str(YOLO11M_PATH),
    "task": "detect",
    "split": "val",
    "imgsz": 640,
    "batch": 64,
    "device": "0",
    "save_json": True,
    "save_hybrid": False,
    "conf": 0.001,
    "iou": 0.6,
    "max_det": 300,
    "half": True,
    "verbose": True,
}
