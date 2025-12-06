# Scooter-YOLO

YOLO11m evaluation framework for object detection on COCO and Taiwan Traffic datasets.

## Project Structure

```
Scooter-YOLO/
├── src/                    # Source modules
│   ├── config.py          # Configuration and paths
│   ├── evaluator.py       # Core evaluation logic
│   └── utils.py           # Helper functions
├── scripts/               # Evaluation scripts
│   ├── evaluate_coco.py
│   └── evaluate_taiwan_traffic.py
├── models/                # Model weights
│   └── yolo11m.pt
├── datasets/              # Datasets
│   ├── coco/
│   └── taiwan-traffic/
└── results/               # Evaluation results (auto-generated)
```

## Setup

1. Install dependencies:
```bash
pip install ultralytics
```

2. Download datasets (already done if you followed setup)

## Usage

### Evaluate on COCO

```bash
python scripts/evaluate_coco.py
```

### Evaluate on Taiwan Traffic

```bash
python scripts/evaluate_taiwan_traffic.py
```

## Results

Evaluation results are saved to `results/` directory in JSON format:
- `coco_evaluation.json` - COCO validation metrics
- `taiwan_traffic_evaluation.json` - Taiwan Traffic validation metrics

## Datasets

### COCO 2017
- 80 object classes
- Standard benchmark dataset
- Model is pre-trained on this dataset

### Taiwan Traffic
- 5 classes: bike, car, motor, person, truck
- All classes map to COCO equivalents
- Taiwan-specific traffic scenarios