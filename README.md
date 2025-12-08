# Scooter-YOLO: Training and Evaluation Framework

This repository provides a framework for training and evaluating YOLOv8 models for object detection, with a focus on Transfer Learning (TL) and Continual Learning (CL) strategies. The primary datasets used are the Taiwan Traffic dataset and the COCO dataset.

The framework is designed to experiment with different learning strategies to improve model performance on a specific target domain (Taiwan Traffic) while retaining knowledge from a broader source domain (COCO).

## Project Structure

```
Scooter-YOLO/
├── README.md              # This file
├── docs/                  # Documentation and plans
├── models/                # Pre-trained model weights (e.g., yolo11m.pt)
├── datasets/              # All datasets reside here
│   ├── coco/              # Standard COCO 2017 dataset
│   ├── taiwan-traffic/    # Target domain dataset
│   └── generated/         # YAML and text files for mixed/subset datasets
├── scripts/               # Core Python scripts
│   ├── base_config.py     # Base configuration for paths and hyperparameters
│   ├── callbacks.py       # Custom callbacks for the Ultralytics trainer
│   ├── train.py           # Main script for TL and DIL training methods
│   ├── train_ewc.py       # Script for EWC continual learning method
│   └── evaluate.py        # Script for evaluating a trained model
├── setup/                 # Scripts for initial dataset setup
│   ├── 0_download_setup.py
│   ├── 1_remap_taiwan_labels.py
│   ├── 2_update_data_yaml.py
│   └── 3_prepare_data.py
├── runs/                  # Default output directory for Ultralytics training runs
└── YOLO-Taiwan-Traffic/   # Project-specific output directory for some experiments
```

## Setup

1.  **Install Dependencies:**
    This project relies on PyTorch and the Ultralytics framework. Install the required packages:
    ```bash
    pip install torch torchvision torchaudio
    pip install ultralytics pandas wandb
    ```

2.  **Dataset Preparation:**
    The initial setup of datasets is handled by the scripts in the `setup/` directory. Run them in order:
    ```bash
    python setup/0_download_setup.py
    python setup/1_remap_taiwan_labels.py
    python setup/2_update_data_yaml.py
    python setup/3_prepare_data.py
    ```
    These scripts will download the necessary datasets, remap class labels for compatibility, update the dataset YAML files, and prepare subsets for various training experiments.

3.  **Weights & Biases (WandB) Login:**
    This project uses WandB for experiment tracking. Log in to your account:
    ```bash
    wandb login
    ```

## Training

The core training logic is handled by `scripts/train.py` and `scripts/train_ewc.py`.

### Transfer Learning (TL) and Data-centric Incremental Learning (DIL)

Use `scripts/train.py` with the `--method` argument to select a training strategy.

**Available Methods:**

*   **`tl1` (Baseline Fine-tuning):** Fine-tunes the entire model on the target dataset.
*   **`tl2` (Layer Freezing):** Fine-tunes the model but freezes the first N layers of the backbone. Use the `--freeze` argument to specify how many layers to freeze.
*   **`tl3` (Progressive Unfreezing):** A multi-stage training process where more layers are gradually unfrozen over time.
*   **`dil1` (Random Mixed Data):** Trains the model on a mix of the target dataset and a random 10% subset of the COCO dataset.
*   **`dil2` (Stratified Mixed Data):** Trains the model on a mix of the target dataset and a 10% stratified subset of the COCO dataset.

**Example Command:**

```bash
# Run baseline fine-tuning (TL-1)
python scripts/train.py --method tl1 --epochs 50 --batch 16 --lr0 0.001

# Run layer-freezing (TL-2) by freezing the first 10 layers
python scripts/train.py --method tl2 --freeze 10 --epochs 50

# Run DIL with a stratified COCO subset
python scripts/train.py --method dil2 --epochs 100
```
You can override any hyperparameter from `base_config.py` via command-line arguments.

### Elastic Weight Consolidation (EWC)

Use `scripts/train_ewc.py` to train using the EWC continual learning method, which helps prevent catastrophic forgetting. This script first calculates parameter importance (Fisher matrix) on the source dataset (COCO) and then uses it as a penalty during training on the target dataset.

**Example Command:**
```bash
python scripts/train_ewc.py --ewc_lambda 2000.0 --epochs 50
```
The `--ewc_lambda` parameter controls the strength of the EWC regularization.

## Evaluation

Use `scripts/evaluate.py` to evaluate a trained model's performance on multiple datasets simultaneously. The script will print a summary table and is useful for a final assessment after training.

**Example Command:**
```bash
python scripts/evaluate.py --model runs/detect/tl1_baseline_lr0.001/weights/best.pt
```

## Results and Logging

*   **Training Artifacts:** All training outputs, including model checkpoints (`best.pt`, `last.pt`) and validation results, are saved in the `runs/detect/` directory, organized by experiment name.
*   **Experiment Tracking:** All training and validation metrics, hyperparameters, and logs are automatically tracked using **Weights & Biases (WandB)**. Make sure to set `WANDB_PROJECT` in `base_config.py` or via the environment.
