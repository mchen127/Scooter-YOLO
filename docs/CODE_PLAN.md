# Code Implementation Plan

## 1. Project Structure

We will use a simplified, script-based structure to minimize complexity.

```
Scooter-YOLO/
├── datasets/           # (Existing) COCO and Taiwan datasets
├── models/             # (Existing) Pretrained weights
├── scripts/            # ALL logic goes here
│   ├── train.py        # Unified script for TL-1, TL-2, TL-3
│   ├── train_discriminative.py # Script for TL-4 (Discriminative LR)
│   ├── train_ewc.py    # Script for DIL-3 (EWC)
│   ├── prepare_data.py # Script for DIL-1, DIL-2 (Data mixing)
│   └── evaluate.py     # Script for dual-domain evaluation
├── utils/              # Helper functions
│   ├── metrics.py      # Custom metric calculation
│   └── plots.py        # Visualization helpers
└── README.md
```

## 2. Script Details

### `scripts/train.py` (TL-1, TL-2, TL-3)
**Purpose:** Handles standard training, layer freezing, and progressive unfreezing.
**Key Features:**
- Uses standard `YOLO` class from `ultralytics`.
- Accepts arguments for `freeze` layers.
- Implements the loop for progressive unfreezing (TL-3).
- **Custom Callback:** Registers `on_fit_epoch_end` callback to run validation on `coco-traffic-subset.yaml` and log metrics to WandB.

### `scripts/train_discriminative.py` (TL-4)
**Purpose:** Implements Discriminative Learning Rates.
**Key Features:**
- Inherits from `ultralytics.models.yolo.detect.DetectionTrainer`.
- Overrides `build_optimizer` to assign different learning rates to backbone, neck, and head parameter groups.
- Uses the standard training loop but with a custom optimizer configuration.

### `scripts/prepare_data.py` (DIL-1, DIL-2)
**Purpose:** Creates mixed datasets for Rehearsal strategies AND prepares the COCO-Traffic-Subset for evaluation.
**Key Features:**
- Reads COCO and Taiwan dataset paths.
- **COCO Subset Generation:**
    - Filters COCO annotations to keep only the 5 traffic classes (Person, Bicycle, Car, Motorcycle, Truck).
    - Creates a `coco-traffic-subset.yaml` for validation.
- **Mixed Dataset Generation:**
    - Samples a subset of COCO images (Random or Stratified).
    - Generates a new `data.yaml` pointing to the mixed dataset.
- Does NOT copy images; just creates text files with image paths.

### `scripts/train_ewc.py` (DIL-3)
**Purpose:** Implements Elastic Weight Consolidation.
**Key Features:**
- **Phase 1 (Fisher Calculation):**
    - Loads a model pretrained on COCO.
    - Runs a pass over COCO validation data to compute the Fisher Information Matrix (diagonal approximation).
    - Saves the Fisher matrix and optimal parameter values.
- **Phase 2 (Training with Penalty):**
    - Inherits from `DetectionTrainer`.
    - Overrides the `criterion` (loss function) or `train_step`.
    - Adds the EWC penalty term: `loss = original_loss + lambda * sum(fisher * (param - old_param)^2)`.

### `scripts/evaluate.py`
**Purpose:** Evaluates a model on BOTH Taiwan and COCO datasets.
**Key Features:**
- Loads a trained model.
- Runs validation on `datasets/taiwan-traffic/data.yaml`.
- Runs validation on `datasets/coco/data.yaml` (or a subset).
- Computes "Forgetting" metric.
- Generates comparison plots.

## 3. Why Wrapped API vs. Custom Loop?

We prefer using the **Wrapped API** (or inheriting from `DetectionTrainer`) because:

1.  **Infrastructure:** It automatically handles:
    - Distributed Data Parallel (DDP) for multi-GPU.
    - Mixed Precision (AMP) for speed.
    - Logging (WandB, Tensorboard, CSV).
    - Data augmentation pipelines (Mosaic, Mixup, etc.).
    - Model saving and checkpointing.
2.  **Reliability:** Writing a raw PyTorch loop from scratch is error-prone. You might miss `model.eval()`, forget `optimizer.zero_grad()`, or mess up the gradient accumulation.
3.  **Compatibility:** It ensures our models remain compatible with the rest of the Ultralytics ecosystem (exporting to ONNX, running inference, etc.).

**Exception:**
For **EWC (DIL-3)**, we *must* intervene in the loss calculation. However, instead of writing a raw loop, we will **inherit from the Trainer class** and override specific methods (`loss`, `train_step`). This gives us the best of both worlds: custom logic where we need it, and robust infrastructure for everything else.
