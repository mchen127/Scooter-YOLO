# Domain Shift and Catastrophic Forgetting: Ablation Study Plan

## Research Objective

Conduct a comprehensive ablation study on methods for solving **Domain Shift** and **Catastrophic Forgetting** when adapting YOLO11m from COCO (Source Domain) to Taiwan Traffic Dataset (Target Domain).

### Current Baseline Performance
- **COCO (Source Domain)**: mAP50-95 = **51.3%** (traffic subset: 51.5%)
- **Taiwan Traffic (Target Domain)**: mAP50-95 = **31.4%** (traffic subset: 31.4%)
- **Performance Gap**: **~20%** drop confirms significant distribution shift

---

## Phase 1: Transfer Learning (Solving Domain Shift)

**Goal:** Maximize performance on Taiwan Traffic Dataset  
**Constraint:** Catastrophic forgetting of COCO is acceptable

### Module TL-1: Standard Fine-Tuning (The Baseline) 🟢

**Difficulty:** Easy (Built-in)

**Method:** Update all weights (Backbone + Head) on Taiwan dataset

**Implementation:**
```bash
yolo train \
  model=yolo11m.pt \
  data=datasets/taiwan-traffic/data.yaml \
  epochs=30 \
  imgsz=640 \
  batch=16 \
  lr0=0.01 \
  device=0
```

---

### Module TL-2: Layer Freezing (Feature Extraction) 🟢

**Difficulty:** Easy (Built-in argument)

**Method:** Freeze backbone layers (0-10), only train detection head (11+)

**Rationale:** Low-level features (edges, shapes) are domain-invariant. Only retrain task-specific layers.

**Implementation:**
```bash
yolo train \
  model=yolo11m.pt \
  data=datasets/taiwan-traffic/data.yaml \
  epochs=30 \
  freeze=10 \
  lr0=0.01 \
  device=0
```

---

### Module TL-3: Progressive Unfreezing 🟢

**Difficulty:** Easy (Scripted)

**Method:** Start with frozen backbone → gradually unfreeze layers during training

**Strategy:**
1. Epochs 0-10: Freeze backbone (layers 0-10)
2. Epochs 11-20: Unfreeze last 5 backbone layers (6-10)
3. Epochs 21-30: Unfreeze all layers

**Implementation:**
```python
# Requires custom training script
from ultralytics import YOLO

model = YOLO('yolo11m.pt')

# Phase 1: Train head only
model.train(data='taiwan-traffic/data.yaml', epochs=10, freeze=10)

# Phase 2: Unfreeze upper backbone
model.train(data='taiwan-traffic/data.yaml', epochs=10, freeze=5, resume=True)

# Phase 3: Fine-tune all
model.train(data='taiwan-traffic/data.yaml', epochs=10, freeze=0, resume=True)
```

### Module TL-4: Discriminative Learning Rates 🟡
**Difficulty:** Medium (Custom optimizer)

**Method:** Apply different learning rates to different layers
* Backbone: LR = 1e-5 (small changes)
* Neck: LR = 1e-4 (moderate changes)
* Head: LR = 1e-3 (large changes)

**Implementation:**
```python
# Requires custom callback or trainer
# We will implement this in scripts/train_discriminative.py
```

---

## Phase 2: Domain Incremental Learning (Solving Forgetting)

### Module DIL-1: Rehearsal with Mixed Dataset 🟢
**Difficulty:** Easy (Data configuration)

**Method:** Train on Taiwan + subset of COCO (10% default)

**Rationale:** "Review old notes" - periodically see old data to maintain memory

**Implementation:**

**Step 1:** Create COCO subset sampler
```python
# src/data/coco_sampler.py
python scripts/create_coco_subset.py \
  --source datasets/coco \
  --output datasets/coco_subset_10pct \
  --ratio 0.1 \
  --sampling random
```

**Step 2:** Create mixed dataset YAML
```yaml
# datasets/mixed_10pct.yaml
train: 
  - datasets/taiwan-traffic/train
  - datasets/coco_subset_10pct/train
val:
  - datasets/taiwan-traffic/val
  - datasets/coco/val  # Evaluate on both
```

**Step 3:** Train on mixed dataset
```bash
yolo train \
  model=yolo11m.pt \
  data=datasets/mixed_10pct.yaml \
  epochs=30 \
  batch=16 \
  device=0
```

---

### Module DIL-2: Rehearsal with Stratified Sampling 🟢

**Difficulty:** Easy (Better data sampling)

**Method:** Intelligently sample COCO subset focusing on traffic-related classes

**Strategy:**
- 70% from traffic classes (person, bicycle, car, motorcycle, truck)
- 30% from diverse classes (maintain general knowledge)
- Total buffer: 5,000-10,000 images

**Implementation:**
```python
python scripts/create_coco_subset.py \
  --source datasets/coco \
  --output datasets/coco_stratified_10pct \
  --ratio 0.1 \
  --sampling stratified \
  --priority-classes [0,1,2,3,7] \  # Traffic classes
  --priority-weight 0.7
```

---

### Module DIL-3: Elastic Weight Consolidation (EWC) 🔴

**Difficulty:** High (Custom training loop)

**Status:** ✅ Priority Goal

**Method:** 
- Compute Fisher Information Matrix on COCO
- Add regularization penalty for important weights
- Loss = Task_Loss + λ × EWC_Penalty

**Implementation:** Requires custom PyTorch training loop overriding the standard YOLO trainer.

---

### Module DIL-4: LoRA / Adapter-Based Training 🟡

**Difficulty:** Medium-High (Library integration)

**Method:** Freeze pre-trained model, add small trainable adapter modules

**Rationale:** "Add-on memory" - original knowledge remains frozen

**Implementation:** Requires custom PyTorch training loop overriding the standard YOLO trainer.
