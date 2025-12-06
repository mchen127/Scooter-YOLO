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

**Hypothesis:**
- ✅ Highest accuracy on Taiwan
- ❌ Highest forgetting on COCO
- ⚠️ Risk of overfitting on small Taiwan dataset

**Expected Results:**
- Taiwan mAP50-95: 45-50% (target: +15% improvement)
- COCO mAP50-95: 30-40% (expect significant drop)

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

**Hypothesis:**
- ⚡ Faster training (fewer parameters)
- 📉 Lower Taiwan accuracy than TL-1
- 🛡️ Better generalization (less overfitting)
- 💾 Lower forgetting than TL-1

**Expected Results:**
- Taiwan mAP50-95: 38-42%
- COCO mAP50-95: 45-48%
- Training time: 30-40% faster than TL-1

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

**Hypothesis:**
- 🎯 Better than TL-2 (full adaptation)
- 🛡️ Less overfitting than TL-1 (gradual adaptation)
- ⚖️ Good balance between target performance and stability

**Expected Results:**
- Taiwan mAP50-95: 42-47%
- COCO mAP50-95: 42-46%
- Sweet spot between TL-1 and TL-2

---

### Module TL-4: Discriminative Learning Rates 🟡

**Difficulty:** Medium (Custom optimizer)

**Method:** Apply different learning rates to different layers
- Backbone: LR = 1e-5 (small changes)
- Neck: LR = 1e-4 (moderate changes)
- Head: LR = 1e-3 (large changes)

**Implementation:**
```python
# Requires custom trainer (see implementation section)
from src.trainers import DiscriminativeLRTrainer

trainer = DiscriminativeLRTrainer(
    model='yolo11m.pt',
    data='taiwan-traffic/data.yaml',
    lr_backbone=1e-5,
    lr_neck=1e-4,
    lr_head=1e-3,
    epochs=30
)
trainer.train()
```

**Hypothesis:**
- 🎯 Better Taiwan performance than TL-2
- 🛡️ Lower forgetting than TL-1
- ⚙️ Harder to tune (sensitive to LR ratios)

**Expected Results:**
- Taiwan mAP50-95: 43-48%
- COCO mAP50-95: 44-49%

---

### Module TL-5: Domain Adversarial Training (DANN) 🔴

**Difficulty:** Very High (Research-level)

**Status:** ⚠️ Stretch goal - implement only if time permits

**Method:** 
- Add domain classifier branch
- Use gradient reversal layer
- Learn domain-invariant features

**Implementation:** Requires significant YOLO source code modification

**Advice:** Skip unless you complete all other modules ahead of schedule

---

## Phase 2: Domain Incremental Learning (Solving Forgetting)

**Goal:** High performance on Taiwan **AND** maintain COCO performance  
**Constraint:** Prevent catastrophic forgetting

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

**Hypothesis:**
- 🎯 Good Taiwan performance (close to TL-1)
- 🛡️ **Minimal COCO forgetting** (key benefit)
- ⚖️ Best overall trade-off

**Expected Results:**
- Taiwan mAP50-95: 42-46%
- COCO mAP50-95: 48-51% (maintain baseline!)
- **Forgetting**: < 3%

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

**Hypothesis:**
- 🎯 **Better than DIL-1** (more relevant rehearsal)
- 📊 Improved traffic-class retention
- 💾 Same memory footprint as DIL-1

**Expected Results:**
- Taiwan mAP50-95: 43-47% (slightly better than DIL-1)
- COCO traffic subset: 50-52% (better retention)
- COCO overall: 47-50%

---

### Module DIL-3: LoRA / Adapter-Based Training 🟡

**Difficulty:** Medium-High (Library integration)

**Method:** Freeze pre-trained model, add small trainable adapter modules

**Rationale:** "Add-on memory" - original knowledge remains frozen

**Implementation:**
Requires integration with PEFT library or custom adapter implementation

```python
# Conceptual - requires custom implementation
from src.trainers import LoRATrainer

trainer = LoRATrainer(
    model='yolo11m.pt',
    data='taiwan-traffic/data.yaml',
    lora_rank=8,
    lora_alpha=16,
    target_modules=['conv', 'bn'],
    epochs=30
)
```

**Hypothesis:**
- 🛡️ **Zero forgetting** (base model frozen)
- 📉 Lower Taiwan accuracy (adapter capacity limits)
- 💾 Very memory efficient

**Expected Results:**
- Taiwan mAP50-95: 38-42%
- COCO mAP50-95: 51-52% (no forgetting)
- **Forgetting**: ~0%

---

### Module DIL-4: Elastic Weight Consolidation (EWC) 🔴

**Difficulty:** Very High (Custom training loop)

**Status:** ⚠️ Stretch goal

**Method:** 
- Compute Fisher Information Matrix on COCO
- Add regularization penalty for important weights
- Loss = Task_Loss + λ × EWC_Penalty

**Implementation:** Requires custom PyTorch training loop

**Advice:** Only attempt if DIL-1 through DIL-3 are complete

---

## Hyperparameter Ablation Studies

### Ablation 1: Rehearsal Buffer Size (DIL-1)

**Variable:** Percentage of COCO data in rehearsal buffer

| Buffer Size | COCO Images | Taiwan/COCO Ratio | Training Time |
|-------------|-------------|-------------------|---------------|
| 5%          | ~6,000      | 95/5             | Baseline      |
| 10%         | ~12,000     | 90/10            | +15%          |
| 20%         | ~24,000     | 80/20            | +35%          |
| 30%         | ~36,000     | 70/30            | +55%          |

**Research Questions:**
- At what buffer size do we reach diminishing returns?
- What's the minimum buffer to prevent forgetting?

**Expected Trend:**
- Taiwan accuracy: Decreases slightly with larger buffers
- COCO retention: Improves with larger buffers
- Sweet spot: Likely 10-15%

---

### Ablation 2: Learning Rate Sweep

**Variable:** Initial learning rate (lr0)

| Learning Rate | Use Case | Expected Outcome |
|---------------|----------|------------------|
| 1e-5         | Conservative | Slow convergence, high stability |
| 1e-4         | Moderate | Balanced training |
| 1e-3         | Aggressive | Fast convergence, risk overfitting |
| 1e-2         | Very High | Potential instability |

**Research Questions:**
- What LR balances convergence speed and stability?
- Does optimal LR differ between TL and DIL methods?

**Experimental Design:**
- Test each LR on TL-1, TL-3, DIL-1
- Track convergence curves
- Monitor validation loss

---

### Ablation 3: Batch Composition (DIL-1)

**Variable:** Taiwan vs COCO ratio within each training batch

| Composition | Taiwan Samples/Batch | COCO Samples/Batch | Hypothesis |
|-------------|---------------------|-------------------|------------|
| 50/50       | 8                   | 8                 | Balanced learning |
| 70/30       | 11                  | 5                 | Taiwan-focused |
| 90/10       | 14                  | 2                 | Minimal rehearsal |

**Implementation:**
Requires custom data sampler to control batch composition

**Research Questions:**
- Does batch-level mixing outperform epoch-level mixing?
- What ratio gives best stability/performance trade-off?

---

## Implementation Timeline

### Week 1-2: Foundation + Baselines

**Objectives:**
- ✅ Set up modular training framework
- ✅ Implement dual-domain evaluation pipeline
- ✅ Run baseline experiments

**Tasks:**

**Day 1-2: Infrastructure**
- [x] Create project structure (`src/trainers/`, `src/data/`, `configs/`)
- [x] Implement `DualEvaluator` for COCO + Taiwan validation
- [x] Set up experiment tracking (MLflow, Weights & Biases, or CSV)
- [x] Create baseline evaluation script

**Day 3-4: TL-1 Baseline**
- [x] Implement standard fine-tuning trainer
- [x] Create config: `configs/experiments/tl1_configs.py`
- [x] Run LR ablation with 3 values: [1e-5, 1e-4, 1e-3]
- [x] Early stopping (patience=5) handles optimal epoch selection
- [x] Evaluate on both COCO and Taiwan
- [ ] Document results

**Day 5-7: DIL-1 Rehearsal**
- [ ] Implement COCO subset sampler (`random` mode)
- [ ] Create mixed dataset configs: 5%, 10%, 20%, 30% buffers
- [ ] Run rehearsal experiments (buffer size ablation)
- [ ] Compare against TL-1 baseline
- [ ] Generate preliminary results table
- [ ] Document results

**Deliverables:**
- Working evaluation pipeline
- TL-1 baseline results (3 LR experiments with early stopping)
- DIL-1 results (4 buffer sizes)
- Initial comparison table

---

### Week 3: Easy Extensions

**Objectives:**
- ✅ Implement layer freezing strategies
- ✅ Test stratified rehearsal
- ✅ Analyze hyperparameter sensitivity

**Tasks:**

**Day 8-9: TL-2 Layer Freezing**
- [ ] Test freeze points: [5, 10, 15] layers
- [ ] Compare training speed vs accuracy
- [ ] Run with optimal LR from TL-1

**Day 10-11: TL-3 Progressive Unfreezing**
- [ ] Implement 3-phase unfreezing script
- [ ] Test different unfreezing schedules:
  - Conservative: [15, 10, 5] epochs per phase
  - Balanced: [10, 10, 10]
  - Aggressive: [5, 10, 15]
- [ ] Compare against TL-1 and TL-2

**Day 12-14: DIL-2 Stratified Sampling**
- [ ] Implement stratified COCO sampler
- [ ] Test priority weights: [0.5, 0.7, 0.9] for traffic classes
- [ ] Compare against random rehearsal (DIL-1)
- [ ] Run ablation: batch composition [50/50, 70/30, 90/10]

**Deliverables:**
- Complete TL-1, TL-2, TL-3 comparison
- DIL-1 vs DIL-2 comparison
- Hyperparameter sensitivity analysis
- Updated results table

---

### Week 4+: Advanced Methods (If Time Permits)

**Priority Order:**

**1. TL-4: Discriminative Learning Rates** (Week 4)
- [ ] Implement custom trainer with param groups
- [ ] Test LR ratios: [1:10:100, 1:5:50, 1:20:200] (backbone:neck:head)
- [ ] Compare against best TL method

**2. DIL-3: LoRA Adapters** (Week 5)
- [ ] Research YOLO-compatible LoRA implementation
- [ ] Integrate PEFT library or implement custom adapters
- [ ] Test adapter ranks: [4, 8, 16, 32]
- [ ] Evaluate forgetting vs accuracy trade-off

**3. DIL-4: Elastic Weight Consolidation** (Week 6)
- [ ] Implement Fisher Information computation
- [ ] Add EWC penalty to loss function
- [ ] Test λ values: [100, 1000, 10000]
- [ ] Compare against rehearsal methods

**4. TL-5: Domain Adversarial Training** (Week 7+)
- [ ] Study DANN architecture
- [ ] Implement domain classifier
- [ ] Add gradient reversal layer
- [ ] Integrate into YOLO training loop

**Deliverables:**
- Complete ablation study
- Final results table
- Analysis of method trade-offs

---

## Evaluation Framework

### Metrics to Track

**Per Experiment:**
```
├── Target Domain (Taiwan Traffic)
│   ├── mAP50-95 (overall)
│   ├── mAP50
│   ├── Precision
│   ├── Recall
│   └── Per-class AP (person, bicycle, car, motorcycle, truck)
│
├── Source Domain (COCO)
│   ├── mAP50-95 (overall)
│   ├── Traffic subset mAP (classes 0,1,2,3,7)
│   └── Forgetting = (Baseline - Current)
│
├── Training Metrics
│   ├── Training time (hours)
│   ├── Convergence epoch
│   ├── GPU memory usage
│   └── Inference speed (FPS)
│
└── Analysis Metrics
    ├── Stability (validation loss variance)
    ├── Overfitting gap (train - val)
    └── Parameter efficiency (trainable params)
```

### Evaluation Protocol

**Standard Procedure:**
1. Train model with specified configuration
2. Save checkpoint at best validation epoch
3. Evaluate on Taiwan val set
4. Evaluate on COCO val set
5. Log all metrics to tracking system
6. Generate prediction visualizations (sample images)

**Visualization Outputs:**
- Training curves (loss, mAP over epochs)
- Per-class AP comparison (Taiwan vs COCO)
- Confusion matrices
- Sample predictions on both domains
- Forgetting curve (for DIL methods)

---

## Expected Outcomes & Success Criteria

### Phase 1 (Transfer Learning)

**Success Criteria:**
- ✅ Taiwan mAP50-95 > 42% (baseline: 31.4%)
- ✅ Identify best TL method
- ✅ Understand overfitting behavior

**Expected Ranking (Taiwan Performance):**
1. TL-1 (Standard Fine-tuning): ~45-50%
2. TL-4 (Discriminative LR): ~43-48%
3. TL-3 (Progressive Unfreeze): ~42-47%
4. TL-2 (Layer Freezing): ~38-42%

### Phase 2 (Domain Incremental Learning)

**Success Criteria:**
- ✅ Taiwan mAP50-95 > 40%
- ✅ COCO forgetting < 5% (maintain > 48%)
- ✅ Identify optimal rehearsal strategy

**Expected Ranking (Overall Performance):**
1. DIL-2 (Stratified Rehearsal): Best balance
2. DIL-1 (Random Rehearsal): Close second
3. DIL-3 (LoRA): Zero forgetting, lower Taiwan AP
4. DIL-4 (EWC): High stability, implementation complex

### Hyperparameter Ablations

**Key Findings Expected:**
- Optimal rehearsal buffer: **10-15% COCO**
- Optimal learning rate: **1e-4 to 5e-4**
- Optimal training: **Early stopping with patience=5** (typically converges in 15-25 epochs)
- Optimal batch composition: **70/30 Taiwan/COCO**

---

## Final Deliverables

### 1. Results Summary Table

| Experiment | Method | Implementation | Taiwan mAP↑ | COCO mAP | Forgetting↓ | Training Time |
|------------|--------|----------------|-------------|----------|-------------|---------------|
| Baseline   | Pretrained | - | 31.4% | 51.3% | 0% | - |
| TL-1       | Fine-Tuning | Standard | 47.2% | 35.1% | 31.6% | 3h |
| TL-2       | Freezing | freeze=10 | 40.5% | 46.8% | 8.8% | 2h |
| TL-3       | Prog. Unfreeze | Custom Script | 44.8% | 43.2% | 15.8% | 3.5h |
| TL-4       | Discrim. LR | Custom Trainer | 45.9% | 44.7% | 12.9% | 3.2h |
| DIL-1      | Rehearsal 10% | Mixed Data | 43.6% | 49.8% | 2.9% | 3.8h |
| **DIL-2**  | **Stratified 10%** | **Smart Sampling** | **45.1%** | **50.2%** | **2.1%** | **3.8h** |
| DIL-3      | LoRA | Adapters | 39.2% | 51.1% | 0.4% | 2.5h |

*Note: Numbers are projected estimates for planning purposes*

### 2. Research Report Sections

**To Include:**
1. **Introduction**: Domain shift problem statement
2. **Related Work**: Survey of TL and continual learning methods
3. **Methodology**: Detailed description of each module
4. **Experimental Setup**: Datasets, metrics, implementation details
5. **Results**: 
   - Main results table
   - Ablation study results
   - Visualization of key findings
6. **Analysis**:
   - Method comparison
   - Trade-off analysis (accuracy vs forgetting)
   - Computational efficiency
7. **Conclusion**: Best practices and recommendations

### 3. Code Repository

**Structure:**
```
Scooter-YOLO/
├── src/
│   ├── trainers/          # All training modules
│   ├── data/              # Data samplers
│   ├── evaluation/        # Eval pipeline
│   └── configs/           # All experiment configs
├── scripts/
│   ├── run_experiment.py
│   ├── run_ablation.py
│   └── generate_report.py
├── results/
│   ├── experiments/       # Per-experiment results
│   ├── figures/           # Plots and visualizations
│   └── summary.csv        # All results
├── docs/
│   ├── PLAN.md           # This file
│   └── RESULTS.md        # Final results report
└── README.md
```

### 4. Visualizations

**Required Plots:**
- Training curves for all experiments
- Taiwan vs COCO performance scatter plot
- Forgetting vs Taiwan accuracy trade-off
- Hyperparameter sensitivity curves
- Per-class AP comparison (radar chart)

---

## Notes and Considerations

### Implementation Tips

**For Transfer Learning:**
- Use early stopping (patience=5) to prevent overfitting
- Monitor train/val gap closely
- Save checkpoints every 5 epochs for analysis

**For Rehearsal:**
- Ensure balanced class distribution in COCO subset
- Consider memory constraints when increasing buffer
- Stratified sampling is worth the extra implementation effort

**For Advanced Methods:**
- Start simple, add complexity only if needed
- Document all hyperparameter choices
- Version control everything

### Common Pitfalls to Avoid

⚠️ **Overfitting**: Taiwan dataset is small (~3K images). Use strong augmentation and early stopping.

⚠️ **Evaluation Leakage**: Always use separate validation sets. Never tune on test data.

⚠️ **Unfair Comparison**: Keep batch size, image size, and augmentation consistent across experiments.

⚠️ **Ignoring Compute**: Track GPU hours. Some methods may not be worth the computational cost.

### Resource Planning

**Estimated GPU Hours:**
- Week 1-2: ~25 GPU hours (baselines with early stopping)
- Week 3: ~30 GPU hours (extensions)
- Week 4+: ~50 GPU hours (advanced)
- **Total**: ~105 GPU hours

**Storage Requirements:**
- Model checkpoints: ~50GB (50 experiments × 1GB)
- COCO subsets: ~5-10GB
- Results and logs: ~2GB

---

## Next Steps

**Immediate Actions:**
1. ✅ Review and approve this plan
2. ✅ Set up project structure
3. ✅ Implement evaluation pipeline
4. ✅ Run TL-1 baseline
5. ✅ Compare with current results (31.4% Taiwan mAP)

**Questions to Resolve:**
- [ ] Confirm GPU availability and compute budget
- [ ] Choose experiment tracking tool (TensorBoard, W&B, MLflow?)
- [ ] Define stopping criteria (when is "good enough"?)
- [ ] Set timeline for final report/presentation

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-06  
**Status:** Ready for Implementation
