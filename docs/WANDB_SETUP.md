# Weights & Biases Setup Guide

## Initial Setup

### 1. Install W&B (Already Done ✓)
```bash
pip install wandb
```

### 2. Login to W&B
```bash
wandb login
```

You'll be prompted to enter your API key. Get it from: https://wandb.ai/authorize

**Alternative: Set API key as environment variable**
```bash
export WANDB_API_KEY="your_api_key_here"
```

### 3. Test W&B Connection
```bash
python -c "import wandb; wandb.login()"
```

## Usage

### Baseline Evaluation with W&B
```bash
# Evaluate pretrained model and log to W&B
python scripts/evaluate_baseline.py --wandb

# Specify custom project
python scripts/evaluate_baseline.py --wandb --wandb-project my-project-name
```

### Training with W&B (Coming Soon)
The training scripts will be updated to automatically log to W&B.

## What Gets Logged

### Metrics
- **Training**: Loss, mAP, precision, recall per epoch
- **Taiwan Evaluation**: mAP50-95, mAP50, precision, recall
- **COCO Evaluation**: mAP50-95, mAP50, precision, recall
- **Forgetting**: Absolute and relative forgetting metrics

### Artifacts
- Model checkpoints (best.pt)
- Configuration files
- Evaluation results (JSON)

### Metadata
- Hyperparameters (LR, batch size, epochs, etc.)
- Dataset information
- Experiment tags and notes

## W&B Dashboard

After logging in, view your experiments at:
```
https://wandb.ai/<your-username>/scooter-yolo-domain-adaptation
```

### Useful Views
- **Runs Table**: Compare all experiments
- **Charts**: Training curves, metric comparisons
- **Artifacts**: Download model checkpoints
- **Reports**: Create shareable reports

## Offline Mode

If you don't have internet or want to log locally:

```bash
# Set offline mode
export WANDB_MODE=offline

# Run experiments
python scripts/evaluate_baseline.py --wandb

# Later, sync to cloud
wandb sync wandb/offline-run-*
```

## Disable W&B

```bash
# Disable W&B completely
export WANDB_MODE=disabled

# Or don't use --wandb flag
python scripts/evaluate_baseline.py  # No W&B logging
```

## Project Structure

```
wandb/
├── latest-run/          # Symlink to most recent run
├── run-20251206_220000/ # Individual run directories
│   ├── files/
│   │   ├── config.yaml
│   │   ├── requirements.txt
│   │   └── wandb-metadata.json
│   └── logs/
└── settings
```

## Tips

### 1. Organize with Tags
```python
wandb_logger = setup_wandb(
    experiment_name="tl1_lr_1e4",
    tags=["tl1", "lr_ablation", "baseline"],
    ...
)
```

### 2. Add Notes
```python
wandb_logger = setup_wandb(
    experiment_name="tl1_lr_1e4",
    notes="Testing conservative learning rate for stability",
    ...
)
```

### 3. Group Related Runs
Use the same group name for related experiments:
```python
config = {"group": "lr_ablation"}
```

### 4. Compare Experiments
In W&B dashboard:
1. Select multiple runs
2. Click "Compare"
3. View side-by-side metrics

## Troubleshooting

### API Key Issues
```bash
# Check if logged in
wandb whoami

# Re-login
wandb login --relogin
```

### Slow Uploads
```bash
# Reduce upload frequency
export WANDB_CONSOLE=off
```

### Storage Limits
Free tier: 100GB storage
- Log only important artifacts
- Clean up old runs periodically

## Next Steps

1. ✅ Run baseline evaluation with W&B
2. Update training scripts to auto-log
3. Create W&B report for ablation study
4. Share results with team

## Resources

- [W&B Documentation](https://docs.wandb.ai/)
- [W&B Python API](https://docs.wandb.ai/ref/python)
- [Example Projects](https://wandb.ai/gallery)
