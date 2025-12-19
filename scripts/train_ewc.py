"""
Elastic Weight Consolidation (EWC) Training Script

This script implements EWC for continual learning, preventing catastrophic forgetting
when fine-tuning YOLO11m from COCO to Taiwan Traffic dataset.

EWC Loss: L_total = L_detection + (λ/2) * Σ F_i * (θ_i - θ*_i)²

Usage:
    python scripts/train_ewc.py --ewc_lambda 100.0 --epochs 50 --batch 16
"""

import sys
import os
from pathlib import Path
import argparse
import copy  # noqa: F401

import torch
import torch.nn as nn  # noqa: F401
from torch.utils.data import DataLoader as TorchDataLoader  # noqa: F401
import wandb  # noqa: F401

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.base_config import config, add_config_args, update_config
from scripts.callbacks import on_fit_epoch_end, on_train_start

# Set WandB project name explicitly BEFORE importing ultralytics
os.environ["WANDB_PROJECT"] = config.wandb_project


class EWCTrainer(DetectionTrainer):
    """
    Custom trainer that adds Elastic Weight Consolidation (EWC) penalty
    to prevent catastrophic forgetting.
    """

    def __init__(  # noqa: PLR0913
        self,
        cfg=None,
        overrides=None,
        ewc_lambda=100.0,
        fisher_dict=None,
        pretrained_params=None,
    ):
        # Use DEFAULT_CFG if cfg is None to avoid validation errors
        from ultralytics.cfg import DEFAULT_CFG

        if cfg is None:
            cfg = dict(vars(DEFAULT_CFG))  # Convert IterableSimpleNamespace to dict
        super().__init__(cfg=cfg, overrides=overrides)
        self.ewc_lambda = ewc_lambda
        self.fisher_dict = fisher_dict or {}
        self.pretrained_params = pretrained_params or {}
        self.ewc_loss_value = 0.0  # Track for logging

    def compute_ewc_penalty(self):
        """
        Compute the EWC penalty: (λ/2) * Σ F_i * (θ_i - θ*_i)²
        """
        ewc_loss = 0.0

        for name, param in self.model.named_parameters():
            if name in self.fisher_dict and name in self.pretrained_params:
                fisher = self.fisher_dict[name]
                pretrained = self.pretrained_params[name]

                # Move to same device as param if needed
                if fisher.device != param.device:
                    fisher = fisher.to(param.device)
                    self.fisher_dict[name] = fisher
                if pretrained.device != param.device:
                    pretrained = pretrained.to(param.device)
                    self.pretrained_params[name] = pretrained

                # Compute penalty for this parameter
                diff = param - pretrained
                ewc_loss += (fisher * diff.pow(2)).sum()

        # Log EWC info once at first call
        if not hasattr(self, "_ewc_logged"):
            print(
                f"[EWC] Lambda: {self.ewc_lambda}, Params with Fisher: {len(self.fisher_dict)}"
            )
            self._ewc_logged = True

        return (self.ewc_lambda / 2) * ewc_loss

    def optimizer_step(self):
        """
        Override optimizer step to add EWC penalty to the loss.
        The EWC penalty is added as additional gradients.
        
        IMPORTANT: EWC gradients must be scaled by the AMP scaler to match
        the detection loss gradients. Otherwise, scaler.unscale_() will
        reduce EWC gradients to nearly zero.
        """
        # Compute and add EWC penalty gradients
        if self.fisher_dict and self.pretrained_params:
            ewc_penalty = self.compute_ewc_penalty()
            self.ewc_loss_value = ewc_penalty.item()

            # Add EWC gradients to existing gradients (only if penalty is non-zero)
            if ewc_penalty.requires_grad and self.ewc_loss_value > 1e-10:
                # Scale the EWC penalty by the AMP scaler to match detection gradients
                # This is crucial - without scaling, unscale_() will reduce EWC grads to ~0
                scaled_ewc_penalty = self.scaler.scale(ewc_penalty)
                scaled_ewc_penalty.backward()

        # Call parent optimizer step
        super().optimizer_step()

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Override to add EWC loss to the logged items.
        """
        keys = super().label_loss_items(loss_items, prefix)
        # Add EWC loss key if we have keys
        if keys and isinstance(keys, dict):
            keys[f"{prefix}/ewc_loss"] = self.ewc_loss_value
        return keys


def compute_fisher_information(model, dataloader, device, num_samples=1000):
    """
    Compute the diagonal Fisher Information Matrix using gradients
    on the source dataset (COCO subset).

    Fisher[i] = E[(∂L/∂θ_i)²]

    We approximate this by averaging squared gradients over samples.

    Args:
        num_samples: Number of samples to use. Set to -1 to use all samples.
    """
    use_all = num_samples < 0
    samples_desc = "all" if use_all else str(num_samples)
    print(f"[EWC] Computing Fisher Information Matrix on {samples_desc} samples...")

    fisher_dict = {}

    # Initialize Fisher dict with zeros
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param.data)

    model.train()
    sample_count = 0

    for batch_idx, batch in enumerate(dataloader):
        # If num_samples >= 0, stop after reaching the limit
        if not use_all and sample_count >= num_samples:
            break

        # Move batch to device
        if isinstance(batch, dict):
            images = batch.get("img", batch.get("image"))
        else:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch

        if images is None:
            continue

        images = (
            images.to(device).float() / 255.0
            if images.max() > 1
            else images.to(device).float()
        )

        # Enable gradient tracking on images for gradient flow
        images = images.requires_grad_(True)

        # Forward pass (we use a simple forward to get features, not full detection loss)
        model.zero_grad()

        try:
            # Use torch.enable_grad() to ensure gradients are computed
            with torch.enable_grad():
                # For YOLO, we can do a forward pass and use the output magnitude as a proxy
                outputs = model(images)

                # Create a simple loss from outputs for gradient computation
                if isinstance(outputs, (list, tuple)):
                    loss = sum(o.sum() for o in outputs if isinstance(o, torch.Tensor))
                else:
                    loss = outputs.sum()

                loss.backward()

            # Accumulate squared gradients
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += param.grad.data.pow(2)

            sample_count += images.shape[0]

            if (batch_idx + 1) % 10 == 0:
                print(f"[EWC] Processed {sample_count}/{num_samples} samples...")

        except Exception as e:
            print(f"[EWC] Warning: Error processing batch {batch_idx}: {e}")
            continue

    # Average the Fisher information
    for name in fisher_dict:
        fisher_dict[name] /= max(sample_count, 1)

    print(f"[EWC] Fisher computation complete. Processed {sample_count} samples.")

    # Print stats before normalization
    total_params = sum(f.numel() for f in fisher_dict.values())
    non_zero = sum((f > 0).sum().item() for f in fisher_dict.values())

    # Compute global max for normalization
    global_max = (
        max(f.max().item() for f in fisher_dict.values()) if fisher_dict else 1.0
    )
    global_mean = (
        sum(f.mean().item() for f in fisher_dict.values()) / len(fisher_dict)
        if fisher_dict
        else 1.0
    )

    print(f"[EWC] Before normalization: max={global_max:.2e}, mean={global_mean:.2e}")

    # Normalize Fisher values so max = 1.0 (makes lambda selection more intuitive)
    if global_max > 0:
        for name in fisher_dict:
            fisher_dict[name] /= global_max

    print(
        f"[EWC] After normalization: max=1.0, Fisher values scaled by 1/{global_max:.2e}"
    )
    print(
        f"[EWC] Total parameters: {total_params:,}, Non-zero Fisher values: {non_zero:,}"
    )

    return fisher_dict


def train_ewc(
    model_path,
    data_yaml,
    epochs,
    batch,
    lr0,
    device,
    patience,
    workers,
    cache,
    amp,
    optimizer,
    seed,
    ewc_lambda,
    fisher_samples,
    coco_subset_yaml,
):
    """
    Train with Elastic Weight Consolidation.

    Steps:
    1. Load pretrained model and store weights
    2. Compute Fisher Information on COCO subset
    3. Train on Taiwan Traffic with EWC penalty
    """
    print(f"[EWC] Starting EWC Training with λ={ewc_lambda}")

    run_name = f"ewc_fishersamples{fisher_samples}_lambda{ewc_lambda}_lr{lr0}"

    # Step 1: Load pretrained model and store weights
    print(f"[EWC] Loading pretrained model: {model_path}")
    pretrained_model = YOLO(model_path)

    device_str = f"cuda:{device}" if device != "cpu" else "cpu"

    # Store pretrained parameters (deep copy)
    # The model is at pretrained_model.model (DetectionModel)
    pretrained_params = {}
    pretrained_model.model.to(device_str)
    pretrained_model.model.train()  # Set to train mode

    # IMPORTANT: Enable requires_grad on all parameters for Fisher computation
    # YOLO models load with requires_grad=False by default
    for param in pretrained_model.model.parameters():
        param.requires_grad_(True)

    # Debug: print model structure
    print(f"[EWC] Model type: {type(pretrained_model.model)}")

    # Get all parameters, not just requires_grad=True
    for name, param in pretrained_model.model.named_parameters():
        pretrained_params[name] = param.data.clone()

    print(f"[EWC] Stored {len(pretrained_params)} pretrained parameter tensors")

    if len(pretrained_params) == 0:
        print("[EWC] WARNING: No parameters found! Checking model.model...")
        # Try alternative access
        if hasattr(pretrained_model, "model") and hasattr(
            pretrained_model.model, "model"
        ):
            for name, param in pretrained_model.model.model.named_parameters():
                pretrained_params[f"model.{name}"] = param.data.clone()
            print(
                f"[EWC] After retry: Stored {len(pretrained_params)} pretrained parameter tensors"
            )

    # Step 2: Compute Fisher Information on COCO subset
    print(f"[EWC] Computing Fisher Information on COCO subset: {coco_subset_yaml}")

    # Use a simpler approach: build dataloader using torch's ImageFolder-like approach
    # or use YOLO's predict/val which handles dataset loading internally
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data.dataset import YOLODataset
    from torch.utils.data import DataLoader

    coco_data = check_det_dataset(coco_subset_yaml)
    coco_train_path = coco_data.get("val", coco_data.get("train"))  # Use val for Fisher

    # Build dataset using YOLODataset directly
    dataset = YOLODataset(
        img_path=coco_train_path,
        imgsz=640,
        augment=False,
        cache=False,
        data=coco_data,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch,
        num_workers=workers,
        shuffle=True,
        collate_fn=getattr(dataset, "collate_fn", None),
    )

    # Compute Fisher
    fisher_dict = compute_fisher_information(
        pretrained_model.model,
        dataloader,
        device_str,
        num_samples=fisher_samples,
    )

    # Step 3: Train with EWC penalty
    print("[EWC] Starting training on Taiwan Traffic with EWC penalty...")

    # Training arguments
    train_args = dict(
        model=model_path,
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        lr0=lr0,
        device=device,
        project=config.project,
        name=run_name,
        patience=patience,
        workers=workers,
        cache=cache,
        amp=amp,
        optimizer=optimizer,
        seed=seed,
        deterministic=True,
    )

    # Create custom EWC trainer (pass overrides only, cfg defaults to internal handling)
    trainer = EWCTrainer(
        overrides=train_args,
        ewc_lambda=ewc_lambda,
        fisher_dict=fisher_dict,
        pretrained_params=pretrained_params,
    )

    # Add callbacks
    trainer.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    trainer.add_callback("on_train_start", on_train_start)

    # Train
    trainer.train()

    print(f"[EWC] Training complete! Model saved to: {config.project}/{run_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLO with Elastic Weight Consolidation"
    )

    # Add all config arguments dynamically (includes ewc_lambda, fisher_samples)
    add_config_args(parser, config)

    args = parser.parse_args()

    # Update global config with command line args
    update_config(config, args)

    # Re-apply env var in case project name was changed via CLI
    os.environ["WANDB_PROJECT"] = config.wandb_project

    # Run EWC training
    train_ewc(
        model_path=config.model,
        data_yaml=config.data,
        epochs=config.epochs,
        batch=config.batch,
        lr0=config.lr0,
        device=config.device,
        patience=config.patience,
        workers=config.workers,
        cache=config.cache,
        amp=config.amp,
        optimizer=config.optimizer,
        seed=config.seed,
        ewc_lambda=config.ewc_lambda,
        fisher_samples=config.fisher_samples,
        coco_subset_yaml=config.coco_traffic_subset_yaml,
    )
