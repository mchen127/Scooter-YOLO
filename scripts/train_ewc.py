import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.base_config import config, add_config_args, update_config

# Set WandB project name explicitly BEFORE importing ultralytics
os.environ["WANDB_PROJECT"] = config.wandb_project

import argparse
import torch
# import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics import YOLO
from tqdm import tqdm


class EWCLoss(v8DetectionLoss):
    def __init__(self, model, fisher, opt_params, ewc_lambda=100.0):
        super().__init__(model)
        self.fisher = fisher
        self.opt_params = opt_params
        self.ewc_lambda = ewc_lambda
        print(f"[EWC] Initialized Loss with lambda={ewc_lambda}")

    def __call__(self, preds, batch):
        # 1. Calculate original task loss
        loss, loss_items = super().__call__(preds, batch)

        # 2. Calculate EWC penalty
        ewc_loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                fisher_val = self.fisher[name].to(param.device)
                opt_val = self.opt_params[name].to(param.device)
                ewc_loss += (fisher_val * (param - opt_val) ** 2).sum()

        # 3. Combine
        total_loss = loss + (self.ewc_lambda * ewc_loss)

        # Update loss items for logging (optional, might break format)
        # loss_items[0] += (self.ewc_lambda * ewc_loss).item()

        return total_loss, loss_items


class EWCTrainer(DetectionTrainer):
    def __init__(self, fisher, opt_params, ewc_lambda, *args, **kwargs):
        self.fisher = fisher
        self.opt_params = opt_params
        self.ewc_lambda = ewc_lambda
        super().__init__(*args, **kwargs)

    def init_model(self, model, cfg=None, verbose=True):
        # Initialize model normally
        super().init_model(model, cfg, verbose)

        # Replace loss function with EWC Loss
        self.loss = EWCLoss(self.model, self.fisher, self.opt_params, self.ewc_lambda)


def calculate_fisher(model_path, data_yaml, device, num_samples=100):
    print(f"Calculating Fisher Information Matrix on {data_yaml}...")

    # Load model
    model = YOLO(model_path)
    model.to(device)
    model.model.train()  # Set to train mode to get gradients

    # Get dataloader (reuse Ultralytics internal dataloader)
    # We can use a Trainer just to build the loader
    trainer = DetectionTrainer(
        overrides={"model": model_path, "data": data_yaml, "batch": 8}
    )
    loader = trainer.get_dataloader(
        trainer.get_dataset(data_yaml, mode="val"), batch_size=8, mode="val"
    )

    fisher = {}
    opt_params = {}

    # Initialize fisher dict
    for name, param in model.model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param)
            opt_params[name] = param.data.clone()

    # Iterate
    count = 0
    criterion = v8DetectionLoss(model.model)

    for batch in tqdm(loader):
        batch = trainer.preprocess_batch(batch)

        # Forward
        preds = model.model(batch["img"])
        loss, _ = criterion(preds, batch)

        # Backward
        model.model.zero_grad()
        loss.backward()

        # Accumulate Fisher
        for name, param in model.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.data**2

        count += len(batch["img"])
        if count >= num_samples:
            break

    # Normalize
    for name in fisher:
        fisher[name] /= count

    print(f"Fisher Matrix calculated on {count} samples.")
    return fisher, opt_params


def train_ewc(model_path, data_yaml, coco_yaml, epochs, batch, device, ewc_lambda):
    # 1. Calculate Fisher on COCO (Source Domain)
    # We use coco_yaml (or subset) to calculate importance
    fisher, opt_params = calculate_fisher(model_path, coco_yaml, device)

    # 2. Train on Taiwan (Target Domain) with EWC
    print(f"Starting EWC Training with lambda={ewc_lambda}")

    # We need to pass the fisher/opt_params to the trainer
    # Since we can't easily pass args to __init__ via overrides, we might need to instantiate directly

    args = dict(
        model=model_path,
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        device=device,
        project=config.project,
        name=f"dil3_ewc_{ewc_lambda}",
        patience=config.patience,
        workers=config.workers,
        cache=config.cache,
        amp=config.amp,
        optimizer=config.optimizer,
    )

    trainer = EWCTrainer(
        fisher=fisher, opt_params=opt_params, ewc_lambda=ewc_lambda, overrides=args
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Script-specific arguments
    parser.add_argument(
        "--coco_data",
        type=str,
        default=config.coco_traffic_subset_yaml,
        help="Data to calculate Fisher on",
    )
    parser.add_argument("--ewc_lambda", type=float, default=100.0)

    # Add all config arguments dynamically
    add_config_args(parser, config)

    args = parser.parse_args()

    # Update global config with command line args
    update_config(config, args)

    # Re-apply env var
    os.environ["WANDB_PROJECT"] = config.wandb_project

    train_ewc(
        config.model,
        config.data,
        args.coco_data,
        config.epochs,
        config.batch,
        config.device,
        args.ewc_lambda,
    )
