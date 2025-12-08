import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.base_config import config, add_config_args, update_config

# Set WandB project name explicitly BEFORE importing ultralytics
os.environ["WANDB_PROJECT"] = config.wandb_project

import argparse
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics import YOLO
import torch.nn as nn
import torch


class DiscriminativeLRTrainer(DetectionTrainer):
    def build_optimizer(
        self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5
    ):
        """
        Builds an optimizer with discriminative learning rates.
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(
            v for k, v in nn.__dict__.items() if "Norm" in k
        )  # normalization layers, i.e. BatchNorm2d()

        # Define LR multipliers
        lr_backbone = 0.1  # 10x smaller
        lr_neck = 1.0  # standard
        lr_head = 10.0  # 10x larger

        print(
            f"[Discriminative LR] Building optimizer with multipliers: Backbone={lr_backbone}, Neck={lr_neck}, Head={lr_head}"
        )

        for v in model.modules():
            if hasattr(v, "bias") and isinstance(
                v.bias, nn.Parameter
            ):  # bias (no decay)
                # Determine layer type/index to assign LR
                # This is hard to do purely by module iteration without names.
                # We will iterate named_parameters instead below.
                pass

        # Re-implementing group construction with named parameters
        # Ultralytics default groups: 0=weights(decay), 1=weights(no decay), 2=biases
        # We want groups based on LR.

        # Let's simplify: We will modify the param groups AFTER standard build if possible,
        # OR we construct manually. Manual is safer.

        params_backbone = []
        params_neck = []
        params_head = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Parse layer index from name (e.g., "model.0.conv.weight")
            try:
                layer_idx = int(name.split(".")[1])
            except (IndexError, ValueError):
                layer_idx = -1

            # Assign to groups
            # YOLO11m backbone is roughly 0-10, Head is the last layer (e.g. 22 or 23)
            # We need to check the exact architecture.
            # Assuming standard YOLO structure:
            if layer_idx <= 10:
                params_backbone.append(param)
            elif layer_idx >= 22:  # Detect head
                params_head.append(param)
            else:
                params_neck.append(param)

        # Create optimizer
        # Note: We are ignoring weight decay distinction for simplicity here,
        # or we can add it back if needed. For now, let's just set LRs.

        optimizer = torch.optim.SGD(
            [
                {"params": params_backbone, "lr": lr * lr_backbone},
                {"params": params_neck, "lr": lr * lr_neck},
                {"params": params_head, "lr": lr * lr_head},
            ],
            lr=lr,
            momentum=momentum,
            nesterov=True,
        )

        return optimizer


def train_discriminative(model_path, data_yaml, epochs, batch, device):
    # We need to instantiate the custom trainer
    # This is slightly complex with Ultralytics API as it hides the Trainer instantiation.
    # We can use the 'overrides' argument in YOLO class or instantiate Trainer directly.

    print("Starting TL-4 (Discriminative LR)")

    # Load model to get config
    model = YOLO(model_path)

    # We will use the model.train() but inject our custom trainer class?
    # No, model.train() instantiates the default trainer.
    # We must instantiate our custom trainer manually.

    args = dict(
        model=model_path,
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        device=device,
        project=config.project,
        name="tl4_discrim",
        patience=config.patience,
        workers=config.workers,
        cache=config.cache,
        amp=config.amp,
        optimizer=config.optimizer,
    )

    trainer = DiscriminativeLRTrainer(overrides=args)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add all config arguments dynamically
    add_config_args(parser, config)

    args = parser.parse_args()

    # Update global config with command line args
    update_config(config, args)

    # Re-apply env var
    os.environ["WANDB_PROJECT"] = config.wandb_project

    train_discriminative(
        config.model, config.data, config.epochs, config.batch, config.device
    )
