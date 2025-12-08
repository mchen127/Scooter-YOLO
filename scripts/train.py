import sys
import os
from pathlib import Path
from ultralytics import YOLO
import wandb
import argparse

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.base_config import config, add_config_args, update_config
from scripts.callbacks import on_fit_epoch_end, on_train_start

# Set WandB project name explicitly BEFORE importing ultralytics
os.environ["WANDB_PROJECT"] = config.wandb_project


def train_tl1(
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
):
    # Determine data suffix
    data_name = Path(data_yaml).stem
    data_suffix = (
        f"_{data_name}" if "taiwan" not in data_name and "data" != data_name else ""
    )

    run_name = f"tl1_baseline_lr{lr0}{data_suffix}"
    print(f"Starting TL-1 (Baseline) - {run_name}")

    model = YOLO(model_path)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_start", on_train_start)

    model.train(
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


def train_tl2(
    model_path,
    data_yaml,
    epochs,
    batch,
    lr0,
    device,
    freeze,
    patience,
    workers,
    cache,
    amp,
    optimizer,
    seed,
):
    print(f"Starting TL-2 (Layer Freezing) with freeze={freeze}")

    # Determine data suffix
    data_name = Path(data_yaml).stem
    data_suffix = (
        f"_{data_name}" if "taiwan" not in data_name and "data" != data_name else ""
    )

    run_name = f"tl2_freeze{freeze}_lr{lr0}{data_suffix}"

    model = YOLO(model_path)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_start", on_train_start)

    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        lr0=lr0,
        device=device,
        freeze=freeze,
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


def train_tl3(
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
):
    print("Starting TL-3 (Progressive Unfreezing)")

    # Generate a unique group ID for this TL-3 session so phases appear together in WandB
    group_id = f"tl3_{wandb.util.generate_id()}"
    os.environ["WANDB_RUN_GROUP"] = group_id
    print(f"WandB Group ID: {group_id}")

    # Determine data suffix
    data_name = Path(data_yaml).stem
    data_suffix = (
        f"_{data_name}" if "taiwan" not in data_name and "data" != data_name else ""
    )

    run_name_p1 = f"tl3_phase1_freeze10{data_suffix}"

    # Split epochs into 3 phases
    phase_epochs = epochs // 3

    # === Phase 1: Freeze Backbone (0-10) ===
    print("\n=== Phase 1: Freeze Backbone (0-10) ===")
    model = YOLO(model_path)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_start", on_train_start)

    model.train(
        data=data_yaml,
        epochs=phase_epochs,
        batch=batch,
        lr0=lr0,
        device=device,
        freeze=10,
        project=config.project,
        name=run_name_p1,
        exist_ok=True,
        workers=workers,
        cache=cache,
        amp=amp,
        optimizer=optimizer,
        seed=seed,
        deterministic=True,
    )

    # Get the best model from Phase 1
    phase1_model = f"{config.project}/{run_name_p1}/weights/best.pt"
    # === Phase 2: Unfreeze Upper Backbone (Freeze 0-5) ===
    print(
        f"\n=== Phase 2: Unfreeze Upper Backbone (Freeze 0-5) - Loading {phase1_model} ==="
    )
    run_name_p2 = f"tl3_phase2_freeze5{data_suffix}"

    model = YOLO(phase1_model)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_start", on_train_start)

    model.train(
        data=data_yaml,
        epochs=phase_epochs,  # Train for another N epochs
        batch=batch,
        lr0=lr0,
        device=device,
        freeze=5,
        project=config.project,
        name=run_name_p2,
        exist_ok=True,
        workers=workers,
        cache=cache,
        amp=amp,
        optimizer=optimizer,
        seed=seed,
        deterministic=True,
    )

    # Get the best model from Phase 2
    phase2_model = f"{config.project}/{run_name_p2}/weights/best.pt"
    # === Phase 3: Unfreeze All ===
    print(f"\n=== Phase 3: Unfreeze All - Loading {phase2_model} ===")
    run_name_p3 = f"tl3_phase3_freeze0{data_suffix}"

    model = YOLO(phase2_model)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_start", on_train_start)

    model.train(
        data=data_yaml,
        epochs=phase_epochs,  # Train for final N epochs
        batch=batch,
        lr0=lr0 * 0.1,  # Lower LR for fine-tuning
        device=device,
        freeze=0,
        project=config.project,
        name=run_name_p3,
        exist_ok=True,
        workers=workers,
        cache=cache,
        amp=amp,
        optimizer=optimizer,
        seed=seed,
        deterministic=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Script-specific arguments
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["tl1", "tl2", "tl3", "dil1", "dil2"],
        help="Training method",
    )
    parser.add_argument(
        "--freeze", type=int, default=10, help="Layers to freeze (for TL-2)"
    )

    # Add all config arguments dynamically
    add_config_args(parser, config)

    args = parser.parse_args()

    # Update global config with command line args
    update_config(config, args)

    # Re-apply env var in case project name was changed via CLI
    os.environ["WANDB_PROJECT"] = config.wandb_project

    # Validating shortcuts for DIL methods
    if args.method == "dil1":
        print("Method DIL-1 selected: Using Mixed Dataset (10% Random)")
        config.data = str(config.generated_path / "mixed_10pct_random.yaml")
        # DIL-1 typically uses standard training (TL-1) on mixed data
        train_tl1(
            config.model,
            config.data,
            config.epochs,
            config.batch,
            config.lr0,
            config.device,
            config.patience,
            config.workers,
            config.cache,
            config.amp,
            config.optimizer,
            config.seed,
        )

    elif args.method == "dil2":
        print("Method DIL-2 selected: Using Mixed Dataset (10% Traffic Stratified)")
        config.data = str(config.generated_path / "mixed_10pct_stratified.yaml")
        # DIL-2 typically uses standard training (TL-1) on mixed data
        train_tl1(
            config.model,
            config.data,
            config.epochs,
            config.batch,
            config.lr0,
            config.device,
            config.patience,
            config.workers,
            config.cache,
            config.amp,
            config.optimizer,
            config.seed,
        )

    elif args.method == "tl1":
        train_tl1(
            config.model,
            config.data,
            config.epochs,
            config.batch,
            config.lr0,
            config.device,
            config.patience,
            config.workers,
            config.cache,
            config.amp,
            config.optimizer,
            config.seed,
        )
    elif args.method == "tl2":
        train_tl2(
            config.model,
            config.data,
            config.epochs,
            config.batch,
            config.lr0,
            config.device,
            args.freeze,
            config.patience,
            config.workers,
            config.cache,
            config.amp,
            config.optimizer,
            config.seed,
        )
    elif args.method == "tl3":
        train_tl3(
            config.model,
            config.data,
            config.epochs,
            config.batch,
            config.lr0,
            config.device,
            config.patience,
            config.workers,
            config.cache,
            config.amp,
            config.optimizer,
            config.seed,
        )
