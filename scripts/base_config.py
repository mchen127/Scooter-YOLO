from dataclasses import dataclass
from pathlib import Path
import argparse
import dataclasses


@dataclass
class BaseConfig:
    # Paths
    model: str = "yolo11m.pt"
    data: str = "datasets/taiwan-traffic/data.yaml"
    coco_path: Path = Path("datasets/coco")
    taiwan_path: Path = Path("datasets/taiwan-traffic")
    generated_path: Path = Path("datasets/generated")

    # Training Hyperparameters
    epochs: int = 30
    batch: int = 32
    lr0: float = 0.01
    device: str = "0"
    patience: int = 5
    optimizer: str = "SGD"  # Force SGD for consistency
    seed: int = 42  # Random seed for reproducibility

    # Performance
    workers: int = 20
    cache: bool = True  # Cache images in RAM for speed
    amp: bool = True  # Automatic Mixed Precision (True by default)

    # Evaluation
    eval_freq: int = 1  # Run COCO subset eval every N epochs

    # Logging
    project: str = "YOLO-Taiwan-Traffic"
    wandb_project: str = "YOLO-Taiwan-Traffic"

    @property
    def coco_traffic_subset_yaml(self) -> str:
        return str(self.generated_path / "coco_traffic_subset.yaml")


def add_config_args(parser: argparse.ArgumentParser, config_instance: BaseConfig):
    """
    Dynamically adds arguments from the config dataclass to the parser.
    """
    for field in dataclasses.fields(config_instance):
        default_val = getattr(config_instance, field.name)
        field_type = field.type
        help_text = f"Default: {default_val}"

        parser.add_argument(
            f"--{field.name}",
            type=field_type,
            default=default_val,
            help=help_text,
        )


def update_config(config_instance: BaseConfig, args: argparse.Namespace):
    """
    Updates the config instance with values from parsed args.
    """
    for field in dataclasses.fields(config_instance):
        if hasattr(args, field.name):
            setattr(config_instance, field.name, getattr(args, field.name))


# Global instance
config = BaseConfig()
