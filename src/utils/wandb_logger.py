"""
Weights & Biases integration for experiment tracking.

This module provides W&B logging for training runs and evaluations.
"""

import wandb
import logging
from typing import Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class WandBLogger:
    """
    Weights & Biases logger for domain adaptation experiments.
    
    Tracks:
    - Training metrics (loss, mAP, etc.)
    - Evaluation results (Taiwan + COCO)
    - Hyperparameters
    - Model artifacts
    """
    
    def __init__(
        self,
        project: str = "scooter-yolo-domain-adaptation",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        mode: str = "online"  # "online", "offline", or "disabled"
    ):
        """
        Initialize W&B logger.
        
        Args:
            project: W&B project name
            entity: W&B entity (username or team)
            name: Run name
            config: Configuration dictionary to log
            tags: List of tags for this run
            notes: Notes about this run
            mode: Logging mode (online/offline/disabled)
        """
        self.project = project
        self.entity = entity
        self.mode = mode
        self.run = None
        
        # Initialize run
        if mode != "disabled":
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=config,
                tags=tags,
                notes=notes,
                mode=mode
            )
            logger.info(f"W&B run initialized: {self.run.name}")
        else:
            logger.info("W&B logging disabled")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to W&B."""
        if self.run:
            wandb.log(metrics, step=step)
    
    def log_config(self, config: Dict):
        """Update run configuration."""
        if self.run:
            wandb.config.update(config)
    
    def log_artifact(self, artifact_path: str, artifact_type: str, name: Optional[str] = None):
        """
        Log an artifact (model checkpoint, dataset, etc.).
        
        Args:
            artifact_path: Path to artifact
            artifact_type: Type of artifact (model, dataset, etc.)
            name: Artifact name (defaults to filename)
        """
        if self.run:
            artifact_path = Path(artifact_path)
            name = name or artifact_path.stem
            
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_file(str(artifact_path))
            wandb.log_artifact(artifact)
            logger.info(f"Logged artifact: {name}")
    
    def log_evaluation_results(self, results: Dict, prefix: str = ""):
        """
        Log evaluation results from DualEvaluator.
        
        Logs three categories of metrics:
        1. Taiwan metrics (target domain)
        2. COCO full metrics (source domain)
        3. COCO subset metrics (traffic classes only)
        
        Args:
            results: Results dictionary from DualEvaluator
            prefix: Prefix for metric names (e.g., "final/")
        """
        if not self.run:
            return
        
        # Log Taiwan metrics
        taiwan = results.get("taiwan", {})
        self.log_metrics({
            f"{prefix}taiwan/mAP50-95": taiwan.get("mAP50-95", 0),
            f"{prefix}taiwan/mAP50": taiwan.get("mAP50", 0),
            f"{prefix}taiwan/mAP75": taiwan.get("mAP75", 0),
            f"{prefix}taiwan/precision": taiwan.get("precision", 0),
            f"{prefix}taiwan/recall": taiwan.get("recall", 0),
        })
        
        # Log COCO full metrics
        coco = results.get("coco", {})
        self.log_metrics({
            f"{prefix}coco_full/mAP50-95": coco.get("mAP50-95", 0),
            f"{prefix}coco_full/mAP50": coco.get("mAP50", 0),
            f"{prefix}coco_full/mAP75": coco.get("mAP75", 0),
            f"{prefix}coco_full/precision": coco.get("precision", 0),
            f"{prefix}coco_full/recall": coco.get("recall", 0),
        })
        
        # Log COCO subset metrics (traffic classes)
        if "traffic_subset" in coco:
            traffic_subset = coco["traffic_subset"]
            self.log_metrics({
                f"{prefix}coco_subset/mAP50-95": traffic_subset.get("mean_mAP", 0),
            })
            # Log individual traffic class metrics
            for class_name, class_map in traffic_subset.get("individual_mAP", {}).items():
                self.log_metrics({
                    f"{prefix}coco_subset/{class_name}_mAP": class_map
                })
        
        # Log forgetting metrics
        forgetting = results.get("forgetting", {})
        self.log_metrics({
            f"{prefix}forgetting/coco_full_absolute": forgetting.get("absolute_forgetting", 0),
            f"{prefix}forgetting/coco_full_relative_pct": forgetting.get("relative_forgetting_pct", 0),
            f"{prefix}forgetting/coco_full_retained_pct": forgetting.get("retained_pct", 0),
        })
        
        # Log subset forgetting if available
        if "traffic_subset" in forgetting:
            subset_forgetting = forgetting["traffic_subset"]
            self.log_metrics({
                f"{prefix}forgetting/coco_subset_absolute": subset_forgetting.get("absolute_forgetting", 0),
                f"{prefix}forgetting/coco_subset_relative_pct": subset_forgetting.get("relative_forgetting_pct", 0),
            })
        
        # Log summary metrics
        summary = results.get("summary", {})
        self.log_metrics({
            f"{prefix}summary/taiwan_improvement": summary.get("taiwan_improvement", 0),
            f"{prefix}summary/taiwan_improvement_pct": summary.get("taiwan_improvement_pct", 0),
            f"{prefix}summary/coco_subset_mAP": summary.get("coco_subset_mAP", 0),
        })
    
    def log_table(self, name: str, data: list, columns: list):
        """Log a table to W&B."""
        if self.run:
            table = wandb.Table(data=data, columns=columns)
            wandb.log({name: table})
    
    def finish(self):
        """Finish the W&B run."""
        if self.run:
            wandb.finish()
            logger.info("W&B run finished")


def setup_wandb(
    experiment_name: str,
    config: Dict,
    project: str = "scooter-yolo-domain-adaptation",
    tags: Optional[list] = None,
    mode: str = "online"
) -> WandBLogger:
    """
    Quick setup for W&B logging.
    
    Args:
        experiment_name: Name of the experiment
        config: Configuration dictionary
        project: W&B project name
        tags: List of tags
        mode: Logging mode
        
    Returns:
        WandBLogger instance
    """
    return WandBLogger(
        project=project,
        name=experiment_name,
        config=config,
        tags=tags,
        mode=mode
    )
