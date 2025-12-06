"""
BaseTrainer: Abstract base class for all training strategies.

Provides common functionality for model training, validation, and checkpoint management.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any
from abc import ABC, abstractmethod
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base class for all training strategies.
    
    Subclasses should implement the train() method to define
    specific training logic (e.g., fine-tuning, freezing, etc.)
    """
    
    def __init__(
        self,
        model_path: str,
        data_yaml: str,
        project: str = "runs/train",
        name: str = "exp",
        device: str = "0",
        **kwargs
    ):
        """
        Initialize BaseTrainer.
        
        Args:
            model_path: Path to pretrained model (.pt file)
            data_yaml: Path to dataset configuration yaml
            project: Project directory for runs
            name: Experiment name
            device: Device to use ('0' for GPU, 'cpu' for CPU)
            **kwargs: Additional training arguments
        """
        self.model_path = Path(model_path)
        self.data_yaml = data_yaml
        self.project = project
        self.name = name
        self.device = device
        self.kwargs = kwargs
        
        # Load model
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the YOLO model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info(f"Loading model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        
    @abstractmethod
    def train(self, **train_args) -> Dict:
        """
        Train the model. Must be implemented by subclasses.
        
        Args:
            **train_args: Training arguments (epochs, lr, etc.)
            
        Returns:
            Training results dictionary
        """
        pass
    
    def get_training_args(self, **overrides) -> Dict:
        """
        Get default training arguments, with optional overrides.
        
        Args:
            **overrides: Arguments to override defaults
            
        Returns:
            Dictionary of training arguments
        """
        defaults = {
            "data": self.data_yaml,
            "epochs": 30,
            "imgsz": 640,
            "batch": 16,
            "device": self.device,
            "project": self.project,
            "name": self.name,
            "exist_ok": True,
            "verbose": True,
            "patience": 5,  # Early stopping
            "save": True,
            "plots": True,
        }
        
        # Merge with class kwargs
        defaults.update(self.kwargs)
        
        # Apply overrides
        defaults.update(overrides)
        
        # Enable W&B if project name is provided
        if "wandb_project" in defaults:
            wandb_project = defaults.pop("wandb_project")
            # Enable W&B through YOLO settings
            try:
                from ultralytics import settings
                settings.update({"wandb": True})
                import os
                os.environ["WANDB_PROJECT"] = wandb_project
                logger.info(f"W&B enabled for project: {wandb_project}")
            except Exception as e:
                logger.warning(f"Failed to enable W&B: {e}")
        
        return defaults
    
    def get_best_checkpoint(self) -> Path:
        """
        Get path to best checkpoint from training.
        
        Returns:
            Path to best.pt file
        """
        checkpoint_path = Path(self.project) / self.name / "weights" / "best.pt"
        if not checkpoint_path.exists():
            logger.warning(f"Best checkpoint not found: {checkpoint_path}")
            checkpoint_path = Path(self.project) / self.name / "weights" / "last.pt"
        
        return checkpoint_path
    
    def validate(self, data_yaml: Optional[str] = None) -> Dict:
        """
        Run validation on a dataset.
        
        Args:
            data_yaml: Dataset yaml (uses training dataset if None)
            
        Returns:
            Validation results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        data_yaml = data_yaml or self.data_yaml
        logger.info(f"Validating on: {data_yaml}")
        
        results = self.model.val(
            data=data_yaml,
            batch=128,
            imgsz=640,
            device=self.device,
            verbose=False
        )
        
        return {
            "mAP50-95": float(results.box.map),
            "mAP50": float(results.box.map50),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        }
