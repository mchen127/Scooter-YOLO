"""
StandardFineTuner: TL-1 baseline implementation.

Performs standard fine-tuning by updating all model weights on the target domain.
"""

import logging
from typing import Dict, Any
from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class StandardFineTuner(BaseTrainer):
    """
    TL-1: Standard Fine-Tuning (The Baseline)
    
    Updates all weights (Backbone + Head) on the target domain dataset.
    No freezing, no special techniques - just straightforward fine-tuning.
    
    Expected Behavior:
    - Highest accuracy on target domain (Taiwan)
    - Highest forgetting on source domain (COCO)
    - Risk of overfitting on small datasets
    """
    
    def __init__(
        self,
        model_path: str,
        data_yaml: str,
        lr0: float = 0.01,
        epochs: int = 30,
        **kwargs
    ):
        """
        Initialize StandardFineTuner.
        
        Args:
            model_path: Path to pretrained model
            data_yaml: Path to target domain dataset yaml
            lr0: Initial learning rate
            epochs: Number of training epochs
            **kwargs: Additional training arguments
        """
        super().__init__(model_path, data_yaml, **kwargs)
        self.lr0 = lr0
        self.epochs = epochs
        
    def train(self, **overrides) -> Dict:
        """
        Train with standard fine-tuning.
        
        Args:
            **overrides: Override default training arguments
            
        Returns:
            Training results dictionary
        """
        logger.info("="*70)
        logger.info("TL-1: Standard Fine-Tuning")
        logger.info("="*70)
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Data: {self.data_yaml}")
        logger.info(f"Learning Rate: {self.lr0}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info("Strategy: Update all weights (no freezing)")
        logger.info("="*70)
        
        # Prepare training arguments
        train_args = self.get_training_args(
            epochs=self.epochs,
            lr0=self.lr0,
            **overrides
        )
        
        # Add custom callbacks if W&B is enabled
        # This allows logging COCO metrics during training
        import os
        is_wandb_enabled = os.environ.get("WANDB_PROJECT") or "wandb_project" in self.kwargs
        
        if train_args.get("project") and is_wandb_enabled:
            from src.utils.callbacks import get_dual_eval_callback
            
            # Create callback
            callback = get_dual_eval_callback(self.name)
            
            # Register callback
            self.model.add_callback("on_fit_epoch_end", callback)
            logger.info("Registered dual validation callback (COCO eval per epoch)")
        
        # Train the model
        logger.info("Starting training...")
        results = self.model.train(**train_args)
        
        logger.info("Training completed!")
        logger.info(f"Best checkpoint: {self.get_best_checkpoint()}")
        
        return results


# Convenience function for quick training
def train_standard_finetuning(
    model_path: str,
    data_yaml: str,
    lr0: float = 0.01,
    epochs: int = 30,
    project: str = "runs/train",
    name: str = "tl1_standard",
    device: str = "0",
    **kwargs
) -> Dict:
    """
    Quick interface for standard fine-tuning.
    
    Args:
        model_path: Path to pretrained model
        data_yaml: Path to dataset yaml
        lr0: Learning rate
        epochs: Number of epochs
        project: Project directory
        name: Experiment name
        device: Training device
        **kwargs: Additional arguments
        
    Returns:
        Training results
    """
    trainer = StandardFineTuner(
        model_path=model_path,
        data_yaml=data_yaml,
        lr0=lr0,
        epochs=epochs,
        project=project,
        name=name,
        device=device,
        **kwargs
    )
    return trainer.train()
