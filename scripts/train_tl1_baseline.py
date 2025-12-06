#!/usr/bin/env python3
"""
Training script for TL-1: Standard Fine-Tuning baseline.

This script runs the TL-1 baseline experiments:
- Learning rate ablation (1e-5, 1e-4, 1e-3)
- Early stopping handles optimal epoch selection automatically

Usage:
    # Run single experiment
    python scripts/train_tl1_baseline.py --experiment tl1_lr_1e4
    
    # Run all LR ablations
    python scripts/train_tl1_baseline.py --ablation all
    
    # List available experiments
    python scripts/train_tl1_baseline.py --list
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trainers import StandardFineTuner
from src.evaluation import DualEvaluator
from src.configs.tl1_configs import (
    get_config, list_experiments, LR_ABLATION, FULL_ABLATION
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_experiment(config: dict, evaluate: bool = True, use_wandb: bool = True):
    """
    Run a single experiment.
    
    Args:
        config: Experiment configuration dictionary
        evaluate: Whether to run dual-domain evaluation
        use_wandb: Whether to log to Weights & Biases
    """
    experiment_name = config["name"]
    logger.info(f"\n{'='*80}")
    logger.info(f"Running experiment: {experiment_name}")
    logger.info(f"Description: {config.get('description', 'N/A')}")
    logger.info(f"{'='*80}\n")
    
    # Extract trainer arguments (keep 'name', remove only 'description')
    trainer_args = {k: v for k, v in config.items() 
                   if k not in ['description']}
    
    # Add W&B project if enabled (YOLO has built-in W&B support)
    if use_wandb:
        # trainer_args["wandb_project"] = "YOLO Taiwan Street"
        # Set environment variable directly to ensure it overrides YOLO's default
        import os
        from ultralytics import settings
        import wandb
        
        # Force enable W&B in Ultralytics settings
        settings.update({"wandb": True})
        os.environ["WANDB_PROJECT"] = "YOLO Taiwan Street"
        
        # Initialize W&B run manually to force project name
        # This prevents Ultralytics from using the 'project' arg (save dir) as the W&B project name
        if wandb.run is None:
            wandb.init(
                project="YOLO Taiwan Street",
                name=experiment_name,
                config=config,
                tags=["tl1", "baseline", "lr_ablation"],
                resume="allow"
            )
        
        logger.info("✓ W&B logging enabled (Project: YOLO Taiwan Street)")
    
    # Initialize trainer
    trainer = StandardFineTuner(**trainer_args)
    
    # Train
    try:
        results = trainer.train()
        logger.info(f"✓ Training completed: {experiment_name}")
        
        # Evaluate on both domains
        if evaluate:
            logger.info(f"\nEvaluating {experiment_name} on both domains...")
            checkpoint = trainer.get_best_checkpoint()
            
            evaluator = DualEvaluator(
                taiwan_yaml=config["data_yaml"],
                device=config["device"]
            )
            
            eval_results = evaluator.evaluate_model(
                model_path=str(checkpoint),
                experiment_name=experiment_name
            )
            
            logger.info(f"✓ Evaluation completed: {experiment_name}\n")
            
        if use_wandb:
            import wandb
            if wandb.run:
                wandb.finish()
            
        return True
        
    except Exception as e:
        logger.error(f"✗ Experiment failed: {experiment_name}")
        logger.error(f"Error: {str(e)}")
        if use_wandb:
            import wandb
            if wandb.run:
                wandb.finish()
        return False


def run_ablation(ablation_type: str, evaluate: bool = True, use_wandb: bool = True):
    """
    Run ablation experiments.
    
    Args:
        ablation_type: 'lr' or 'all' (both mean the same now)
        evaluate: Whether to run evaluation
        use_wandb: Whether to log to W&B
    """
    if ablation_type in ['lr', 'all']:
        experiments = LR_ABLATION
        logger.info("Running Learning Rate Ablation (3 experiments)")
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")
    
    logger.info(f"Experiments to run: {list(experiments.keys())}\n")
    
    success_count = 0
    total_count = len(experiments)
    
    for exp_name, config in experiments.items():
        success = run_experiment(config, evaluate=evaluate, use_wandb=use_wandb)
        if success:
            success_count += 1
    
    logger.info(f"\n{'='*80}")
    logger.info(f"ABLATION COMPLETE")
    logger.info(f"Successful: {success_count}/{total_count}")
    logger.info(f"{'='*80}\n")
    
    # Generate comparison if all succeeded
    if success_count == total_count and evaluate:
        logger.info("Generating experiment comparison...")
        evaluator = DualEvaluator()
        comparison = evaluator.compare_experiments(list(experiments.keys()))
        
        logger.info("\nEXPERIMENT RANKINGS (by Taiwan mAP):")
        logger.info("-" * 80)
        for exp in comparison.get("experiments", []):
            logger.info(
                f"{exp['rank']}. {exp['name']:20s} | "
                f"Taiwan: {exp['taiwan_mAP']:.1%} | "
                f"COCO: {exp['coco_mAP']:.1%} | "
                f"Forgetting: {exp['forgetting_pct']:.1f}%"
            )
        logger.info("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Train TL-1 baseline experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        help='Single experiment name to run'
    )
    
    parser.add_argument(
        '--ablation',
        type=str,
        choices=['lr', 'all'],
        help='Run LR ablation (lr and all are equivalent)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available experiments'
    )
    
    parser.add_argument(
        '--no-eval',
        action='store_true',
        help='Skip dual-domain evaluation (only train)'
    )
    
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    
    args = parser.parse_args()
    
    # List experiments
    if args.list:
        print("Available experiments:")
        for exp_name in list_experiments():
            config = get_config(exp_name)
            print(f"  {exp_name:20s} - {config.get('description', 'N/A')}")
        return
    
    evaluate = not args.no_eval
    use_wandb = not args.no_wandb
    
    # Run single experiment
    if args.experiment:
        config = get_config(args.experiment)
        run_experiment(config, evaluate=evaluate, use_wandb=use_wandb)
    
    # Run ablation
    elif args.ablation:
        run_ablation(args.ablation, evaluate=evaluate, use_wandb=use_wandb)
    
    # No arguments provided
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
