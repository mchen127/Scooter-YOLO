#!/usr/bin/env python3
"""
Baseline Evaluation Script

Evaluates the pretrained YOLO11m model on both COCO and Taiwan Traffic datasets
to establish baseline performance before any fine-tuning.

This provides the reference point for measuring:
- Domain shift (COCO → Taiwan performance gap)
- Improvement from fine-tuning
- Forgetting after adaptation

Usage:
    python scripts/evaluate_baseline.py
    python scripts/evaluate_baseline.py --model models/yolo11m.pt
    python scripts/evaluate_baseline.py --wandb  # Log to W&B
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import DualEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline model performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/yolo11m.pt',
        help='Path to model checkpoint (default: models/yolo11m.pt)'
    )
    
    parser.add_argument(
        '--taiwan-yaml',
        type=str,
        default='datasets/taiwan-traffic/data.yaml',
        help='Path to Taiwan dataset yaml'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device to use (default: 0 for GPU 0)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/baseline',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Log results to Weights & Biases'
    )
    
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='scooter-yolo-domain-adaptation',
        help='W&B project name'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("BASELINE EVALUATION")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Taiwan dataset: {args.taiwan_yaml}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*80)
    
    # Initialize evaluator
    evaluator = DualEvaluator(
        taiwan_yaml=args.taiwan_yaml,
        results_dir=str(output_dir),
        device=args.device
    )
    
    # Run evaluation
    logger.info("\nEvaluating pretrained model on both domains...")
    results = evaluator.evaluate_model(
        model_path=args.model,
        experiment_name="baseline_pretrained",
        save_results=True
    )
    
    # Log to W&B if requested
    if args.wandb:
        try:
            from src.utils.wandb_logger import setup_wandb
            
            logger.info("\nLogging to Weights & Biases...")
            wandb_logger = setup_wandb(
                experiment_name="baseline_pretrained",
                config={
                    "model": args.model,
                    "taiwan_dataset": args.taiwan_yaml,
                    "device": args.device,
                    "experiment_type": "baseline_evaluation"
                },
                project=args.wandb_project,
                tags=["baseline", "pretrained", "evaluation"],
                mode="online"
            )
            
            # Log results
            wandb_logger.log_evaluation_results(results, prefix="baseline/")
            
            # Log model as artifact
            wandb_logger.log_artifact(
                artifact_path=args.model,
                artifact_type="model",
                name="yolo11m_pretrained"
            )
            
            wandb_logger.finish()
            logger.info("✓ Results logged to W&B")
            
        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("BASELINE EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {output_dir}/baseline_pretrained.json")
    logger.info("\nKey Metrics:")
    logger.info(f"  Taiwan mAP50-95:  {results['taiwan']['mAP50-95']:.1%}")
    logger.info(f"  COCO mAP50-95:    {results['coco']['mAP50-95']:.1%}")
    logger.info(f"  Performance Gap:  {results['coco']['mAP50-95'] - results['taiwan']['mAP50-95']:.1%}")
    logger.info("="*80)
    
    return results


if __name__ == "__main__":
    main()
