"""
DualEvaluator: Evaluate models on both COCO and Taiwan Traffic datasets.

This module provides comprehensive evaluation across source and target domains,
tracking forgetting metrics and performance on traffic-related classes.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from ultralytics import YOLO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DualEvaluator:
    """
    Evaluates YOLO models on both COCO and Taiwan Traffic datasets.
    
    Tracks:
    - Target domain (Taiwan) performance
    - Source domain (COCO) performance
    - Forgetting metrics
    - Per-class performance on traffic classes
    """
    
    # Traffic-related COCO class IDs
    TRAFFIC_CLASSES = [0, 1, 2, 3, 7]  # person, bicycle, car, motorcycle, truck
    TRAFFIC_CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'truck']
    
    def __init__(
        self,
        coco_yaml: str = "coco.yaml",
        taiwan_yaml: str = None,
        results_dir: str = "results/experiments",
        device: str = "0"
    ):
        """
        Initialize DualEvaluator.
        
        Args:
            coco_yaml: Path to COCO dataset yaml
            taiwan_yaml: Path to Taiwan Traffic dataset yaml
            results_dir: Directory to save evaluation results
            device: Device to run evaluation on ('0' for GPU 0, 'cpu' for CPU)
        """
        self.coco_yaml = coco_yaml
        self.taiwan_yaml = taiwan_yaml or "datasets/taiwan-traffic/data.yaml"
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Baseline performance (from pretrained model)
        self.baseline_coco_map = 0.513  # 51.3% from your results
        self.baseline_taiwan_map = 0.314  # 31.4% from your results
        
    def evaluate_model(
        self,
        model_path: str,
        experiment_name: str,
        save_results: bool = True
    ) -> Dict:
        """
        Evaluate a model on both domains.
        
        Args:
            model_path: Path to model checkpoint (.pt file)
            experiment_name: Name of the experiment (for logging)
            save_results: Whether to save results to JSON
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Evaluating {experiment_name}: {model_path}")
        
        # Load model
        model = YOLO(model_path)
        
        # Evaluate on Taiwan Traffic
        logger.info("Evaluating on Taiwan Traffic dataset...")
        taiwan_results = self._evaluate_on_dataset(
            model, self.taiwan_yaml, "Taiwan-Traffic"
        )
        
        # Evaluate on COCO
        logger.info("Evaluating on COCO dataset...")
        coco_results = self._evaluate_on_dataset(
            model, self.coco_yaml, "COCO"
        )
        
        # Compute forgetting metrics
        forgetting_metrics = self._compute_forgetting(coco_results)
        
        # Compile full results
        full_results = {
            "experiment_name": experiment_name,
            "model_path": str(model_path),
            "timestamp": datetime.now().isoformat(),
            "taiwan": taiwan_results,
            "coco": coco_results,
            "forgetting": forgetting_metrics,
            "summary": self._create_summary(taiwan_results, coco_results, forgetting_metrics)
        }
        
        # Save results
        if save_results:
            self._save_results(full_results, experiment_name)
            
        # Log summary
        self._log_summary(full_results)
        
        return full_results
    
    def _evaluate_on_dataset(
        self,
        model: YOLO,
        data_yaml: str,
        dataset_name: str
    ) -> Dict:
        """Evaluate model on a single dataset."""
        # Run validation
        metrics = model.val(
            data=data_yaml,
            batch=128,
            imgsz=640,
            device=self.device,
            conf=0.001,
            iou=0.6,
            max_det=300,
            half=True,
            verbose=False
        )
        
        # Extract metrics
        results = {
            "dataset": dataset_name,
            "mAP50-95": float(metrics.box.map),
            "mAP50": float(metrics.box.map50),
            "mAP75": float(metrics.box.map75),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
        
        # Add per-class metrics
        if hasattr(metrics.box, 'maps'):
            results["per_class_mAP"] = [float(m) for m in metrics.box.maps]
            
            # Extract traffic class performance
            if dataset_name == "COCO" and len(results["per_class_mAP"]) >= 80:
                traffic_maps = [results["per_class_mAP"][i] for i in self.TRAFFIC_CLASSES]
                results["traffic_subset"] = {
                    "classes": self.TRAFFIC_CLASS_NAMES,
                    "individual_mAP": dict(zip(self.TRAFFIC_CLASS_NAMES, traffic_maps)),
                    "mean_mAP": sum(traffic_maps) / len(traffic_maps)
                }
        
        return results
    
    def _compute_forgetting(self, coco_results: Dict) -> Dict:
        """
        Compute forgetting metrics.
        
        Forgetting = Baseline Performance - Current Performance
        """
        current_map = coco_results["mAP50-95"]
        absolute_forgetting = self.baseline_coco_map - current_map
        relative_forgetting = (absolute_forgetting / self.baseline_coco_map) * 100
        
        forgetting = {
            "baseline_mAP": self.baseline_coco_map,
            "current_mAP": current_map,
            "absolute_forgetting": absolute_forgetting,
            "relative_forgetting_pct": relative_forgetting,
            "retained_pct": 100 - relative_forgetting
        }
        
        # Traffic subset forgetting (if available)
        if "traffic_subset" in coco_results:
            traffic_baseline = 0.515  # 51.5% from your baseline
            traffic_current = coco_results["traffic_subset"]["mean_mAP"]
            traffic_forgetting = traffic_baseline - traffic_current
            
            forgetting["traffic_subset"] = {
                "baseline_mAP": traffic_baseline,
                "current_mAP": traffic_current,
                "absolute_forgetting": traffic_forgetting,
                "relative_forgetting_pct": (traffic_forgetting / traffic_baseline) * 100
            }
        
        return forgetting
    
    def _create_summary(
        self,
        taiwan_results: Dict,
        coco_results: Dict,
        forgetting_metrics: Dict
    ) -> Dict:
        """Create a concise summary of results."""
        taiwan_improvement = taiwan_results["mAP50-95"] - self.baseline_taiwan_map
        
        summary = {
            "taiwan_mAP": taiwan_results["mAP50-95"],
            "taiwan_improvement": taiwan_improvement,
            "taiwan_improvement_pct": (taiwan_improvement / self.baseline_taiwan_map) * 100,
            "coco_mAP": coco_results["mAP50-95"],
            "forgetting_pct": forgetting_metrics["relative_forgetting_pct"],
            "retained_pct": forgetting_metrics["retained_pct"]
        }
        
        # Add COCO subset metrics if available
        if "traffic_subset" in coco_results:
            summary["coco_subset_mAP"] = coco_results["traffic_subset"]["mean_mAP"]
            if "traffic_subset" in forgetting_metrics:
                summary["coco_subset_forgetting_pct"] = forgetting_metrics["traffic_subset"]["relative_forgetting_pct"]
        
        return summary
    
    def _save_results(self, results: Dict, experiment_name: str):
        """Save results to JSON file."""
        output_path = self.results_dir / f"{experiment_name}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")
    
    def _log_summary(self, results: Dict):
        """Log a formatted summary of results."""
        summary = results["summary"]
        forgetting = results["forgetting"]
        coco_results = results["coco"]
        
        logger.info("\n" + "="*70)
        logger.info(f"EVALUATION SUMMARY: {results['experiment_name']}")
        logger.info("="*70)
        logger.info(f"Taiwan Traffic:")
        logger.info(f"  mAP50-95: {summary['taiwan_mAP']:.1%}")
        logger.info(f"  Improvement: {summary['taiwan_improvement']:+.1%} ({summary['taiwan_improvement_pct']:+.1f}%)")
        logger.info(f"\nCOCO (Full):")
        logger.info(f"  mAP50-95: {summary['coco_mAP']:.1%}")
        logger.info(f"  Forgetting: {forgetting['absolute_forgetting']:.1%} ({summary['forgetting_pct']:.1f}%)")
        logger.info(f"  Retained: {summary['retained_pct']:.1f}%")
        
        # Log COCO subset if available
        if "traffic_subset" in coco_results:
            logger.info(f"\nCOCO (Traffic Subset):")
            logger.info(f"  mAP50-95: {coco_results['traffic_subset']['mean_mAP']:.1%}")
            if "traffic_subset" in forgetting:
                logger.info(f"  Forgetting: {forgetting['traffic_subset']['absolute_forgetting']:.1%} ({forgetting['traffic_subset']['relative_forgetting_pct']:.1f}%)")
        
        logger.info("="*70 + "\n")
    
    def compare_experiments(
        self,
        experiment_names: List[str]
    ) -> Dict:
        """
        Compare multiple experiments.
        
        Args:
            experiment_names: List of experiment names to compare
            
        Returns:
            Comparison dictionary with rankings
        """
        results = []
        
        for exp_name in experiment_names:
            result_file = self.results_dir / f"{exp_name}.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    results.append(json.load(f))
            else:
                logger.warning(f"Results not found for: {exp_name}")
        
        if not results:
            logger.error("No results to compare")
            return {}
        
        # Sort by Taiwan performance
        results.sort(key=lambda x: x["summary"]["taiwan_mAP"], reverse=True)
        
        comparison = {
            "sorted_by": "taiwan_mAP",
            "experiments": [
                {
                    "rank": i + 1,
                    "name": r["experiment_name"],
                    "taiwan_mAP": r["summary"]["taiwan_mAP"],
                    "coco_mAP": r["summary"]["coco_mAP"],
                    "forgetting_pct": r["summary"]["forgetting_pct"]
                }
                for i, r in enumerate(results)
            ]
        }
        
        return comparison


# Convenience function
def evaluate_model(
    model_path: str,
    experiment_name: str,
    taiwan_yaml: str = None,
    device: str = "0"
) -> Dict:
    """
    Quick evaluation function.
    
    Args:
        model_path: Path to model checkpoint
        experiment_name: Name of experiment
        taiwan_yaml: Path to Taiwan dataset yaml (optional)
        device: Device to use
        
    Returns:
        Evaluation results dictionary
    """
    evaluator = DualEvaluator(
        taiwan_yaml=taiwan_yaml,
        device=device
    )
    return evaluator.evaluate_model(model_path, experiment_name)
