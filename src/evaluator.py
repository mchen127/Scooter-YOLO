"""
Model evaluator for YOLO models.
"""
from pathlib import Path
from typing import Dict, Any, Optional, List
from ultralytics import YOLO

from .utils import save_results_json, print_summary


class ModelEvaluator:
    """
    Evaluator class for YOLO models on various datasets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluator with configuration.
        
        Args:
            config: Configuration dictionary containing model and dataset paths
        """
        self.config = config
        self.model = None
        self.results = None
        
    def load_model(self) -> None:
        """Load the YOLO model from the specified path."""
        model_path = self.config["model_path"]
        print(f"Loading model from: {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = YOLO(model_path)
        print(f"✓ Model loaded successfully")
        
    def evaluate(self) -> Dict[str, Any]:
        """
        Run evaluation on the configured dataset.
        
        Returns:
            Dictionary containing evaluation results
        """
        if self.model is None:
            self.load_model()
        
        print(f"\nStarting evaluation on {self.config['name']}...")
        print(f"Dataset: {self.config['data_yaml']}")
        
        # Run validation
        results = self.model.val(
            data=self.config["data_yaml"],
            split=self.config["split"],
            imgsz=self.config["imgsz"],
            batch=self.config["batch"],
            device=self.config["device"],
            save_json=self.config["save_json"],
            save_hybrid=self.config["save_hybrid"],
            conf=self.config["conf"],
            iou=self.config["iou"],
            max_det=self.config["max_det"],
            half=self.config["half"],
            verbose=self.config["verbose"],
        )
        
        # Extract per-class metrics
        per_class_metrics = self._extract_per_class_metrics(results)
        
        # Calculate subset mAP for common traffic classes
        traffic_classes = ['person', 'bicycle', 'car', 'motorcycle', 'truck']
        subset_map = self._calculate_subset_map(per_class_metrics, traffic_classes)
        
        # Extract metrics
        self.results = {
            "config": self.config,
            "results_dict": results.results_dict,
            "speed": results.speed,
            "maps": results.maps.tolist() if hasattr(results.maps, 'tolist') else None,
            "per_class_metrics": per_class_metrics,
            "traffic_subset_mAP": {
                "classes": traffic_classes,
                "mAP50-95": subset_map,
                "description": "Mean mAP over traffic-relevant classes for fair comparison"
            }
        }
        
        print(f"\n✓ Evaluation completed")
        
        return self.results
    
    def _extract_per_class_metrics(self, results) -> Dict[str, Dict[str, float]]:
        """
        Extract per-class metrics from YOLO results.
        
        Args:
            results: YOLO validation results object
            
        Returns:
            Dictionary mapping class names to their metrics
        """
        per_class = {}
        
        # Get class names from the model
        class_names = self.model.names  # Dict: {0: 'person', 1: 'bicycle', ...}
        
        # Extract per-class AP (Average Precision)
        if hasattr(results, 'maps') and results.maps is not None:
            maps = results.maps if isinstance(results.maps, list) else results.maps.tolist()
            
            for class_id, ap in enumerate(maps):
                if class_id in class_names:
                    class_name = class_names[class_id]
                    per_class[class_name] = {
                        "class_id": class_id,
                        "mAP50-95": float(ap),
                    }
        
        # Try to extract additional per-class metrics if available
        if hasattr(results, 'box') and hasattr(results.box, 'ap_class_index'):
            # Get precision, recall per class
            if hasattr(results.box, 'p'):  # precision per class
                for idx, class_id in enumerate(results.box.ap_class_index):
                    class_id = int(class_id)
                    if class_id in class_names:
                        class_name = class_names[class_id]
                        if class_name not in per_class:
                            per_class[class_name] = {"class_id": class_id}
                        
                        if idx < len(results.box.p):
                            per_class[class_name]["precision"] = float(results.box.p[idx])
                        if hasattr(results.box, 'r') and idx < len(results.box.r):
                            per_class[class_name]["recall"] = float(results.box.r[idx])
                        if hasattr(results.box, 'ap50') and idx < len(results.box.ap50):
                            per_class[class_name]["mAP50"] = float(results.box.ap50[idx])
        
        return per_class
    
    def _calculate_subset_map(
        self, 
        per_class_metrics: Dict[str, Dict[str, float]], 
        class_subset: List[str]
    ) -> float:
        """
        Calculate mean mAP over a subset of classes.
        
        Args:
            per_class_metrics: Dictionary of per-class metrics
            class_subset: List of class names to include
            
        Returns:
            Mean mAP50-95 over the specified classes
        """
        maps = []
        for class_name in class_subset:
            if class_name in per_class_metrics:
                map_value = per_class_metrics[class_name].get('mAP50-95', 0.0)
                maps.append(map_value)
        
        return sum(maps) / len(maps) if maps else 0.0
    
    def save_results(self, output_path: str) -> None:
        """
        Save evaluation results to a JSON file.
        
        Args:
            output_path: Path to save the results JSON file
        """
        if self.results is None:
            raise ValueError("No results to save. Run evaluate() first.")
        
        output_path = Path(output_path)
        save_results_json(self.results, output_path)
        
    def print_summary(self) -> None:
        """Print evaluation summary."""
        if self.results is None:
            raise ValueError("No results to print. Run evaluate() first.")
        
        print_summary(self.config, self.results)
