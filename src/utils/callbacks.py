"""
Custom callbacks for YOLO training.

Provides callbacks for:
- Dual-domain evaluation (COCO + Taiwan) during training
- W&B logging of additional metrics
"""

import logging
from ultralytics import YOLO
from src.evaluation.dual_evaluator import DualEvaluator

logger = logging.getLogger(__name__)

def get_dual_eval_callback(experiment_name: str, coco_yaml: str = "coco.yaml"):
    """
    Create a callback for dual-domain evaluation at the end of each epoch.
    
    Args:
        experiment_name: Name of the experiment
        coco_yaml: Path to COCO dataset yaml
        
    Returns:
        Callback function
    """
    
    def on_fit_epoch_end(trainer):
        """
        Callback to run after each training epoch.
        
        Evaluates on COCO dataset and logs metrics to W&B.
        (Taiwan evaluation is handled automatically by the trainer)
        """
        # Only run every epoch (or configure interval)
        # Note: COCO eval is slow, so we might want to restrict this
        # But user requested "all the time", so we run every epoch
        
        try:
            current_epoch = trainer.epoch + 1
            logger.info(f"Running COCO evaluation for epoch {current_epoch}...")
            
            # Load the current model state (last.pt)
            # trainer.model is the raw PyTorch model, we need the YOLO wrapper
            last_pt = trainer.save_dir / 'weights' / 'last.pt'
            
            if not last_pt.exists():
                logger.warning(f"last.pt not found at {last_pt}, skipping COCO eval")
                return
                
            # Initialize model with current weights
            model = YOLO(str(last_pt))
            
            # Run COCO evaluation
            metrics = model.val(
                data=coco_yaml,
                batch=64,
                imgsz=640,
                device=trainer.device,
                conf=0.001,
                iou=0.6,
                max_det=300,
                half=True,
                verbose=False,
                plots=False
            )
            
            # Extract COCO full metrics
            coco_map50_95 = float(metrics.box.map)
            coco_map50 = float(metrics.box.map50)
            
            # Extract COCO subset metrics (Traffic classes: person, bicycle, car, motorcycle, truck)
            traffic_classes = [0, 1, 2, 3, 7]
            
            # Find indices of traffic classes in the results
            traffic_indices = [
                i for i, cls_id in enumerate(metrics.box.ap_class_index)
                if cls_id in traffic_classes
            ]
            
            subset_map50 = 0.0
            subset_map50_95 = 0.0
            
            if traffic_indices:
                # all_ap is (NC, 10), index 0 is 0.5, mean(1) is 0.50-0.95
                subset_map50 = metrics.box.all_ap[traffic_indices, 0].mean()
                subset_map50_95 = metrics.box.all_ap[traffic_indices, :].mean()
            
            # Log to W&B
            # Use global wandb instance which is more reliable
            import wandb
            if wandb.run:
                # Prepare log dict
                log_dict = {
                    "metrics/coco_full_mAP50-95": coco_map50_95,
                    "metrics/coco_full_mAP50": coco_map50,
                    "metrics/coco_subset_mAP50-95": subset_map50_95,
                    "metrics/coco_subset_mAP50": subset_map50,
                }
                
                # Also log Taiwan metrics from trainer (if available)
                # trainer.metrics contains keys like 'metrics/mAP50-95(B)'
                if hasattr(trainer, 'metrics') and trainer.metrics:
                    metrics = trainer.metrics
                    
                    # Map mAP metrics
                    if 'metrics/mAP50-95(B)' in metrics:
                        log_dict["metrics/taiwan_mAP50-95"] = metrics['metrics/mAP50-95(B)']
                    if 'metrics/mAP50(B)' in metrics:
                        log_dict["metrics/taiwan_mAP50"] = metrics['metrics/mAP50(B)']
                    
                    # Map Precision/Recall
                    if 'metrics/precision(B)' in metrics:
                        log_dict["metrics/taiwan_precision"] = metrics['metrics/precision(B)']
                    if 'metrics/recall(B)' in metrics:
                        log_dict["metrics/taiwan_recall"] = metrics['metrics/recall(B)']
                        
                    # Map Losses (if available in metrics, usually they are)
                    if 'val/box_loss' in metrics:
                        log_dict["val/taiwan_box_loss"] = metrics['val/box_loss']
                    if 'val/cls_loss' in metrics:
                        log_dict["val/taiwan_cls_loss"] = metrics['val/cls_loss']
                    if 'val/dfl_loss' in metrics:
                        log_dict["val/taiwan_dfl_loss"] = metrics['val/dfl_loss']
                
                wandb.log(log_dict)
                logger.debug(f"Logged all metrics to W&B run: {wandb.run.name}")
            else:
                logger.warning("No active W&B run found, skipping logging")
                
            logger.info(f"COCO Eval - Epoch {current_epoch}: Full mAP={coco_map50_95:.3f}, Subset mAP50={subset_map50:.3f}, Subset mAP50-95={subset_map50_95:.3f}")
            
        except Exception as e:
            logger.warning(f"Failed to run dual evaluation callback: {e}")
            
    return on_fit_epoch_end
