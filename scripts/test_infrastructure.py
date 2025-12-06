#!/usr/bin/env python3
"""
Quick test script to verify the infrastructure setup.

This script tests:
1. Model loading
2. DualEvaluator functionality
3. Training configurations

Run: python scripts/test_infrastructure.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import DualEvaluator
from src.trainers import StandardFineTuner
from src.configs.tl1_configs import list_experiments, get_config

def test_imports():
    """Test that all modules import correctly."""
    print("✓ Testing imports...")
    try:
        from src.evaluation import DualEvaluator
        from src.trainers import BaseTrainer, StandardFineTuner
        from src.configs import tl1_configs
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_evaluator():
    """Test DualEvaluator initialization."""
    print("\n✓ Testing DualEvaluator...")
    try:
        evaluator = DualEvaluator(
            coco_yaml="coco.yaml",
            taiwan_yaml="datasets/taiwan-traffic/data.yaml",
            device="0"
        )
        print(f"  ✓ Evaluator initialized")
        print(f"    - COCO baseline: {evaluator.baseline_coco_map:.1%}")
        print(f"    - Taiwan baseline: {evaluator.baseline_taiwan_map:.1%}")
        print(f"    - Traffic classes: {evaluator.TRAFFIC_CLASS_NAMES}")
        return True
    except Exception as e:
        print(f"  ✗ Evaluator test failed: {e}")
        return False


def test_configs():
    """Test experiment configurations."""
    print("\n✓ Testing experiment configurations...")
    try:
        experiments = list_experiments()
        print(f"  ✓ Found {len(experiments)} experiments:")
        for exp_name in experiments:
            config = get_config(exp_name)
            print(f"    - {exp_name}: LR={config.get('lr0')}, Epochs={config.get('epochs')}")
        return True
    except Exception as e:
        print(f"  ✗ Config test failed: {e}")
        return False


def test_model_loading():
    """Test model loading (without training)."""
    print("\n✓ Testing model loading...")
    try:
        model_path = Path("models/yolo11m.pt")
        if not model_path.exists():
            print(f"  ⚠ Model not found: {model_path}")
            print(f"    (This is expected if you haven't downloaded it yet)")
            return True
        
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        print(f"  ✓ Model loaded successfully: {model_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        return False


def test_trainer_init():
    """Test trainer initialization (without training)."""
    print("\n✓ Testing StandardFineTuner initialization...")
    try:
        # Only test if model exists
        model_path = Path("models/yolo11m.pt")
        if not model_path.exists():
            print(f"  ⚠ Skipping (model not found)")
            return True
        
        trainer = StandardFineTuner(
            model_path="models/yolo11m.pt",
            data_yaml="datasets/taiwan-traffic/data.yaml",
            lr0=1e-4,
            epochs=1,  # Don't actually train
            project="runs/test",
            name="test_init"
        )
        print(f"  ✓ Trainer initialized successfully")
        print(f"    - Model: {trainer.model_path.name}")
        print(f"    - Dataset: {trainer.data_yaml}")
        print(f"    - LR: {trainer.lr0}")
        return True
    except Exception as e:
        print(f"  ✗ Trainer init failed: {e}")
        return False


def main():
    print("="*70)
    print("INFRASTRUCTURE TEST SUITE")
    print("="*70)
    
    tests = [
        test_imports,
        test_configs,
        test_evaluator,
        test_model_loading,
        test_trainer_init,
    ]
    
    passed = sum(test() for test in tests)
    total = len(tests)
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n✓ All tests passed! Infrastructure is ready.")
        print("\nNext steps:")
        print("  1. Download model: python scripts/0_download_setup.py")
        print("  2. List experiments: python scripts/train_tl1_baseline.py --list")
        print("  3. Run TL-1 baseline: python scripts/train_tl1_baseline.py --experiment tl1_lr_1e4")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
