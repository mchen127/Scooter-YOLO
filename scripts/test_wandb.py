#!/usr/bin/env python3
"""
Test W&B integration and baseline evaluation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_wandb_import():
    """Test W&B import."""
    print("✓ Testing W&B import...")
    try:
        import wandb
        print(f"  ✓ wandb version: {wandb.__version__}")
        return True
    except ImportError as e:
        print(f"  ✗ Failed: {e}")
        return False

def test_wandb_logger():
    """Test WandBLogger class."""
    print("\n✓ Testing WandBLogger...")
    try:
        from src.utils.wandb_logger import WandBLogger, setup_wandb
        
        # Test initialization (disabled mode)
        logger = WandBLogger(
            project="test-project",
            name="test-run",
            mode="disabled"
        )
        print("  ✓ WandBLogger initialized (disabled mode)")
        
        # Test logging (should not error in disabled mode)
        logger.log_metrics({"test_metric": 0.5})
        logger.finish()
        print("  ✓ Logging methods work")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def test_baseline_script():
    """Test baseline evaluation script exists."""
    print("\n✓ Testing baseline evaluation script...")
    script_path = Path("scripts/evaluate_baseline.py")
    if script_path.exists():
        print(f"  ✓ Script exists: {script_path}")
        return True
    else:
        print(f"  ✗ Script not found: {script_path}")
        return False

def main():
    print("="*70)
    print("W&B INTEGRATION TEST SUITE")
    print("="*70)
    
    tests = [
        test_wandb_import,
        test_wandb_logger,
        test_baseline_script,
    ]
    
    passed = sum(test() for test in tests)
    total = len(tests)
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n✓ All tests passed! W&B integration is ready.")
        print("\nNext steps:")
        print("  1. Login to W&B: wandb login")
        print("  2. Run baseline: python scripts/evaluate_baseline.py --wandb")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
