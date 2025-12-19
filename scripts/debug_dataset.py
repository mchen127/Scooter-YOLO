from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics import YOLO

try:
    trainer = DetectionTrainer(overrides={"model": "yolo11m.pt", "data": "coco8.yaml"})
    print("Trainer initialized")

    import inspect

    print(f"Signature: {inspect.signature(trainer.get_dataset)}")

    try:
        print("Attempting get_dataset('coco8.yaml')...")
        trainer.get_dataset("coco8.yaml")
        print("Success positional")
    except Exception as e:
        print(f"Failed positional: {e}")

    try:
        print("Attempting get_dataset()...")
        trainer.get_dataset()
        print("Success empty")
    except Exception as e:
        print(f"Failed empty: {e}")

except Exception as e:
    print(f"Global error: {e}")
