#!/usr/bin/env python3
"""
Update Taiwan-Traffic data.yaml to use COCO class names.
"""
import sys
from pathlib import Path
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.base_config import config


# Full COCO class names (80 classes)
COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def main():
    """Update data.yaml with COCO class names."""
    data_yaml_path = Path(config.taiwan_path) / "data.yaml"

    print("\n" + "=" * 60)
    print("UPDATING DATA.YAML WITH COCO CLASS NAMES")
    print("=" * 60)

    # Read existing yaml
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    print(f"\nOriginal configuration:")
    print(f"  nc: {data.get('nc')}")
    print(f"  names: {data.get('names')}")

    # Update with COCO classes
    data["nc"] = 80
    data["names"] = COCO_CLASSES

    # Write back
    with open(data_yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"\nUpdated configuration:")
    print(f"  nc: {data['nc']}")
    print(f"  names: {data['names'][:8]}... (showing first 8)")

    print("\n" + "=" * 60)
    print(f"✓ Updated: {data_yaml_path}")
    print("=" * 60)
    print("\nYou can now run evaluation:")
    print("  python scripts/evaluate.py")
    print()


if __name__ == "__main__":
    main()
