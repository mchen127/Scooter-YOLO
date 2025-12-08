#!/usr/bin/env python3
"""
Remap Taiwan-Traffic dataset labels to COCO class indices.

This script modifies label files in-place to use COCO class indices
instead of the original 0-4 indexing.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.base_config import config


# Taiwan-Traffic to COCO class mapping
CLASS_MAPPING = {
    0: 1,  # bike -> bicycle
    1: 2,  # car -> car
    2: 3,  # motor -> motorcycle
    3: 0,  # person -> person
    4: 7,  # truck -> truck
}


def remap_label_file(label_path: Path) -> int:
    """
    Remap class indices in a single label file.

    Args:
        label_path: Path to the label file

    Returns:
        Number of lines remapped
    """
    lines = []
    count = 0

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:  # class x y w h
                old_class = int(parts[0])
                if old_class in CLASS_MAPPING:
                    parts[0] = str(CLASS_MAPPING[old_class])
                    count += 1
                lines.append(" ".join(parts))

    # Write back
    with open(label_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return count


def remap_split(split_name: str) -> tuple:
    """
    Remap all labels in a dataset split.

    Args:
        split_name: Name of the split (train, valid, test)

    Returns:
        Tuple of (files_processed, lines_remapped)
    """
    labels_dir = Path(config.taiwan_path) / split_name / "labels"

    if not labels_dir.exists():
        print(f"⚠ Warning: {labels_dir} not found, skipping")
        return 0, 0

    label_files = list(labels_dir.glob("*.txt"))
    total_lines = 0

    for label_file in label_files:
        lines_remapped = remap_label_file(label_file)
        total_lines += lines_remapped

    return len(label_files), total_lines


def main():
    """Main remapping function."""
    print("\n" + "=" * 60)
    print("REMAPPING TAIWAN-TRAFFIC LABELS TO COCO INDICES")
    print("=" * 60)
    print("\nClass Mapping:")
    print("  0 (bike)   → 1 (bicycle)")
    print("  1 (car)    → 2 (car)")
    print("  2 (motor)  → 3 (motorcycle)")
    print("  3 (person) → 0 (person)")
    print("  4 (truck)  → 7 (truck)")
    print("\n" + "=" * 60)

    # Remap each split
    splits = ["train", "valid", "test"]
    total_files = 0
    total_lines = 0

    for split in splits:
        print(f"\nProcessing {split} split...")
        files, lines = remap_split(split)
        total_files += files
        total_lines += lines
        print(f"  ✓ {files} files, {lines} labels remapped")

    print("\n" + "=" * 60)
    print(f"✓ COMPLETE: {total_files} files, {total_lines} labels remapped")
    print("=" * 60)
    print("\nNext step: Update data.yaml with COCO class names")
    print("Run: python scripts/setup/2_update_data_yaml.py")
    print()


if __name__ == "__main__":
    main()
