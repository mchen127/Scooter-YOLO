import os
import yaml
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.base_config import config

# Constants
TRAFFIC_CLASSES = [0, 1, 2, 3, 7]  # Person, Bicycle, Car, Motorcycle, Truck
COCO_PATH = Path(config.coco_path)
TAIWAN_PATH = Path(config.taiwan_path)
OUTPUT_PATH = Path(config.generated_path)


def setup_directories():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def get_image_files(folder):
    return list(Path(folder).glob("*.jpg")) + list(Path(folder).glob("*.png"))


def filter_coco_subset(source_dir, output_name, ratio=0.1, mode="random"):
    """
    Creates a subset of COCO data.
    mode: 'random' or 'stratified' (stratified focuses on traffic classes)
    """
    print(f"Creating COCO subset: {output_name} (mode={mode}, ratio={ratio})")

    # Define paths
    images_dir = COCO_PATH / "images" / source_dir  # e.g., train2017
    labels_dir = COCO_PATH / "labels" / source_dir

    if not images_dir.exists() or not labels_dir.exists():
        print(f"Error: COCO {source_dir} not found at {images_dir}")
        return None

    all_images = get_image_files(images_dir)
    selected_images = []

    if mode == "random":
        count = int(len(all_images) * ratio)
        selected_images = random.sample(all_images, count)

    elif mode == "stratified":
        # Scan labels to find traffic-relevant images
        traffic_images = []
        other_images = []

        print("Scanning labels for stratified sampling...")
        for img_path in tqdm(all_images):
            label_path = labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            has_traffic = False
            with open(label_path, "r") as f:
                for line in f:
                    try:
                        cls = int(line.split()[0])
                        if cls in TRAFFIC_CLASSES:
                            has_traffic = True
                            break
                    except ValueError:
                        continue

            if has_traffic:
                traffic_images.append(img_path)
            else:
                other_images.append(img_path)

        total_count = int(len(all_images) * ratio)
        traffic_count = int(total_count * 0.7)  # 70% traffic
        other_count = total_count - traffic_count

        # Sample
        if len(traffic_images) > traffic_count:
            selected_images.extend(random.sample(traffic_images, traffic_count))
        else:
            selected_images.extend(traffic_images)

        if len(other_images) > other_count:
            selected_images.extend(random.sample(other_images, other_count))
        else:
            selected_images.extend(other_images)

    # Create output text file with image paths
    output_file = OUTPUT_PATH / f"{output_name}.txt"
    with open(output_file, "w") as f:
        for img_path in selected_images:
            f.write(str(img_path.absolute()) + "\n")

    print(f"Created subset list at {output_file} with {len(selected_images)} images")
    return output_file


def create_mixed_yaml(taiwan_yaml_path, coco_subset_file, output_filename):
    """
    Creates a new data.yaml that combines Taiwan data with the COCO subset.
    """
    print(f"Creating mixed YAML: {output_filename}")

    with open(taiwan_yaml_path, "r") as f:
        taiwan_data = yaml.safe_load(f)

    # Get absolute path for Taiwan train
    # Assuming taiwan_yaml_path is in datasets/taiwan-traffic/data.yaml
    # and train path is relative like ../train/images
    base_dir = Path(taiwan_yaml_path).parent
    taiwan_train = (base_dir / taiwan_data["train"]).resolve()

    # Create new data dict
    mixed_data = taiwan_data.copy()

    # Set train to list of [taiwan_dir, coco_subset_file]
    # Ultralytics supports mixing directories and text files
    mixed_data["train"] = [str(taiwan_train), str(coco_subset_file.absolute())]

    # Save
    output_yaml = OUTPUT_PATH / output_filename
    with open(output_yaml, "w") as f:
        yaml.dump(mixed_data, f)

    print(f"Saved mixed YAML to {output_yaml}")
    return output_yaml


def create_coco_traffic_subset_yaml():
    """
    Creates a YAML for evaluating ONLY on COCO images that contain traffic classes.
    """
    print("Creating COCO Traffic Subset YAML for evaluation...")

    # 1. Find all COCO val images with traffic classes
    val_images_dir = COCO_PATH / "images" / "val2017"
    val_labels_dir = COCO_PATH / "labels" / "val2017"

    if not val_images_dir.exists():
        print("COCO val2017 not found.")
        return

    traffic_val_images = []
    for img_path in tqdm(get_image_files(val_images_dir)):
        label_path = val_labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        with open(label_path, "r") as f:
            for line in f:
                try:
                    cls = int(line.split()[0])
                    if cls in TRAFFIC_CLASSES:
                        traffic_val_images.append(img_path)
                        break
                except ValueError:
                    continue

    # 2. Save list
    subset_list = OUTPUT_PATH / "coco_traffic_val.txt"
    with open(subset_list, "w") as f:
        for img in traffic_val_images:
            f.write(str(img.absolute()) + "\n")

    # 3. Create YAML
    # We can copy the structure from taiwan data.yaml since classes match
    with open(TAIWAN_PATH / "data.yaml", "r") as f:
        base_data = yaml.safe_load(f)

    subset_data = base_data.copy()
    subset_data["val"] = str(subset_list.absolute())
    # We don't strictly need train/test for this eval-only yaml
    subset_data["train"] = str(subset_list.absolute())

    output_yaml = OUTPUT_PATH / "coco_traffic_subset.yaml"
    with open(output_yaml, "w") as f:
        yaml.dump(subset_data, f)

    print(
        f"Created COCO Traffic Subset YAML at {output_yaml} ({len(traffic_val_images)} images)"
    )


if __name__ == "__main__":
    setup_directories()

    # 1. Create COCO Traffic Subset for Evaluation
    create_coco_traffic_subset_yaml()

    # 2. Create DIL-1 Dataset (Random 10%)
    subset_file_random = filter_coco_subset(
        "train2017", "coco_10pct_random", ratio=0.1, mode="random"
    )
    if subset_file_random:
        create_mixed_yaml(
            TAIWAN_PATH / "data.yaml", subset_file_random, "mixed_10pct_random.yaml"
        )

    # 3. Create DIL-2 Dataset (Stratified 10%)
    subset_file_strat = filter_coco_subset(
        "train2017", "coco_10pct_stratified", ratio=0.1, mode="stratified"
    )
    if subset_file_strat:
        create_mixed_yaml(
            TAIWAN_PATH / "data.yaml", subset_file_strat, "mixed_10pct_stratified.yaml"
        )
