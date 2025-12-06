from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

def main():
    # 1. Download YOLO11m model
    print("Downloading/Loading YOLO11m model...")
    # This will automatically download the model weights from the latest release if not found locally
    model = YOLO("yolo11m.pt")
    print(f"Model loaded: {model.ckpt_path}")

    # 2. Download COCO 2017 Dataset
    print("Checking/Downloading COCO 2017 dataset...")
    # This function checks if the dataset exists based on coco.yaml
    # If not, it executes the download script embedded in coco.yaml (or the one ultralytics handles)
    dataset_info = check_det_dataset('coco.yaml')
    print("COCO 2017 dataset check complete.")
    print(f"Dataset info: {dataset_info}")

if __name__ == "__main__":
    main()
