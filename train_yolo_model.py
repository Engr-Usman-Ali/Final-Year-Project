# ============================================================================
# MICROCLEAR — NEW ROBoflow DATASET TRAINING
# YOLOv8s | 640x640 | Tesla T4
# Dataset:
# microplastic_detection.v1i.yolov8.zip
# This is for collab AI Training code so don't run on local system
# ============================================================================

import os
import shutil
import zipfile
import yaml
import torch

from google.colab import drive
from ultralytics import YOLO

print("=" * 80)
print("MICROCLEAR — NEW DATASET TRAINING")
print("=" * 80)

# ============================================================================
# 1. MOUNT GOOGLE DRIVE
# ============================================================================

drive.mount("/content/drive")

# ============================================================================
# 2. EXACT ZIP PATH
# ============================================================================

ZIP_PATH = "/content/drive/MyDrive/microplastic_detection.v1i.yolov8.zip"

if not os.path.exists(ZIP_PATH):
    raise FileNotFoundError(
        f"""
ZIP FILE NOT FOUND:

{ZIP_PATH}

Make sure the ZIP file is directly inside:
MyDrive/

Expected filename:
microplastic_detection.v1i.yolov8.zip
"""
    )

print("\nZIP FOUND:")
print(ZIP_PATH)

# ============================================================================
# 3. EXTRACT DATASET
# ============================================================================

EXTRACT_DIR = "/content/microplastic_new"

if os.path.exists(EXTRACT_DIR):
    print("\nRemoving previous extracted dataset...")
    shutil.rmtree(EXTRACT_DIR)

os.makedirs(EXTRACT_DIR, exist_ok=True)

print("\nExtracting dataset...")

with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

print("Extraction complete.")

# ============================================================================
# 4. FIND DATASET ROOT
# ============================================================================

def find_dataset_root(base_dir):

    # Case 1:
    # /content/microplastic_new/train
    # /content/microplastic_new/valid
    # /content/microplastic_new/test

    required = ["train", "valid", "test"]

    if all(
        os.path.isdir(os.path.join(base_dir, folder))
        for folder in required
    ):
        return base_dir

    # Case 2:
    # ZIP contains another folder:
    # /content/microplastic_new/microplastic_detection.v1i.yolov8/train

    for item in os.listdir(base_dir):

        candidate = os.path.join(base_dir, item)

        if os.path.isdir(candidate):

            if all(
                os.path.isdir(os.path.join(candidate, folder))
                for folder in required
            ):
                return candidate

    return None


DATASET_ROOT = find_dataset_root(EXTRACT_DIR)

if DATASET_ROOT is None:
    print("\nExtracted folders:")
    for root, dirs, files in os.walk(EXTRACT_DIR):
        level = root.replace(EXTRACT_DIR, "").count(os.sep)
        indent = "  " * level
        print(indent + os.path.basename(root) + "/")

    raise RuntimeError(
        "\nCould not find train/valid/test folders."
    )

print("\nDATASET ROOT:")
print(DATASET_ROOT)

# ============================================================================
# 5. CHECK DATASET STRUCTURE
# ============================================================================

print("\n" + "=" * 80)
print("DATASET STRUCTURE CHECK")
print("=" * 80)

splits = ["train", "valid", "test"]

for split in splits:

    images_dir = os.path.join(DATASET_ROOT, split, "images")
    labels_dir = os.path.join(DATASET_ROOT, split, "labels")

    print(f"\n{split.upper()}")

    print("Images:", images_dir)
    print("Labels:", labels_dir)

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"Missing images folder:\n{images_dir}"
        )

    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(
            f"Missing labels folder:\n{labels_dir}"
        )

    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
    ]

    label_files = [
        f for f in os.listdir(labels_dir)
        if f.lower().endswith(".txt")
    ]

    print("Images:", len(image_files))
    print("Labels:", len(label_files))

# ============================================================================
# 6. FIND data.yaml
# ============================================================================

DATA_YAML = os.path.join(DATASET_ROOT, "data.yaml")

if not os.path.exists(DATA_YAML):

    # Search recursively
    found_yaml = None

    for root, dirs, files in os.walk(DATASET_ROOT):

        if "data.yaml" in files:
            found_yaml = os.path.join(root, "data.yaml")
            break

    if found_yaml is None:
        raise FileNotFoundError(
            "data.yaml was not found inside the dataset."
        )

    DATA_YAML = found_yaml

print("\nDATA.YAML:")
print(DATA_YAML)

# ============================================================================
# 7. READ data.yaml
# ============================================================================

with open(DATA_YAML, "r") as f:
    data = yaml.safe_load(f)

print("\nOriginal data.yaml:")
print(data)

# ============================================================================
# 8. FIX PATHS FOR COLAB
# ============================================================================

# We create our own clean data.yaml.
# This prevents Windows paths or Roboflow paths from causing problems.

CLEAN_YAML = "/content/microplastic_new_data.yaml"

clean_data = {
    "path": DATASET_ROOT,
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 1,
    "names": ["Microplastic"]
}

with open(CLEAN_YAML, "w") as f:
    yaml.safe_dump(clean_data, f, sort_keys=False)

print("\nClean data.yaml created:")
print(CLEAN_YAML)

print("\nClean configuration:")
print(clean_data)

# ============================================================================
# 9. GPU CHECK
# ============================================================================

print("\n" + "=" * 80)
print("GPU CHECK")
print("=" * 80)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():

    print("GPU:", torch.cuda.get_device_name(0))

else:

    raise RuntimeError(
        "CUDA GPU is not available. Change Colab runtime to GPU."
    )

# ============================================================================
# 10. INSTALL / VERIFY ULTRALYTICS
# ============================================================================

print("\nChecking Ultralytics...")

try:
    import ultralytics
    print("Ultralytics:", ultralytics.__version__)

except Exception:

    print("Installing Ultralytics...")

    !pip install -q -U ultralytics

    import ultralytics
    print("Ultralytics:", ultralytics.__version__)

# ============================================================================
# 11. TRAINING CONFIGURATION
# ============================================================================

PROJECT_DIR = "/content/MicroClear_new_training"

RUN_NAME = "roboflow_microplastic_yolov8s_640"

print("\n" + "=" * 80)
print("TRAINING CONFIGURATION")
print("=" * 80)

print("Model       : YOLOv8s")
print("Image size  : 640")
print("Batch size  : 8")
print("Epochs      : 100")
print("Optimizer   : AdamW")
print("Learning rate:", 0.0003)
print("GPU         :", torch.cuda.get_device_name(0))
print("Dataset     :", DATASET_ROOT)

# ============================================================================
# 12. LOAD PRETRAINED YOLOv8s
# ============================================================================

model = YOLO("yolov8s.pt")

# ============================================================================
# 13. TRAIN
# ============================================================================

print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)

results = model.train(

    data=CLEAN_YAML,

    # Model
    model="yolov8s.pt",

    # Training
    epochs=100,
    imgsz=640,
    batch=8,

    # Optimizer
    optimizer="AdamW",
    lr0=0.0003,
    lrf=0.01,
    weight_decay=0.0005,

    # Scheduler
    cos_lr=True,

    # Augmentation
    mosaic=0.5,
    close_mosaic=15,

    fliplr=0.5,
    flipud=0.0,

    degrees=3.0,
    translate=0.03,
    scale=0.15,
    shear=0.5,

    hsv_h=0.015,
    hsv_s=0.3,
    hsv_v=0.2,

    # Stability
    patience=30,
    seed=42,
    deterministic=True,

    # GPU
    device=0,
    amp=True,

    # Workers
    workers=2,

    # Output
    project=PROJECT_DIR,
    name=RUN_NAME,

    # Validation
    val=True,
    plots=True,

    # Save
    save=True,
    exist_ok=True,

    verbose=True
)

# ============================================================================
# 14. BEST MODEL PATH
# ============================================================================

BEST_MODEL = os.path.join(
    PROJECT_DIR,
    RUN_NAME,
    "weights",
    "best.pt"
)

LAST_MODEL = os.path.join(
    PROJECT_DIR,
    RUN_NAME,
    "weights",
    "last.pt"
)

print("\n" + "=" * 80)
print("TRAINING FINISHED")
print("=" * 80)

print("Best model:")
print(BEST_MODEL)

print("\nLast model:")
print(LAST_MODEL)

if not os.path.exists(BEST_MODEL):

    raise FileNotFoundError(
        f"best.pt was not created:\n{BEST_MODEL}"
    )

# ============================================================================
# 15. VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("VALIDATION RESULTS")
print("=" * 80)

best_model = YOLO(BEST_MODEL)

val_results = best_model.val(
    data=CLEAN_YAML,
    split="val",
    imgsz=640,
    batch=8,
    device=0,
    plots=True
)

# ============================================================================
# 16. EXTRACT VALIDATION METRICS
# ============================================================================

try:

    val_precision = float(val_results.box.mp)
    val_recall = float(val_results.box.mr)
    val_map50 = float(val_results.box.map50)
    val_map5095 = float(val_results.box.map)

except Exception:

    val_precision = 0
    val_recall = 0
    val_map50 = 0
    val_map5095 = 0

print("\nVALIDATION METRICS")

print(f"Precision : {val_precision:.4f}")
print(f"Recall    : {val_recall:.4f}")
print(f"mAP50     : {val_map50:.4f}")
print(f"mAP50-95  : {val_map5095:.4f}")

# ============================================================================
# 17. TEST SET
# ============================================================================

print("\n" + "=" * 80)
print("TEST RESULTS")
print("=" * 80)

test_results = best_model.val(
    data=CLEAN_YAML,
    split="test",
    imgsz=640,
    batch=8,
    device=0,
    plots=True
)

# ============================================================================
# 18. EXTRACT TEST METRICS
# ============================================================================

try:

    test_precision = float(test_results.box.mp)
    test_recall = float(test_results.box.mr)
    test_map50 = float(test_results.box.map50)
    test_map5095 = float(test_results.box.map)

except Exception:

    test_precision = 0
    test_recall = 0
    test_map50 = 0
    test_map5095 = 0

print("\nTEST METRICS")

print(f"Precision : {test_precision:.4f}")
print(f"Recall    : {test_recall:.4f}")
print(f"mAP50     : {test_map50:.4f}")
print(f"mAP50-95  : {test_map5095:.4f}")

# ============================================================================
# 19. COPY BEST MODEL TO MICROCLEAR DRIVE
# ============================================================================

FINAL_MODEL_DIR = (
    "/content/drive/MyDrive/MicroClear/models/"
    "microplastic_yolo_roboflow_640"
)

os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

FINAL_MODEL = os.path.join(
    FINAL_MODEL_DIR,
    "best.pt"
)

shutil.copy2(
    BEST_MODEL,
    FINAL_MODEL
)

# ============================================================================
# 20. FINAL SUMMARY
# ============================================================================

print("\n")
print("=" * 80)
print("MICROCLEAR — FINAL MODEL SUMMARY")
print("=" * 80)

print("\nVALIDATION")
print("-" * 40)
print(f"Precision : {val_precision:.4f}")
print(f"Recall    : {val_recall:.4f}")
print(f"mAP50     : {val_map50:.4f}")
print(f"mAP50-95  : {val_map5095:.4f}")

print("\nTEST")
print("-" * 40)
print(f"Precision : {test_precision:.4f}")
print(f"Recall    : {test_recall:.4f}")
print(f"mAP50     : {test_map50:.4f}")
print(f"mAP50-95  : {test_map5095:.4f}")

print("\nMODEL SAVED TO")
print("-" * 40)
print(FINAL_MODEL)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)