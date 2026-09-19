from pathlib import Path
from ultralytics import YOLO
import yaml


# ============================================================
# MICROCLEAR - YOLOv8s TRAINING + EVALUATION
# ============================================================

# ============================================================
# DATASET PATH
# ============================================================
# GOOGLE COLAB + GOOGLE DRIVE:
DATASET = Path(
    "/content/drive/MyDrive/microplastic_detection.v1i.yolov8"
)

# LOCAL WINDOWS / VS CODE:
# DATASET = Path(r"D:\MicroClear\microplastic_detection.v1i.yolov8")


# ============================================================
# OUTPUT PATH
# ============================================================
PROJECT = Path("MicroClear_training")
RUN_NAME = "roboflow_microplastic_yolov8s_640"


# ============================================================
# DATASET YAML
# ============================================================
DATA_YAML = DATASET / "data.yaml"

data = {
    "path": str(DATASET),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 1,
    "names": ["Microplastic"]
}

with open(DATA_YAML, "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)


# ============================================================
# TRAINING
# ============================================================
model = YOLO("yolov8s.pt")

model.train(
    data=str(DATA_YAML),

    # Training
    epochs=100,
    imgsz=640,
    batch=8,

    # Optimizer
    optimizer="AdamW",
    lr0=0.0003,
    weight_decay=0.0005,

    # Early stopping
    patience=30,

    # Augmentation
    mosaic=0.5,
    close_mosaic=15,
    fliplr=0.5,
    flipud=0.0,
    degrees=3,
    translate=0.03,
    scale=0.15,
    shear=0.5,
    hsv_h=0.015,
    hsv_s=0.3,
    hsv_v=0.2,

    # Reproducibility
    seed=42,
    deterministic=True,

    # Hardware
    device=0,
    amp=True,
    workers=2,

    # Output
    project=str(PROJECT),
    name=RUN_NAME,
    save=True,
    plots=True
)


# ============================================================
# LOAD BEST MODEL
# ============================================================
BEST_MODEL = (
    PROJECT / RUN_NAME / "weights" / "best.pt"
)

model = YOLO(str(BEST_MODEL))


# ============================================================
# VALIDATION
# ============================================================
val = model.val(
    data=str(DATA_YAML),
    split="val",
    imgsz=640,
    batch=8,
    device=0,
    plots=True,
    project=str(PROJECT),
    name="validation",
    exist_ok=True
)

print("\nVALIDATION RESULTS")
print("------------------")
print(f"Precision : {val.box.mp:.4f}")
print(f"Recall    : {val.box.mr:.4f}")
print(f"mAP50     : {val.box.map50:.4f}")
print(f"mAP50-95  : {val.box.map:.4f}")


# ============================================================
# TEST
# ============================================================
test = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=640,
    batch=8,
    device=0,
    plots=True,
    project=str(PROJECT),
    name="test",
    exist_ok=True
)

precision = test.box.mp
recall = test.box.mr
map50 = test.box.map50
map5095 = test.box.map

f1 = 2 * precision * recall / (precision + recall)


print("\nFINAL TEST RESULTS")
print("------------------")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"mAP50     : {map50:.4f}")
print(f"mAP50-95  : {map5095:.4f}") 