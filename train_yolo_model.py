
# ============================================================
# 0. MOUNT GOOGLE DRIVE (Colab only, skipped automatically
#    if the script is run outside Colab)
# ============================================================
try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    pass
 
 
# ============================================================
# 1. INSTALL DEPENDENCIES (Colab only)
# ============================================================
# Uncomment the line below the first time you run this in a fresh
# Colab session. It only needs to be run once per session.
# !pip install ultralytics pandas matplotlib pyyaml -q
 
from pathlib import Path
import shutil
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
 
 
# ============================================================
# 2. PROJECT PATHS (Google Drive, mirrors the FYP folder)
# ============================================================
ROOT = Path("/content/drive/MyDrive/FYP")

DATASET = ROOT / "datasets" / "microplastic_detection.v1i.yolov8"

YOLO_BASE_WEIGHTS = ROOT / "yolov8s.pt"
 
RUN_DIR = ROOT / "models" / "microplastic_yolo_roboflow_640_run2"
RUN_DIR.mkdir(parents=True, exist_ok=True)
 
TRAIN_NAME = "training"
VAL_NAME = "validation"
TEST_NAME = "test"
 
FIGURES_DIR = RUN_DIR / "report_figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
 
 
# ============================================================
# 3. CHECK DATASET
# ============================================================
if not DATASET.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET}\n\n"
        "Please check the DATASET path in section 2."
    )
 
print("=" * 70)
print("MICROCLEAR - YOLOv8s TRAINING + EVALUATION (Google Colab)")
print("=" * 70)
print("\nDataset:", DATASET)
print("Run folder:", RUN_DIR)
 
required_folders = [
    DATASET / "train" / "images",
    DATASET / "train" / "labels",
    DATASET / "valid" / "images",
    DATASET / "valid" / "labels",
    DATASET / "test" / "images",
    DATASET / "test" / "labels",
]
 
for folder in required_folders:
    if not folder.exists():
        raise FileNotFoundError(f"\nRequired folder not found:\n{folder}")
 
print("\nDataset structure: OK")
 
 
# ============================================================
# 4. CREATE DATASET YAML
#    (saved inside the run folder, matching the project tree file
#    "microplastic_new_data.yaml")
# ============================================================
DATA_YAML = RUN_DIR / "microplastic_new_data.yaml"
 
data = {
    "path": str(DATASET),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 1,
    "names": ["Microplastic"],
}
 
with open(DATA_YAML, "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)
 
print("\nDataset YAML:", DATA_YAML)
 
 
# ============================================================
# 5. SAVE DATASET INFORMATION
# ============================================================
dataset_info_file = RUN_DIR / "dataset_information.txt"
 
train_count = len(list((DATASET / "train" / "images").glob("*")))
valid_count = len(list((DATASET / "valid" / "images").glob("*")))
test_count = len(list((DATASET / "test" / "images").glob("*")))
 
with open(dataset_info_file, "w") as f:
    f.write("MICROCLEAR - YOLOv8s DATASET INFORMATION\n")
    f.write("=========================================\n\n")
    f.write(f"Dataset path : {DATASET}\n")
    f.write("Classes      : 1 (Microplastic)\n")
    f.write(f"Train images : {train_count}\n")
    f.write(f"Valid images : {valid_count}\n")
    f.write(f"Test images  : {test_count}\n")
 
print("Dataset information saved:", dataset_info_file)
 
 
# ============================================================
# 6. TRAINING CONFIGURATION
# ============================================================
print("\n" + "=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)
print("Model        : YOLOv8s")
print("Image size   : 640x640")
print("Epochs       : 100")
print("Batch size   : 8")
print("Optimizer    : AdamW")
print("Learning rate: 0.0003")
print("Weight decay : 0.0005")
print("Seed         : 42")
 
 
# ============================================================
# 7. LOAD YOLOv8s PRETRAINED WEIGHTS
# ============================================================
if YOLO_BASE_WEIGHTS.exists():
    model = YOLO(str(YOLO_BASE_WEIGHTS))
else:
    # Falls back to downloading the official pretrained weights
    # if yolov8s.pt is not yet present in the Drive folder.
    model = YOLO("yolov8s.pt")
 
 
# ============================================================
# 8. TRAINING
# ============================================================
print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)
 
model.train(
    data=str(DATA_YAML),
 
    # Training
    epochs=100,
    imgsz=640,
    batch=8,
 
    # Optimizer
    optimizer="AdamW",
    lr0=0.0003,
    lrf=0.01,
    weight_decay=0.0005,
 
    # Early stopping
    patience=30,
 
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
 
    # Learning rate scheduler
    cos_lr=True,
 
    # Reproducibility
    seed=42,
    deterministic=True,
 
    # Hardware (device=0 uses the Colab GPU; set to "cpu" if no GPU runtime)
    device=0,
    amp=True,
    workers=2,
 
    # Output
    project=str(RUN_DIR),
    name=TRAIN_NAME,
    save=True,
    plots=True,
    exist_ok=True,
    verbose=True,
)
 
 
# ============================================================
# 9. COPY BEST / LAST MODEL TO THE RUN ROOT 
# ============================================================
TRAIN_WEIGHTS_DIR = RUN_DIR / TRAIN_NAME / "weights"
BEST_MODEL_SRC = TRAIN_WEIGHTS_DIR / "best.pt"
LAST_MODEL_SRC = TRAIN_WEIGHTS_DIR / "last.pt"
 
BEST_MODEL = RUN_DIR / "best.pt"
LAST_MODEL = RUN_DIR / "last.pt"
 
if not BEST_MODEL_SRC.exists():
    raise FileNotFoundError(f"\nBest model was not created:\n{BEST_MODEL_SRC}")
 
shutil.copy(BEST_MODEL_SRC, BEST_MODEL)
shutil.copy(LAST_MODEL_SRC, LAST_MODEL)
 
print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print("\nBest model:", BEST_MODEL)
print("Last model:", LAST_MODEL)
 
 
# ============================================================
# 10. LOAD BEST MODEL
# ============================================================
model = YOLO(str(BEST_MODEL))
 
 
# ============================================================
# 11. VALIDATION
# ============================================================
print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)
 
val = model.val(
    data=str(DATA_YAML),
    split="val",
    imgsz=640,
    batch=8,
    device=0,
    plots=True,
    project=str(RUN_DIR),
    name=VAL_NAME,
    exist_ok=True,
)
 
val_precision = float(val.box.mp)
val_recall = float(val.box.mr)
val_map50 = float(val.box.map50)
val_map5095 = float(val.box.map)
 
val_f1 = (
    2 * val_precision * val_recall / (val_precision + val_recall)
    if (val_precision + val_recall) > 0
    else 0
)
 
print("\nVALIDATION RESULTS")
print("------------------")
print(f"Precision : {val_precision:.4f}")
print(f"Recall    : {val_recall:.4f}")
print(f"F1 Score  : {val_f1:.4f}")
print(f"mAP50     : {val_map50:.4f}")
print(f"mAP50-95  : {val_map5095:.4f}")
 
 
# ============================================================
# 12. TESTING
# ============================================================
print("\n" + "=" * 70)
print("TESTING")
print("=" * 70)
 
test = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=640,
    batch=8,
    device=0,
    plots=True,
    project=str(RUN_DIR),
    name=TEST_NAME,
    exist_ok=True,
)
 
test_precision = float(test.box.mp)
test_recall = float(test.box.mr)
test_map50 = float(test.box.map50)
test_map5095 = float(test.box.map)
 
test_f1 = (
    2 * test_precision * test_recall / (test_precision + test_recall)
    if (test_precision + test_recall) > 0
    else 0
)
 
print("\nFINAL TEST RESULTS")
print("------------------")
print(f"Precision : {test_precision:.4f}")
print(f"Recall    : {test_recall:.4f}")
print(f"F1 Score  : {test_f1:.4f}")
print(f"mAP50     : {test_map50:.4f}")
print(f"mAP50-95  : {test_map5095:.4f}")
 
 
# ============================================================
# 13. SAVE FINAL METRICS CSV 
# ============================================================
metrics = pd.DataFrame(
    {
        "split": ["validation", "test"],
        "precision": [val_precision, test_precision],
        "recall": [val_recall, test_recall],
        "f1_score": [val_f1, test_f1],
        "mAP50": [val_map50, test_map50],
        "mAP50-95": [val_map5095, test_map5095],
    }
)
 
METRICS_FILE = RUN_DIR / "final_metrics.csv"
metrics.to_csv(METRICS_FILE, index=False)
print("\nMetrics saved to:", METRICS_FILE)
 
 
# ============================================================
# 14. SAVE TRAINING LOSS FIGURE 
# ============================================================
RESULTS_CSV = RUN_DIR / TRAIN_NAME / "results.csv"
 
if RESULTS_CSV.exists():
    df = pd.read_csv(RESULTS_CSV)
    df.columns = df.columns.str.strip()
 
    plt.figure(figsize=(10, 6))
 
    if "train/box_loss" in df.columns:
        plt.plot(df["epoch"], df["train/box_loss"], label="Box Loss")
    if "train/cls_loss" in df.columns:
        plt.plot(df["epoch"], df["train/cls_loss"], label="Classification Loss")
    if "train/dfl_loss" in df.columns:
        plt.plot(df["epoch"], df["train/dfl_loss"], label="DFL Loss")
 
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("YOLOv8s Training Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
 
    FIGURE_FILE = FIGURES_DIR / "YOLOv8s_training_loss_curves.png"
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.show()
 
    print("\nTraining figure saved to:", FIGURE_FILE)
 
 
# ============================================================
# 15. SAVE FINAL SUMMARY TEXT FILE 
# ============================================================
SUMMARY_FILE = RUN_DIR / "final_summary.txt"
 
with open(SUMMARY_FILE, "w") as f:
    f.write("MICROCLEAR - YOLOv8s TRAINING AND EVALUATION SUMMARY\n")
    f.write("======================================================\n\n")
    f.write(f"Dataset      : {DATASET}\n")
    f.write(f"Base weights : {YOLO_BASE_WEIGHTS}\n")
    f.write(f"Run folder   : {RUN_DIR}\n\n")
 
    f.write("TRAINING CONFIGURATION\n")
    f.write("-----------------------\n")
    f.write("Model        : YOLOv8s\n")
    f.write("Image size   : 640x640\n")
    f.write("Epochs       : 100 (patience 30)\n")
    f.write("Batch size   : 8\n")
    f.write("Optimizer    : AdamW (lr0=0.0003, weight_decay=0.0005)\n")
    f.write("Seed         : 42\n\n")
 
    f.write("VALIDATION RESULTS\n")
    f.write("-------------------\n")
    f.write(f"Precision : {val_precision * 100:.2f}%\n")
    f.write(f"Recall    : {val_recall * 100:.2f}%\n")
    f.write(f"F1 Score  : {val_f1 * 100:.2f}%\n")
    f.write(f"mAP50     : {val_map50 * 100:.2f}%\n")
    f.write(f"mAP50-95  : {val_map5095 * 100:.2f}%\n\n")
 
    f.write("TEST RESULTS\n")
    f.write("-------------\n")
    f.write(f"Precision : {test_precision * 100:.2f}%\n")
    f.write(f"Recall    : {test_recall * 100:.2f}%\n")
    f.write(f"F1 Score  : {test_f1 * 100:.2f}%\n")
    f.write(f"mAP50     : {test_map50 * 100:.2f}%\n")
    f.write(f"mAP50-95  : {test_map5095 * 100:.2f}%\n\n")
 
    f.write(f"Best model   : {BEST_MODEL}\n")
    f.write(f"Last model   : {LAST_MODEL}\n")
    f.write(f"Metrics CSV  : {METRICS_FILE}\n")
 
print("\nFinal summary saved to:", SUMMARY_FILE)
 
 
# ============================================================
# 16. FINAL OUTPUT
# ============================================================
print("\n")
print("=" * 70)
print("MICROCLEAR - TRAINING AND EVALUATION COMPLETE")
print("=" * 70)
 
print("\nVALIDATION")
print("------------------")
print(f"Precision : {val_precision:.4f}")
print(f"Recall    : {val_recall:.4f}")
print(f"F1 Score  : {val_f1:.4f}")
print(f"mAP50     : {val_map50:.4f}")
print(f"mAP50-95  : {val_map5095:.4f}")
 
print("\nTEST")
print("------------------")
print(f"Precision : {test_precision:.4f}")
print(f"Recall    : {test_recall:.4f}")
print(f"F1 Score  : {test_f1:.4f}")
print(f"mAP50     : {test_map50:.4f}")
print(f"mAP50-95  : {test_map5095:.4f}")
 
print("\nBEST MODEL")
print("------------------")
print(BEST_MODEL)
 
print("\nFINAL METRICS")
print("------------------")
print(METRICS_FILE)
 
print("\nALL RESULTS SAVED UNDER")
print("------------------")
print(RUN_DIR)
 
print("\n" + "=" * 70)
print("ALL TRAINING AND EVALUATION RESULTS ARE READY")
print("=" * 70)
 