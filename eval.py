# =====================================================
# Save YOLO Evaluation Results (FYP Evidence)
# =====================================================

import os
import csv
from ultralytics import YOLO

# =====================================================
# PATH TO YOUR TRAINED MODEL
# =====================================================
MODEL_PATH = r"C:\Users\husnain malik\OneDrive\Desktop\FYP\runs\detect\models\microplastic_yolo\weights\best.pt"

# =====================================================
# DATASET CONFIG
# =====================================================
DATASET_PATH = r"C:\Users\husnain malik\OneDrive\Desktop\FYP\dataset_yolo\data.yaml"

# =====================================================
# OUTPUT FILE
# =====================================================
OUTPUT_CSV = "evaluation_results.csv"


def save_results():
    """
    Runs validation and saves metrics for university submission
    """

    # Load trained model
    model = YOLO(MODEL_PATH)

    print("\n📊 Running Evaluation...\n")

    # Run validation
    metrics = model.val(data=DATASET_PATH)

    # Extract results
    results_dict = metrics.results_dict

    # Print results (for terminal)
    print("\n✅ FINAL RESULTS")
    for k, v in results_dict.items():
        print(f"{k}: {v}")

    # =====================================================
    # SAVE TO CSV (FOR UNIVERSITY SUBMISSION)
    # =====================================================
    file_exists = os.path.isfile(OUTPUT_CSV)

    with open(OUTPUT_CSV, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write header only once
        if not file_exists:
            writer.writerow([
                "Precision",
                "Recall",
                "mAP50",
                "mAP50-95"
            ])

        writer.writerow([
            results_dict.get("metrics/precision(B)"),
            results_dict.get("metrics/recall(B)"),
            results_dict.get("metrics/mAP50(B)"),
            results_dict.get("metrics/mAP50-95(B)")
        ])

    print(f"\n💾 Results saved to: {OUTPUT_CSV}")

    # =====================================================
    # SAVE PREDICTED IMAGES (BOUNDING BOX OUTPUT)
    # =====================================================
    print("\n🖼️ Saving prediction images...")

    model.predict(
        source=r"C:\Users\husnain malik\OneDrive\Desktop\FYP\dataset_yolo\images\test",
        save=True,
        conf=0.5
    )

    print("\n✅ Prediction images saved in 'runs/detect/predict' folder")


# =====================================================
# RUN SCRIPT
# =====================================================
if __name__ == "__main__":
    save_results()