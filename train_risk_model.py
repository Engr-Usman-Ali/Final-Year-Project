# =====================================================
# File: 2_train_model.py
# Purpose: Train Random Forest + Save Evaluation Results
# =====================================================

import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

import seaborn as sns
import matplotlib.pyplot as plt
import joblib


# =====================================================
# 1. PATHS
# =====================================================

MODEL_DIR = r"C:\Users\husnain malik\OneDrive\Desktop\FYP\models"

RESULT_DIR = r"C:\Users\husnain malik\OneDrive\Desktop\FYP\EVALUATION\result_randomforest"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# =====================================================
# 2. LOAD DATASET
# =====================================================

df = pd.read_csv("datasets/water_quality_dataset.csv")

print("\nDataset Loaded")
print("Dataset Shape:", df.shape)


# =====================================================
# 3. BASIC CLEANING
# =====================================================

df = df.dropna()
df = df.drop_duplicates()

print("\nAfter Cleaning:", df.shape)


# =====================================================
# 4. FEATURE SELECTION
# =====================================================

features = ["pH", "TDS", "Turbidity", "MP_Count"]

X = df[features]
y = df["Risk"]


# =====================================================
# 5. TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================================
# 6. MODEL TRAINING
# =====================================================

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=1,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed")


# =====================================================
# 7. PREDICTIONS
# =====================================================

y_pred = model.predict(X_test)


# =====================================================
# 8. PERFORMANCE METRICS
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")


# =====================================================
# 9. CLASSIFICATION REPORT
# =====================================================

labels = ["Low", "Medium", "High"]

report = classification_report(
    y_test,
    y_pred,
    labels=labels
)

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(report)


# =====================================================
# 10. SAVE PERFORMANCE RESULTS
# =====================================================

performance_file = os.path.join(
    RESULT_DIR,
    "performance_metrics.txt"
)

with open(performance_file, "w") as f:

    f.write("RANDOM FOREST MODEL PERFORMANCE\n")
    f.write("================================\n\n")

    f.write(f"Accuracy  : {accuracy * 100:.2f}%\n")
    f.write(f"Precision : {precision * 100:.2f}%\n")
    f.write(f"Recall    : {recall * 100:.2f}%\n")
    f.write(f"F1 Score  : {f1 * 100:.2f}%\n\n")

    f.write("CLASSIFICATION REPORT\n")
    f.write("=====================\n\n")

    f.write(report)


print("\nPerformance metrics saved.")


# =====================================================
# 11. CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(cm)


# =====================================================
# 12. SAVE CONFUSION MATRIX CSV
# =====================================================

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

cm_file = os.path.join(
    RESULT_DIR,
    "confusion_matrix.csv"
)

cm_df.to_csv(cm_file)

print("\nConfusion matrix CSV saved.")


# =====================================================
# 13. SAVE CONFUSION MATRIX IMAGE
# =====================================================

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Risk")
plt.ylabel("Actual Risk")

plt.tight_layout()

confusion_image = os.path.join(
    RESULT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_image,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion matrix image saved.")


# =====================================================
# 14. SAVE MODEL IN EXISTING MODELS FOLDER
# =====================================================

model_file = os.path.join(
    MODEL_DIR,
    "risk_model.pkl"
)

joblib.dump(model, model_file)

print("\nModel saved successfully at:")
print(model_file)


# =====================================================
# 15. FINAL OUTPUT
# =====================================================

print("\n======================================")
print("RANDOM FOREST EVALUATION COMPLETED")
print("======================================")

print("\nModel:")
print(model_file)

print("\nEvaluation Results:")
print(RESULT_DIR) 