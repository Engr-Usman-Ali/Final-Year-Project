# =====================================================
# File: 2_train_model.py (FINAL IMPROVED VERSION)
# Purpose: Train ML model + SAVE all performance metrics
# =====================================================

import pandas as pd
import numpy as np
import os
import json

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
# 1. LOAD DATASET
# =====================================================

df = pd.read_csv("datasets/water_quality_dataset.csv")

print("\n Dataset Loaded")
print(df.shape)


# =====================================================
# 2. BASIC CLEANING (SAFE ONLY)
# =====================================================

df = df.dropna()
df = df.drop_duplicates()

print("\n After cleaning:", df.shape)


# =====================================================
# 3. FEATURE SELECTION
# =====================================================

features = ["pH", "TDS", "Turbidity", "MP_Count"]

X = df[features]
y = df["Risk"]


# =====================================================
# 4. TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================================
# 5. MODEL TRAINING
# =====================================================

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=1,
    random_state=42
)

model.fit(X_train, y_train)

print("\n Model Training Completed")


# =====================================================
# 6. PREDICTIONS
# =====================================================

y_pred = model.predict(X_test)


# =====================================================
# 7. PERFORMANCE METRICS (SAVE EVERYTHING 🔥)
# =====================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\n==============================")
print(" MODEL PERFORMANCE")
print("==============================")

print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")


# =====================================================
# 8. CLASSIFICATION REPORT (DETAILED)
# =====================================================

report = classification_report(y_test, y_pred, output_dict=True)

print("\n Classification Report Generated")


# =====================================================
# 9. SAVE METRICS TO FILE 
# =====================================================

os.makedirs("models", exist_ok=True)

metrics_data = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "classification_report": report
}

with open("models/performance_metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=4)

print("\n Metrics saved to models/performance_metrics.json")


# =====================================================
# 10. CONFUSION MATRIX
# =====================================================

labels = ["Low", "Medium", "High"]

cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# =====================================================
# 11. SAVE MODEL
# =====================================================

joblib.dump(model, "models/risk_model.pkl")

print("\n Model saved successfully!")