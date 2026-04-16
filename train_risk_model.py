# =====================================================
# File: 2_train_model.py (IMPROVED VERSION)
# =====================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# =====================================================
# 1. LOAD DATASET
# =====================================================

df = pd.read_csv("datasets/water_quality_dataset.csv")

print("\nDataset Loaded!")
print(df.head())


# =====================================================
# 2. DATA CLEANING
# =====================================================

df = df.dropna()
df = df.drop_duplicates()

print("\nAfter cleaning:", df.shape)


# =====================================================
# 3. REMOVE NOISY DATA (VERY IMPORTANT FIX 🔥)
# Remove samples near boundaries where ML gets confused
# =====================================================

def remove_boundary_noise(df):
    return df[
        (df["pH"].between(6.6, 6.9) == False) &
        (df["pH"].between(7.0, 7.2) == False) &
        (df["TDS"].between(190, 210) == False) &
        (df["TDS"].between(260, 280) == False) &
        (df["Turbidity"].between(0.9, 1.1) == False) &
        (df["Turbidity"].between(4.8, 5.2) == False)
    ]

df = remove_boundary_noise(df)

print("\nAfter removing boundary noise:", df.shape)


# =====================================================
# 4. FEATURE SELECTION (ONLY RAW VALUES ✅)
# =====================================================

features = ["pH", "TDS", "Turbidity", "MP_Count"]

X = df[features]
y = df["Risk"]


# =====================================================
# 5. TRAIN / TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================================
# 6. MODEL TRAINING (TUNED 🔥)
# =====================================================

model = RandomForestClassifier(
    n_estimators=500,     # more trees
    max_depth=12,         # slightly deeper
    min_samples_split=4,
    min_samples_leaf=1,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed!")


# =====================================================
# 7. PREDICTIONS
# =====================================================

y_pred = model.predict(X_test)


# =====================================================
# 8. PERFORMANCE METRICS
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n============================")
print("MODEL PERFORMANCE")
print("============================")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# =====================================================
# 8. CONFUSION MATRIX (FIXED LABELS ✅)
# =====================================================

labels = ["Low", "Medium", "High"]

cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# =====================================================
# 10. SAVE MODEL
# =====================================================

# Shuffle dataset (important for better training)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/risk_model.pkl")

print("\nModel saved successfully!")