# =====================================================
# File: train_risk_model_updated.py
# Purpose: Train ML model for Water Pollution Risk with
#          improved confidence using calibration & optimized RF
# =====================================================

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# =====================================================
# STEP 1: LOAD DATASET
# =====================================================

DATASET_PATH = "./datasets/drinking_water_dataset.csv"
df = pd.read_csv(DATASET_PATH)

print("Dataset Loaded Successfully")
print(df.head())

# =====================================================
# STEP 2: BASIC DATA CLEANING
# =====================================================

df = df.drop_duplicates()

if df.isnull().sum().any():
    print("Missing values found. Dropping rows...")
    df = df.dropna()
else:
    print("No missing values found.")

# Ensure correct data types
df["pH"] = df["pH"].astype(float)
df["TDS"] = df["TDS"].astype(int)
df["Turbidity"] = df["Turbidity"].astype(float)
df["MP_Count"] = df["MP_Count"].astype(int)

print("Data cleaning completed")

# =====================================================
# STEP 3: FEATURE & TARGET SEPARATION
# =====================================================

X = df[["pH", "TDS", "Turbidity", "MP_Count"]]
y = df["Risk"]

# =====================================================
# STEP 4: ENCODE TARGET LABEL
# =====================================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Risk label encoding:")
for cls, val in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(f"{cls} -> {val}")

# =====================================================
# STEP 5: TRAIN-TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Train-test split completed")

# =====================================================
# STEP 6: TRAIN RANDOM FOREST MODEL (OPTIMIZED)
# =====================================================

rf_model = RandomForestClassifier(
    n_estimators=1000,          # more trees for better averaging
    max_depth=None,            # let trees fully grow
    min_samples_leaf=1,        # avoid overfitting on small samples
    random_state=42,
    class_weight="balanced",   # handle class imbalance
)

# Fit Random Forest
rf_model.fit(X_train, y_train)

# =====================================================
# STEP 7: CALIBRATE PROBABILITIES (HIGHLY RECOMMENDED)
# =====================================================

# CalibratedClassifierCV improves probability confidence
calibrated_model = CalibratedClassifierCV(
    estimator=rf_model,
    cv=5,          # 5-fold cross-validation for calibration
    method='sigmoid'  # isotonic calibration often works better for small datasets
)

calibrated_model.fit(X_train, y_train)

print("Random Forest trained and probabilities calibrated successfully")

# =====================================================
# STEP 8: MODEL EVALUATION
# =====================================================

y_pred = calibrated_model.predict(X_test)
y_proba = calibrated_model.predict_proba(X_test)  # predicted probabilities

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# =====================================================
# STEP 9: SAVE MODEL & ENCODER
# =====================================================

os.makedirs("./models", exist_ok=True)
MODEL_PATH = "./models/risk_model_calibrated.pkl"
ENCODER_PATH = "./models/label_encoder.pkl"

joblib.dump(calibrated_model, MODEL_PATH)
joblib.dump(label_encoder, ENCODER_PATH)

print("-" * 30)
print("Calibrated Model and Label Encoder saved successfully!")