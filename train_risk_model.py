# =====================================================
# File: 2_train_model.py
# Purpose: Train Random Forest model using RAW VALUES ONLY
# NO categorical flags - pure ML pattern learning
# =====================================================

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =====================================================
# STEP 1: LOAD DATASET
# =====================================================

print("=" * 80)
print("TRAINING ML MODEL - RAW VALUES ONLY (NO FLAGS)")
print("=" * 80)

DATASET_PATH = "./datasets/water_quality_dataset.csv"

if not os.path.exists(DATASET_PATH):
    print("❌ Dataset not found!")
    print("Run: python 1_generate_dataset.py first")
    exit()

df = pd.read_csv(DATASET_PATH)
print(f"\n✓ Dataset loaded: {len(df)} samples")
print(f"\nRisk Distribution:")
print(df['Risk'].value_counts())

# =====================================================
# STEP 2: DEFINE FEATURES (RAW + ENGINEERED ONLY)
# =====================================================

# ✅ USE THESE: Raw continuous values + engineered features
feature_columns = [
    # Raw parameters (continuous values)
    "pH",
    "TDS",
    "Turbidity",
    "MP_Count",
    
    # Engineered features (continuous, NOT categorical)
    "pH_deviation",              # How far from neutral
    "TDS_normalized",            # Scaled TDS
    "pollution_index",           # Composite pollution score
    "Turbidity_MP_interaction",  # Interaction term
    "pH_boundary_risk",          # Distance from safety boundaries
    "TDS_boundary_risk"          # Distance from TDS boundaries
] 

print(f"\n✓ Using {len(feature_columns)} features (RAW + ENGINEERED):")
for i, feat in enumerate(feature_columns, 1):
    print(f"   {i}. {feat}")

# =====================================================
# STEP 3: PREPARE DATA
# =====================================================

X = df[feature_columns]
y = df["Risk"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\n✓ Target Label Mapping:")
for label, code in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(f"   {label} → {code}")

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    stratify=y_encoded, 
    random_state=42
)

print(f"\n✓ Data Split:")
print(f"   Training: {len(X_train)} samples")
print(f"   Testing: {len(X_test)} samples")

# =====================================================
# STEP 4: TRAIN RANDOM FOREST MODEL
# =====================================================

print("\n" + "=" * 80)
print("TRAINING RANDOM FOREST CLASSIFIER")
print("=" * 80)

# Model configuration
model = RandomForestClassifier(
    n_estimators=200,        # Number of trees
    max_depth=10,            # Max tree depth (prevents overfitting)
    min_samples_split=10,    # Min samples to split node
    min_samples_leaf=5,      # Min samples in leaf
    max_features='sqrt',     # Features to consider at split
    class_weight='balanced', # Handle class imbalance
    random_state=42,
    n_jobs=-1,               # Use all CPU cores
    verbose=1
)

print("\nModel Configuration:")
print(f"  n_estimators: {model.n_estimators}")
print(f"  max_depth: {model.max_depth}")
print(f"  min_samples_split: {model.min_samples_split}")
print(f"  min_samples_leaf: {model.min_samples_leaf}")

# Train model
print("\n⏳ Training model...")
model.fit(X_train, y_train)
print("✓ Training complete!")

# =====================================================
# STEP 5: EVALUATE MODEL
# =====================================================

print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\n📊 Accuracy Scores:")
print(f"   Training Accuracy: {train_accuracy*100:.2f}%")
print(f"   Testing Accuracy:  {test_accuracy*100:.2f}%")

# Cross-validation (5-fold)
print(f"\n🔄 5-Fold Cross-Validation:")
cv_scores = cross_val_score(model, X, y_encoded, cv=5, scoring='accuracy')
print(f"   CV Scores: {[f'{s*100:.2f}%' for s in cv_scores]}")
print(f"   Mean CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# Classification report
print(f"\n📋 Classification Report (Test Set):")
print(classification_report(
    y_test, y_pred_test, 
    target_names=label_encoder.classes_,
    digits=3
))

# Confusion matrix
print(f"\n🔢 Confusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_pred_test)
cm_df = pd.DataFrame(
    cm,
    index=[f'True {c}' for c in label_encoder.classes_],
    columns=[f'Pred {c}' for c in label_encoder.classes_]
)
print(cm_df)

# Feature importance
print(f"\n🌟 Feature Importance (Top 10):")
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['Feature']:30} {row['Importance']:.4f}")

# =====================================================
# STEP 6: TEST PREDICTIONS
# =====================================================

print("\n" + "=" * 80)
print("SAMPLE PREDICTIONS")
print("=" * 80)

# Test cases
test_cases = [
    {
        "name": "Safe Water (All Good)",
        "pH": 7.5, "TDS": 250, "Turbidity": 0.5, "MP_Count": 0
    },
    {
        "name": "Moderate Risk (Boundary Values)",
        "pH": 6.5, "TDS": 350, "Turbidity": 4.8, "MP_Count": 0
    },
    {
        "name": "High Risk (Poor Parameters)",
        "pH": 6.0, "TDS": 400, "Turbidity": 6.0, "MP_Count": 10
    }
]

for tc in test_cases:
    print(f"\n{tc['name']}:")
    print(f"  Input: pH={tc['pH']}, TDS={tc['TDS']}, Turbidity={tc['Turbidity']}, MP_Count={tc['MP_Count']}")
    
    # Calculate engineered features
    pH_dev = abs(tc['pH'] - 7.5)
    TDS_norm = tc['TDS'] / 1000.0
    poll_idx = (tc['MP_Count']/10) + ((tc['TDS']-200)/800) + (tc['Turbidity']/20)
    turb_mp = tc['Turbidity'] * (tc['MP_Count'] + 1)
    pH_bound = min(abs(tc['pH'] - 6.5), abs(tc['pH'] - 8.5)) / 2.0
    TDS_bound = min(abs(tc['TDS'] - 150), abs(tc['TDS'] - 350)) / 200.0
    
    # Create feature array
    features = np.array([[
        tc['pH'], tc['TDS'], tc['Turbidity'], tc['MP_Count'],
        pH_dev, TDS_norm, poll_idx, turb_mp, pH_bound, TDS_bound
    ]])
    
    # Predict
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    risk = label_encoder.inverse_transform([pred])[0]
    confidence = max(proba) * 100
    
    print(f"  Prediction: {risk} ({confidence:.1f}% confidence)")
    print(f"  Probabilities: ", end="")
    for i, prob in enumerate(proba):
        print(f"{label_encoder.classes_[i]}={prob*100:.1f}% ", end="")
    print()

# =====================================================
# STEP 7: SAVE MODEL
# =====================================================

print("\n" + "=" * 80)
print("SAVING MODEL")
print("=" * 80)

os.makedirs("./models", exist_ok=True)

# Save model
joblib.dump(model, "./models/ml_model.pkl")
print("✓ Model saved: ./models/ml_model.pkl")

# Save label encoder
joblib.dump(label_encoder, "./models/ml_label_encoder.pkl")
print("✓ Label encoder saved: ./models/ml_label_encoder.pkl")

# Save feature columns
joblib.dump(feature_columns, "./models/ml_feature_columns.pkl")
print("✓ Feature columns saved: ./models/ml_feature_columns.pkl")

# Save model metadata
metadata = {
    'train_accuracy': float(train_accuracy),
    'test_accuracy': float(test_accuracy),
    'cv_mean_accuracy': float(cv_scores.mean()),
    'cv_std_accuracy': float(cv_scores.std()),
    'n_features': len(feature_columns),
    'feature_names': feature_columns,
    'n_estimators': model.n_estimators,
    'max_depth': model.max_depth,
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

import json
with open('./models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✓ Metadata saved: ./models/model_metadata.json")

print("\n" + "=" * 80)
print("✅ MODEL TRAINING COMPLETE")
print("=" * 80)
print(f"\nExpected Accuracy Range: 85-92%")
print(f"Your Test Accuracy: {test_accuracy*100:.2f}%")

if test_accuracy >= 0.85 and test_accuracy <= 0.95:
    print("✅ Accuracy is in expected range - Good ML performance!")
elif test_accuracy > 0.95:
    print("⚠️  Accuracy is very high (>95%) - Model might be overfitting or memorizing rules")
else:
    print("⚠️  Accuracy is below 85% - Consider tuning hyperparameters")

print("\nNext step: Update app.py to use these features")
print("=" * 80)