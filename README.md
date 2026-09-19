# MicroClear — AI-Based Microplastic Pollution Risk Assessment and Mitigation System

## 1. Project Overview

**MicroClear** is an AI-based water pollution assessment system designed to detect **microplastics in microscope images** and assess the overall **water pollution risk** using water-quality parameters.

The system combines **Computer Vision, Machine Learning, and Rule-Based Decision Making** to provide a final pollution risk level.

The system uses:

* **YOLOv8** for microplastic detection and counting.
* **Random Forest** for water pollution risk classification.
* **Rule-Based Safety Layer** for the final risk decision.

---

## 2. Problem Statement

Microplastic pollution in water is difficult to identify and assess manually. Traditional inspection requires time and may depend on human observation.

MicroClear provides an automated approach where a microscope image is analyzed using AI to detect microplastic particles. The detected microplastic count is then combined with water-quality parameters to determine the pollution risk.

---

## 3. Main Objectives

* Detect microplastic particles from microscope images.
* Automatically count detected microplastic particles.
* Measure the detection confidence of YOLOv8.
* Collect important water-quality parameters:

  * pH
  * TDS
  * Turbidity
* Classify water pollution risk using Machine Learning.
* Apply safety rules to produce the final risk level.
* Provide a simple web-based interface using Streamlit.

---

## 4. System Workflow

```text
Microscope Image
       ↓
    YOLOv8
       ↓
Microplastic Detection
       ↓
Microplastic Count + Confidence
       ↓
pH + TDS + Turbidity
       ↓
Random Forest Model
       ↓
ML Risk Prediction
       ↓
Rule-Based Safety Layer
       ↓
Final Risk Level
```

The final risk level can be:

* **Low**
* **Medium**
* **High**

---

## 5. AI Models

### YOLOv8 — Microplastic Detection

YOLOv8 is used to detect microplastic particles in microscope images.

The model:

* Accepts microscope images.
* Detects microplastic particles.
* Draws bounding boxes around detected particles.
* Counts the detected particles.
* Provides detection confidence.

Detection is performed using:

```text
Image Size: 640 × 640
Confidence Threshold: 0.50
```

### Random Forest — Risk Classification

The Random Forest model classifies water pollution risk using:

```text
pH
TDS
Turbidity
Microplastic Count
```

The model predicts one of three classes:

```text
Low
Medium
High
```

---

## 6. Final Risk Decision

MicroClear uses a hybrid approach combining Machine Learning with rule-based safety logic.

The Random Forest model first provides the ML-based risk prediction.

A safety layer then checks the water-quality conditions. If any important parameter falls into the **Poor** category, the final risk can be overridden to **High**.

This provides an additional safety check instead of relying only on the Machine Learning prediction.

---

## 7. Model Performance

### YOLOv8 Validation Results

| Metric    |     Result |
| --------- | ---------: |
| Precision | **94.87%** |
| Recall    | **89.04%** |
| F1-Score  | **91.86%** |
| mAP@50    | **93.38%** |
| mAP@50–95 | **66.67%** |

The YOLOv8 model achieved **94.87% precision**, exceeding the project's required **85% precision target**.

### YOLOv8 Test Results

| Metric    |     Result |
| --------- | ---------: |
| Precision | **96.70%** |
| Recall    | **91.26%** |
| F1-Score  | **93.90%** |
| mAP@50    | **95.32%** |
| mAP@50–95 | **68.82%** |

### Random Forest Results

| Metric    |     Result |
| --------- | ---------: |
| Accuracy  | **99.00%** |
| Precision | **99.03%** |
| Recall    | **99.00%** |
| F1-Score  | **99.00%** |

The Random Forest model achieved **99% accuracy**, exceeding the project's required **85% accuracy target**.

---

## 8. Technology Stack

### Frontend / Interface

* Streamlit

### Programming Language

* Python

### Computer Vision

* YOLOv8
* Ultralytics

### Machine Learning

* Random Forest
* Scikit-learn

### Data Processing

* NumPy
* Pandas

### Visualization

* Matplotlib

### Image Processing

* Pillow

---

## 9. Project Structure

```text
MicroClear/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── microplastic_yolo.pt
│   └── risk_model.pkl
│
├── datasets/
│
├── EVALUATION/
│   └── result_randomforest/
│       ├── performance_metrics.txt
│       ├── confusion_matrix.csv
│       └── confusion_matrix.png
│
└── ...
```

---

## 10. Installation

Clone the repository:

```bash
git clone hhttps://github.com/Engr-Usman-Ali/Final-Year-Project.git
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## 11. Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in the browser.

---

## 12. How to Use

### Step 1 — Upload Image

Upload a clear microscope image of the water sample.

### Step 2 — Run Detection

Run the YOLOv8 detector.

The system automatically displays:

* Detected microplastic particles
* Bounding boxes
* Microplastic count
* Detection confidence

### Step 3 — Enter Water Parameters

Enter:

* pH
* TDS
* Turbidity

The microplastic count does **not** need to be entered manually because it is obtained automatically from YOLOv8.

### Step 4 — Analyze Risk

The system sends the detected microplastic count and water-quality parameters to the Random Forest model.

The rule-based safety layer then checks the conditions and produces the **final pollution risk level**.

---

## 13. Expected Output

After analyzing the microscope image and water-quality parameters, MicroClear provides a detailed assessment through the following sections:

### 📋 Analysis Results

The system displays the overall analysis results, including:

* Final Risk Level
* ML Predicted Risk
* ML Confidence
* Urgency Level
* Recommended Treatment Method
* Treatment Plan

### 📊 Risk Assessment

The system determines the final pollution risk level:

```text
LOW
MEDIUM
HIGH
```

It also displays the Machine Learning prediction and confidence, for example:

```text
Final Risk Level: MEDIUM
ML Predicted: Medium
ML Confidence: 98.5%
```

### 💡 Recommendations

Based on the final risk level, MicroClear provides a recommended response, such as:

* Urgency level
* Recommended treatment method
* Required action
* Treatment plan
* Implementation steps

For example:

```text
Urgency Level:
Install filtration within 1 week

Treatment Method:
Household Water Purifier with Carbon Filter

Treatment Plan:
FILTRATION RECOMMENDED
```

The system can also provide implementation steps such as filtration, microfiltration, regular water testing, and other appropriate treatment recommendations.

### 📈 Detailed Report

The detailed report provides complete information about the assessment.

#### 🤖 ML Model Information

The report includes:

* Algorithm used
* ML prediction
* ML confidence
* Rule-based assessment
* Safety override status

Example:

```text
Algorithm: Random Forest Classifier
ML Prediction: Medium
ML Confidence: 98.5%
Rule-Based Assessment: Medium
Safety Override Applied: No
```

#### 🖼️ Sample Image vs. Detection

The system displays:

* Original microscope image
* YOLOv8 detection image
* Detected microplastic particles
* Microplastic count

Example:

```text
YOLOv8 Detection: 1 particle found
```

#### 📋 Parameter Summary

The system summarizes the measured parameters:

| Parameter          |     Example Value |
| ------------------ | ----------------: |
| Microplastic Count | 1 particle/100 ml |
| pH                 |               7.1 |
| TDS                |        270.0 mg/L |
| Turbidity          |           1.0 NTU |

#### 📊 Parameter Categorization

Each parameter is categorized according to the system's defined standards.

| Parameter          |      Value | Category | Status |
| ------------------ | ---------: | -------- | ------ |
| pH                 |        7.1 | Good     | ✅      |
| TDS                | 270.0 mg/L | Good     | ✅      |
| Turbidity          |    1.0 NTU | Moderate | ⚠️     |
| Microplastic Count | 1 particle | Moderate | ⚠️     |

The report also shows the corresponding standard/reference range for each parameter.

### 📚 Reference Standards

The system provides the reference standards used for the water-quality assessment, including:

* Pakistan Council of Research in Water Resources (PCRWR)
* Environmental Protection Agency, Azad Jammu & Kashmir (EPA-AJK)
* World Health Organization (WHO) Drinking Water Guidelines
* Muslim Hands International WASH Program

Overall, MicroClear provides both a **simple final risk level** and a **detailed technical report** containing the AI prediction, confidence, detected microplastics, water-quality parameters, parameter categories, safety decision, and recommended treatment actions.
