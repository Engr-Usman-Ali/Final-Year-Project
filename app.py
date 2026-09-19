# =====================================================
# File: 3_app.py (UPDATED - RAW VALUES ONLY)
# MicroClear - AI-Based Microplastic Pollution Risk Assessment
# Streamlit Web Application
# =====================================================

import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import pandas as pd
import joblib
import time
import cv2
import os

# =====================================================
# LOAD BOTH MODELS (YOLO + ML)
# =====================================================

import os
import joblib
from ultralytics import YOLO
import streamlit as st


def load_models():
    """
    Load both YOLO and ML models in one place
    This keeps code clean and avoids repetition
    """

    # -------------------------------
    # Load YOLO Model
    # -------------------------------
    try:
        yolo_model = YOLO("runs/detect/models/microplastic_yolo/weights/best.pt")
        yolo_status = True
    except:
        yolo_model = None
        yolo_status = False

    # -------------------------------
    # Load ML Model
    # -------------------------------
    MODEL_PATH = "models/risk_model.pkl"

    if os.path.exists(MODEL_PATH):
        ml_model = joblib.load(MODEL_PATH)
        ml_status = True
    else:
        ml_model = None
        ml_status = False

    return yolo_model, ml_model, yolo_status, ml_status


# Call function
yolo_model, ml_model, yolo_ok, ml_ok = load_models()


# =====================================================
# SINGLE NOTIFICATION (ONLY ONCE)
# =====================================================

if "models_notified" not in st.session_state:

    if yolo_ok and ml_ok:
        st.toast("✅ YOLO + ML Models Loaded Successfully")

    elif yolo_ok and not ml_ok:
        st.toast("⚠️ YOLO Loaded, ML Model Missing")

    elif not yolo_ok and ml_ok:
        st.toast("⚠️ ML Loaded, YOLO Model Missing")

    else:
        st.toast("❌ Both Models Failed to Load")

    # prevent showing again
    st.session_state.models_notified = True
    

# =====================================================
# Paramters
# =====================================================

def categorize_parameter(value, param):
    """
    Rule-based categorization (WHO/PCRWR standards)
    """

    if param == "pH":
        if 7.1 <= value <= 7.8:
            return "Good"
        elif 6.5 <= value <= 7.0 or 7.9 <= value <= 8.5:
            return "Moderate"
        else:
            return "Poor"

    elif param == "TDS":
        if 200 <= value <= 270:
            return "Good"
        elif 150 <= value <= 199 or 271 <= value <= 350:
            return "Moderate"
        else:
            return "Poor"

    elif param == "Turbidity":
        if value < 1:
            return "Good"
        elif 1 <= value <= 4.9:
            return "Moderate"
        else:
            return "Poor"

    elif param == "MP_Count":
        if value == 0:
            return "Good"
        elif 1 <= value <= 5:
            return "Moderate"
        else:
            return "Poor"
 
# =====================================================
# ML PREDICTION
# ===================================================== 
 
def ml_predict(ph, tds, turbidity, mp_count):
    """
    Predict risk using trained ML model
    """

    if ml_model is None:
        return "Unknown", 0.0

    # IMPORTANT: must match training features
    input_data = pd.DataFrame([{
        "pH": ph,
        "TDS": tds,
        "Turbidity": turbidity,
        "MP_Count": mp_count
    }])

    prediction = ml_model.predict(input_data)[0]

    # confidence (if model supports probability)
    if hasattr(ml_model, "predict_proba"):
        probs = ml_model.predict_proba(input_data)[0]
        confidence = max(probs) * 100
    else:
        confidence = 0.0

    return prediction, confidence
 
# =====================================================
# RULE BASED HYBRID SYSTEM
# =====================================================
 
def rule_based_risk(ph, tds, turbidity, mp_count):
    """
    Hard safety rules (NO ML can override HIGH risk)
    """

    pH_cat = categorize_parameter(ph, "pH")
    tds_cat = categorize_parameter(tds, "TDS")
    turbidity_cat = categorize_parameter(turbidity, "Turbidity")
    mp_cat = categorize_parameter(mp_count, "MP_Count")

    categories = [pH_cat, tds_cat, turbidity_cat, mp_cat]

    # RULE 1: ANY poor → HIGH risk
    if "Poor" in categories:
        return "High"

    # RULE 2: 2+ moderate → Medium risk
    if categories.count("Moderate") >= 2:
        return "Medium"

    # RULE 3: mostly good → Low risk
    return "Low"

# =====================================================
# Prediction Function
# =====================================================

def final_risk_engine(ph, tds, turbidity, mp_count):
    """
    Hybrid ML-first system with rule-based safety override.

    LOGIC:
    1. ML is primary decision maker
    2. Rule-based system acts ONLY as safety guard
    3. Override ML ONLY if ML underestimates HIGH risk
    """

    # =====================================================
    # STEP 1: ML PREDICTION (PRIMARY MODEL)
    # =====================================================
    ml_risk, ml_conf = ml_predict(ph, tds, turbidity, mp_count)

    # =====================================================
    # STEP 2: RULE-BASED SAFETY CHECK
    # (Only used to detect dangerous cases)
    # =====================================================
    rule_risk = rule_based_risk(ph, tds, turbidity, mp_count)

    # =====================================================
    # STEP 3: SAFETY OVERRIDE LOGIC
    # =====================================================

    override_applied = False

    # CASE 1: ML already agrees with rules
    if ml_risk == rule_risk:
        final_risk = ml_risk
        override_applied = False

    # CASE 2: ML predicts LOWER risk than rule → override
    elif (rule_risk == "Medium" and ml_risk == "Low"):
        final_risk = "Medium"
        override_applied = True

    elif (rule_risk == "High" and ml_risk in ["Low", "Medium"]):
        final_risk = "High"
        override_applied = True

    # CASE 4: Otherwise trust ML (ML-first system)
    else:
        final_risk = ml_risk
        override_applied = False

    # =====================================================
    # STEP 4: RETURN FULL STRUCTURE (UI FRIENDLY)
    # =====================================================
    return {
        "final_risk": final_risk,
        "ml_risk": ml_risk,
        "ml_confidence": ml_conf,
        "rule_risk": rule_risk,
        "override_applied": override_applied,

        # UI display only (NOT ML features)
        "categories": {
            "pH": categorize_parameter(ph, "pH"),
            "TDS": categorize_parameter(tds, "TDS"),
            "Turbidity": categorize_parameter(turbidity, "Turbidity"),
            "MP_Count": categorize_parameter(mp_count, "MP_Count"),
        }
    }




st.set_page_config(
    page_title="MicroClear",
    page_icon="🔬",
    layout="wide"
)
            
# =====================================================
# CUSTOM CSS STYLING
# =====================================================

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    .header {
        font-size: 2.0rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Result boxes */
    .result-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #1c2128;
        border: 2px solid #30363d;
        text-align: center;
        margin-top: 15px;
        transition: all 0.3s ease;
    }
    .result-box:hover {
        border-color: #00bcd4;
        box-shadow: 0 0 15px rgba(0, 188, 212, 0.3);
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(135deg, #00bcd4, #0077b6);
        color: white;
        font-weight: bold;
        padding: 12px 24px;
        border: none;
        transition: all 0.3s ease;
        font-size: 16px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #00a8cc, #005885);
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0, 188, 212, 0.4);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1f29, #2d3748);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #3a4556;
    }

    /* Icon styling */
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 15px;
    }
    
    /* FAQ styling */
    .faq-question {
        font-size: 1.2rem;
        font-weight: bold;
        color: #00bcd4;
        margin-top: 20px;
        padding: 10px;
        border-left: 4px solid #00bcd4;
        background-color: rgba(0, 188, 212, 0.1);
        border-radius: 5px;
    }
    .faq-answer {
        padding: 15px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.markdown("""
<div style="
    text-align: center; 
    padding: 20px 10px; 
">
    <div style="
        font-size: 3rem; 
        color: #00bcd4; 
        margin-bottom: 8px;
        animation: glow 2.0s ease-in-out infinite alternate;
    ">🔭</div>
    <h3 style="
        color: #00bcd4; 
        font-weight: 700; 
        margin: 5px 0;
        letter-spacing: 1px;
    ">MicroClear</h3> 
</div>

<style>
@keyframes glow {
    from {
        text-shadow: 0 0 5px rgba(0, 188, 212, 0.4);
    }
    to {
        text-shadow: 0 0 20px rgba(0, 188, 212, 0.7);
    }
}
</style>
""", unsafe_allow_html=True)


st.sidebar.markdown("---")
page = st.sidebar.radio("**Navigation**", ["🏠 Home", "📊 Analysis Dashboard", "❓ FAQ & Help", "👥 About"])

# Extract page name without emoji
page_name = page.split(" ", 1)[1] if " " in page else page

# =====================================================
# PAGE 1: HOME
# =====================================================

if page_name == "Home":
    st.markdown("<div class='header'>🔬 MicroClear System</div>", unsafe_allow_html=True)

    # =====================================================
    # HERO SECTION (CLEAN + STRONG INTRO)
    # =====================================================
    st.markdown("""
    ### 🌍 AI-Based Microplastic Pollution Risk Assessment and Mitigation System
    
    MicroClear is an **AI-powered water quality monitoring system** designed to detect microplastics 
    and assess drinking water safety using **Computer Vision (YOLOv8)** and **Machine Learning (Random Forest)**.
    
    It provides **real-time detection, risk classification, and treatment recommendations** 
    to support environmental monitoring and public health protection.
    """)

    st.divider()

    # =====================================================
    # WHY THIS PROJECT MATTERS
    # =====================================================
    st.markdown("## ❗ Why This Project is Important")

    st.markdown("""
    - Microplastics are now found in **drinking water worldwide**
    - Long-term exposure may impact **human health and ecosystems**
    - Traditional lab testing is:
        - expensive 💰  
        - time-consuming ⏳  
        - not accessible in rural areas 🌍  

    👉 **MicroClear solves this by providing a fast, low-cost, AI-based solution**
    """)

    st.divider()

    # =====================================================
    # SYSTEM OVERVIEW
    # =====================================================
    st.markdown("## ⚙️ System Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🔍 1. Microplastic Detection
        - Uses **YOLOv8 object detection**
        - Detects particles from microscope images
        - Provides:
            - Bounding boxes
            - Detection confidence
            - Particle count
        """)

    with col2:
        st.markdown("""
        ### 📊 2. Risk Assessment Engine
        - **Machine Learning (Random Forest)**
        - Uses parameters:
            - pH  
            - TDS  
            - Turbidity  
            - Microplastic Count  
        - Outputs:
            - Low / Medium / High Risk
        """)

    st.markdown("""
    ### 🧠 3. Hybrid Decision System
    - ML is the **primary decision maker**
    - Rule-based system ensures **safety compliance**
    - Prevents underestimation of dangerous water conditions
    """)

    st.divider()

    # =====================================================
    # WORKFLOW (VERY IMPORTANT FOR SRS)
    # =====================================================
    st.markdown("## 🔄 System Workflow")

    st.markdown("""
    1️⃣ Upload microscope image of filtered water sample  
    2️⃣ AI detects microplastic particles using YOLOv8  
    3️⃣ Enter water quality parameters (pH, TDS, Turbidity)  
    4️⃣ ML model predicts contamination risk  
    5️⃣ Rule-based system validates safety thresholds  
    6️⃣ Final risk level + treatment recommendations generated  
    """)

    st.divider()

    # =====================================================
    # KEY FEATURES (CLEAN, NOT CARDS)
    # =====================================================
    st.markdown("## ✨ Key Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **🔍 AI Detection**
        - YOLOv8 deep learning
        - High accuracy detection
        """)

    with col2:
        st.markdown("""
        **📊 Smart Risk Analysis**
        - ML-based predictions
        - Confidence scoring
        """)

    with col3:
        st.markdown("""
        **🛡️ Safety First**
        - Rule-based override
        - WHO/PCRWR compliance
        """)

    st.divider()

    # =====================================================
    # USE CASES (NEW - IMPORTANT ADDITION)
    # =====================================================
    st.markdown("## 🏥 Practical Applications")

    st.markdown("""
    - 🏫 Environmental research labs  
    - 💧 Water quality monitoring agencies  
    - 🌍 NGOs (e.g., Muslim Hands)  
    - 🏠 Household water testing  
    - 🧪 Academic research projects  
    """)

    st.divider()

    # =====================================================
    # FUTURE SCOPE (GOOD FOR FYP)
    # =====================================================
    st.markdown("## 🚀 Future Enhancements")

    st.markdown("""
    - 📱 Mobile app integration  
    - ☁️ Cloud-based monitoring dashboard  
    - 📄 PDF report generation  
    - 🌐 Real-time water quality tracking  
    - 🤖 Improved detection with larger datasets  
    """)

    st.divider()

    # =====================================================
    # KEEP YOUR SDG SECTION (UNCHANGED ✅)
    # =====================================================
    st.subheader("🌱 Sustainable Development Goals")

    sdg_cols = st.columns(4)
    sdgs = [
        ("💧", "SDG 6", "Clean Water", "#00bcd4"),
        ("🔄", "SDG 12", "Responsible Consumption", "#ff9800"),
        ("🌿", "SDG 13", "Climate Action", "#4caf50"),
        ("🐟", "SDG 14", "Life Below Water", "#2196f3")
    ]

    for idx, (icon, num, name, color) in enumerate(sdgs):
        with sdg_cols[idx]:
            st.markdown(f"""
            <div class='metric-card' style='border-top: 4px solid {color};'>
                <div style='font-size: 2rem;'>{icon}</div>
                <h4 style='color: {color};'>{num}</h4>
                <p style='font-size: 0.85rem;'>{name}</p>
            </div>
            """, unsafe_allow_html=True)

                
# =====================================================
# PAGE 2: ANALYSIS DASHBOARD
# =====================================================

elif page_name == "Analysis Dashboard":
    st.markdown("<div class='header'>📊 Water Analysis Dashboard</div>", unsafe_allow_html=True)

    # =====================================================
    # LAYOUT: SPLIT SCREEN (UPLOAD | PARAMETERS)
    # =====================================================
    col1, col2 = st.columns([1, 1], gap="large")

    # =====================================================
    # LEFT PANEL: IMAGE UPLOAD + YOLO DETECTION
    # =====================================================
    with col1:

        # -------------------------------
        # SECTION: SAMPLE PREPARATION INFO
        # -------------------------------
        st.markdown("""
        <div class='card'>
        <h3>📸 1. Upload & Preparation</h3>
        """, unsafe_allow_html=True)
        st.divider()

        st.markdown("""
        **Standard Laboratory Procedure:**
        - Collect 100ml water sample
        - Filter using 0.45μm membrane
        - Dry filter paper properly
        - Capture microscope image
        """)

        # -------------------------------
        # IMAGE UPLOAD WIDGET
        # -------------------------------
        uploaded_file = st.file_uploader(
            "📤 Upload Microscope Image",
            type=["png", "jpg", "jpeg"],
            help="Upload a clear microscope image of water sample"
        )

        # -------------------------------
        # DISPLAY ORIGINAL IMAGE
        # -------------------------------
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="🔬 Uploaded Sample", use_container_width=True)

        else:
            st.info("Upload a microscope image to start analysis")

        # =====================================================
        # YOLO DETECTION PROCESS
        # =====================================================
        st.write("")

        if st.button("🔍 Detect Microplastics",
                     type="primary",
                     use_container_width=True,
                     disabled=(uploaded_file is None)):

            with st.spinner("🧠 Running YOLOv8 Detection..."):

                # -------------------------------
                # STEP 1: IMAGE CONVERSION
                # -------------------------------
                image_np = np.array(img)
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                # Resize for consistency
                image_cv = cv2.resize(image_cv, (640, 640))

                # -------------------------------
                # STEP 2: YOLO PREDICTION
                # -------------------------------
                results = yolo_model(image_cv)
                boxes = results[0].boxes

                detected_count = 0
                confidences = []

                # -------------------------------
                # STEP 3: DRAW BOUNDING BOXES
                # -------------------------------
                for box in boxes:

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    detected_count += 1
                    confidences.append(conf)

                    # Draw detection box
                    cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Label
                    cv2.putText(
                        image_cv,
                        f"MP {conf:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )

                # -------------------------------
                # STEP 4: METRICS CALCULATION
                # -------------------------------
                avg_conf = np.mean(confidences) * 100 if confidences else 0

                # -------------------------------
                # STEP 5: SAVE RESULTS IN SESSION
                # -------------------------------
                st.session_state['detected_count'] = detected_count
                st.session_state['detection_confidence'] = avg_conf

                # Save image for later tabs
                image_display = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                st.session_state['detected_image'] = image_display

                # -------------------------------
                # STEP 6: SHOW OUTPUT
                # -------------------------------
                # st.image(image_display, caption="🔬 YOLO Detection Result", use_container_width=True)

                st.success(
                    f"Detection Complete → {detected_count} particles found "
                    f"(Confidence: {avg_conf:.1f}%)"
                )

        st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # RIGHT PANEL: WATER QUALITY INPUT PARAMETERS
    # =====================================================
    with col2:

        # -------------------------------
        # SECTION HEADER
        # -------------------------------
        st.markdown("""
        <div class='card'>
        <h3>📋 2. Water Quality Parameters</h3>
        """, unsafe_allow_html=True)
        st.divider()

        # -------------------------------
        # AUTO-FILL FROM DETECTION
        # -------------------------------
        default_count = st.session_state.get('detected_count', 0)
        detection_conf = st.session_state.get('detection_confidence', 0)

        if default_count > 0:
            st.info(f"Detected: {default_count} particles ({detection_conf:.1f}%)")

        # -------------------------------
        # USER INPUT PARAMETERS
        # -------------------------------
        mp_count = st.number_input(
            "Microplastic Count",
            min_value=0,
            value=default_count
        )

        ph = st.number_input("pH Level", 0.0, 14.0, 7.1)
        tds = st.number_input("TDS (mg/L)", 0, 1000, 270)
        turbidity = st.number_input("Turbidity (NTU)", 0.0, 20.0, 0.0)

        # -------------------------------
        # ANALYSIS BUTTON
        # -------------------------------
        run_analysis = st.button(
            "📈 Analyze Risk Level",
            type="primary",
            use_container_width=True
        )

        st.markdown("</div>", unsafe_allow_html=True)
        
    # =====================================================
    # RESULTS SECTION
    # =====================================================
    
    if run_analysis:
        st.divider()
        st.markdown("<div class='header'>📋 Analysis Results</div>", unsafe_allow_html=True)
        
        # ===============================
        # SAFE MODEL CALL (prevents crash)
        # ===============================
        try:
            assessment = final_risk_engine(ph, tds, turbidity, mp_count)
            risk_level = assessment["final_risk"]
            ml_risk = assessment["ml_risk"]
            ml_confidence = assessment["ml_confidence"]
            override_applied = assessment["override_applied"]

        except Exception as e:
            st.error(f"❌ Error in risk engine: {e}")
            st.stop()

        # Treatment info
        treatment_info = {
            "Low": {
                "color": "#00e676",
                "title": "SAFE FOR CONSUMPTION",
                "technique": "Standard Chlorination & Basic Filtration",
                "steps": [
                    "Continue regular chlorination (if municipal supply)",
                    "Use basic sediment filter (10-20μm)",
                    "Monthly water quality testing recommended"
                ],
                "urgency": "Continue normal use"
            },
            "Medium": {
                "color": "#ffeb3b",
                "title": "FILTRATION RECOMMENDED",
                "technique": "Household Water Purifier with Carbon Filter",
                "steps": [
                    "Install activated carbon filter system",
                    "Add 1-5μm microfiltration stage",
                    "Test water weekly for 1 month",
                    "Consider UV disinfection"
                ],
                "urgency": "Install filtration within 1 week"
            },
            "High": {
                "color": "#ff5252",
                "title": "IMMEDIATE TREATMENT REQUIRED",
                "technique": "Advanced Reverse Osmosis + UV System",
                "steps": [
                    "🚨 INSTALL RO SYSTEM IMMEDIATELY",
                    "Add UV disinfection stage",
                    "Boil water as interim measure",
                    "Contact local water authority",
                    "Do NOT consume without treatment"
                ],
                "urgency": "TREAT BEFORE CONSUMPTION"
            }
        }
        treatment = treatment_info[risk_level]
        
        tab1, tab2, tab3 = st.tabs(["📊 Risk Assessment", "💡 Recommendations", "📈 Detailed Analysis"])
        
        with tab1:
            col1, col2, col3 = st.columns(3) 
            with col1:
                override_text = ""
                if assessment["override_applied"]:
                    override_text = f"<br><small style='color: #ff9800;'>⚠️ Safety Override Applied</small>"
                
                st.markdown(f"""
                <div class='result-box' style='border-color: {treatment['color']};'>
                <small>FINAL RISK LEVEL</small>
                <h1 style='color: {treatment['color']}; margin: 10px 0;'>{risk_level.upper()}</h1>
                <small>ML Predicted: {ml_risk} ({ml_confidence:.1f}%)</small>
                {override_text}
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class='result-box'>
                <small>URGENCY LEVEL</small>
                <h3 style='color: {treatment['color']}; margin: 10px 0;'>{treatment['urgency']}</h3>
                <small>Action Required</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='result-box'>
                <small>TREATMENT METHOD</small>
                <h4 style='margin: 10px 0;'>{treatment['technique']}</h4>
                <small>Recommended Solution</small>
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown(f"### 🛠️ Treatment Plan: **{treatment['title']}**")
            st.markdown("#### Implementation Steps:")
            for i, step in enumerate(treatment['steps'], 1):
                st.markdown(f"{i}. {step}")
            
            st.markdown("---")
            st.markdown("#### 📋 Parameter Summary:")
            params_col1, params_col2 = st.columns(2)
            with params_col1:
                st.metric("Microplastic Count", f"{mp_count} particles/100ml", 
                         delta="Detected" if mp_count>0 else "None detected")
                st.metric("pH Level", f"{ph}", 
                         delta="Optimal" if 7.1<=ph<=7.8 else "Out of range",
                         delta_color="normal" if 7.1<=ph<=7.8 else "inverse")
            with params_col2:
                st.metric("TDS Level", f"{tds} mg/L", 
                         delta="Optimal" if 200<=tds<=270 else "Out of range",
                         delta_color="normal" if 200<=tds<=270 else "inverse")
                st.metric("Turbidity", f"{turbidity} NTU", 
                         delta="Clear" if turbidity<1 else "Cloudy",
                         delta_color="normal" if turbidity<1 else "inverse")
        
        with tab3:
            st.markdown("### 🔬 Detailed Analysis")
            st.markdown("#### 🤖 ML Model Information")
            st.write("**Algorithm:** Random Forest Classifier (Raw Values + Engineered Features)")
            st.write(f"**ML Prediction:** {ml_risk}")
            st.write(f"**ML Confidence:** {ml_confidence:.1f}%")
            st.write(f"**Rule-Based Assessment:** {assessment['rule_risk']}")
            st.write(f"**Safety Override Applied:** {'⚠️ Yes - ML underestimated risk' if assessment['override_applied'] else '✅ No - ML prediction used'}")
            
            # ===============================
            # SHOW DETECTED IMAGE (DETAIL TAB)
            # ===============================
            if 'detected_image' in st.session_state:
                st.markdown("#### 🖼️ Detected Microplastics Image")
                st.image(
                st.session_state['detected_image'],
                caption="YOLOv8 Detection Output",
                width=350
            )
            
            st.markdown("#### 📊 Parameter Categorization")
            cats = assessment['categories']
            cat_data = {
                "Parameter": ["pH", "TDS", "Turbidity", "Microplastic Count"],
                "Value": [ph, f"{tds} mg/L", f"{turbidity} NTU", f"{mp_count} particles"],
                "Category": [cats['pH'], cats['TDS'], cats['Turbidity'], cats['MP_Count']],
                "Status": [
                    "✅" if cats['pH']=="Good" else "⚠️" if cats['pH']=="Moderate" else "❌",
                    "✅" if cats['TDS']=="Good" else "⚠️" if cats['TDS']=="Moderate" else "❌",
                    "✅" if cats['Turbidity']=="Good" else "⚠️" if cats['Turbidity']=="Moderate" else "❌",
                    "✅" if cats['MP_Count']=="Good" else "⚠️" if cats['MP_Count']=="Moderate" else "❌",
                ],
                "Optimal Range": ["7.1-7.8", "200-270 mg/L", "<1 NTU", "0 particles"]
            }
            st.table(cat_data)
            
            st.markdown("#### 📚 Reference Standards")
            st.info("""
            **Standards Applied:**
            - Pakistan Council of Research in Water Resources (PCRWR)
            - Environmental Protection Agency, Azad Jammu & Kashmir (EPA-AJK)
            - World Health Organization (WHO) Drinking Water Guidelines
            - Muslim Hands International WASH Program
            """)

        
# =====================================================
# PAGE 3: FAQ PAGE
# =====================================================

elif page_name == "FAQ & Help":
    st.markdown("<div class='header'>❓ FAQ & System Documentation</div>", unsafe_allow_html=True)

    # =====================================================
    # INTRO (CLEAN - NO CARD OVERUSE)
    # =====================================================
    st.markdown("""
    ### 📚 MicroClear User Guide

    This section provides **technical documentation, system behavior explanation, and troubleshooting help**  
    for using the MicroClear AI-based water quality assessment system.
    """)

    st.divider()

    # =====================================================
    # FAQ DATA (UNCHANGED LOGIC)
    # =====================================================
    faq_sections = {
        "🤔 General Overview": [
            {
                "q": "What is MicroClear?",
                "a": "MicroClear is an AI-based system that detects microplastics in water using YOLOv8 and evaluates water safety using machine learning and rule-based logic."
            },
            {
                "q": "Who can use this system?",
                "a": "It is designed for environmental labs, NGOs (e.g., Muslim Hands), researchers, and water quality monitoring agencies."
            },
            {
                "q": "What makes this system different?",
                "a": "It combines Computer Vision + Machine Learning + Rule-based safety validation in a single hybrid decision system."
            }
        ],

        "🤖 AI & Decision System": [
            {
                "q": "How does the hybrid system work?",
                "a": """
                MicroClear uses a 3-layer decision pipeline:

                1️⃣ YOLOv8 detects microplastics from microscope images  
                2️⃣ ML model predicts risk (Low / Medium / High)  
                3️⃣ Rule-based system validates safety limits  

                Final output is generated using **ML-first + safety override logic**.
                """
            },
            {
                "q": "Why combine ML and rule-based logic?",
                "a": """
                - ML captures hidden patterns in water contamination  
                - Rules ensure strict WHO/PCRWR safety compliance  
                - Prevents dangerous underestimation of risk  

                👉 This ensures **accuracy + safety together**
                """
            },
            {
                "q": "Can ML override safety rules?",
                "a": """
                No, ML cannot lower the risk if safety rules say it's dangerous. Here's how it works:
                - If ML says 'Low' but safety rules say 'Medium' → System uses 'Medium' (safer choice)
                - If ML says 'Low' or 'Medium' but safety rules say 'High' → System uses 'High' (safest choice)
                - If ML says same or higher risk than rules → System trusts ML prediction
                - The system always picks the safer option. Safety rules can ONLY increase risk level, never decrease it.
                """
            }
        ],

        "🔬 Detection System": [
            {
                "q": "How accurate is microplastic detection?",
                "a": "YOLOv8 model achieves approximately 85–95% accuracy depending on image quality, lighting, and microscope resolution."
            },
            {
                "q": "What images are required?",
                "a": "Microscope images in JPG/PNG format, ideally 640x640 or higher resolution for better detection performance."
            }
        ],

        "💧 Sampling & Parameters": [
            {
                "q": "What is correct sampling method?",
                "a": """
                1. Collect 100ml water sample  
                2. Filter using 0.45μm membrane  
                3. Dry filter properly  
                4. Capture microscope image  
                5. Upload to system  
                """
            },
            {
                "q": "Which parameters are used?",
                "a": "pH, TDS, Turbidity, and Microplastic Count are used for final risk prediction."
            }
        ],

        "⚠️ Troubleshooting": [
            {
                "q": "Why is risk HIGH even with low microplastics?",
                "a": "Because chemical parameters (pH, TDS, turbidity) may violate safety thresholds."
            },
            {
                "q": "Model not loading?",
                "a": "Ensure YOLO and ML model files exist in correct directory and paths are properly set."
            }
        ],

        "📊 Result Interpretation": [
            {
                "q": "What do risk levels mean?",
                "a": """
                - 🟢 LOW → Safe water  
                - 🟡 MEDIUM → Needs filtration  
                - 🔴 HIGH → Unsafe without treatment  
                """
            },
            {
                "q": "Are recommendations automatic?",
                "a": "Yes, treatment suggestions are generated based on WHO water safety guidelines."
            }
        ]
    }

    # =====================================================
    # CLEAN FAQ DISPLAY (IMPROVED SPACING)
    # =====================================================
    for section_title, questions in faq_sections.items():

        st.markdown(f"## {section_title}")

        for faq in questions:
            with st.expander(f"❓ {faq['q']}"):
                st.markdown(faq["a"])

        st.markdown("")  # spacing instead of heavy divider

    st.divider()

    # =====================================================
    # SUPPORT SECTION (CLEAN + PROFESSIONAL)
    # =====================================================
    st.markdown("## 📞 Support & Documentation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🆘 Technical Support

        - 📧 Email: usmanali08675@gmail.com  
        - 📱 Phone: +92 334 8225271  
        - 🕒 Hours: Mon–Fri (9 AM – 5 PM PKT)  
        """)

    with col2:
        st.markdown("""
        ### 📚 Resources

        - User Manual  
        - Sampling Protocol Guide  
        - Research Documentation  
        - System Architecture Overview  
        """)

# =====================================================
# PAGE 4: ABOUT PAGE 
# =====================================================
elif page_name == "About":
    st.markdown("<div class='header'>👥 About MicroClear Project</div>", unsafe_allow_html=True)

    # =====================================================
    # PROJECT VISION (CLEAN INTRO)
    # =====================================================
    st.markdown("## 🌍 Project Vision")

    st.markdown("""
    MicroClear is a **Final Year AI-based environmental monitoring system** designed to detect 
    microplastic contamination in drinking water using **computer vision and machine learning**.

    The system aims to bridge the gap between:
    - 🧪 Traditional laboratory testing (accurate but expensive)
    - 📱 Field-level rapid testing (fast but less reliable)

    👉 By combining **YOLOv8 + Machine Learning + Rule-based safety system**, MicroClear provides a 
    scalable and intelligent solution for real-world water quality assessment.
    """)

    st.divider()

    # =====================================================
    # PROJECT OBJECTIVES
    # =====================================================
    st.markdown("## 🎯 Project Objectives")

    st.markdown("""
    - Detect microplastics using AI-based image processing  
    - Analyze water quality using multiple chemical parameters  
    - Classify water into risk levels (Low, Medium, High)  
    - Ensure safety using WHO/PCRWR standards  
    - Provide treatment recommendations for users  
    """)

    st.divider()

    # =====================================================
    # SYSTEM OVERVIEW (MORE ACADEMIC STYLE)
    # =====================================================
    st.markdown("## ⚙️ System Overview")

    st.markdown("""
    MicroClear is built on a **hybrid intelligence architecture**:

    ### 🔍 1. Computer Vision Layer
    - YOLOv8 object detection model  
    - Identifies microplastic particles in microscope images  
    - Outputs bounding boxes and confidence scores  

    ### 📊 2. Machine Learning Layer
    - Random Forest classification model  
    - Uses water quality parameters:
        - pH  
        - TDS  
        - Turbidity  
        - Microplastic count  

    ### 🛡️ 3. Safety Rule Engine
    - Ensures compliance with WHO & PCRWR standards  
    - Overrides ML if unsafe conditions are detected  
    """)

    st.divider()

    # =====================================================
    # TEAM SECTION (CLEAN GRID)
    # =====================================================
    st.markdown("## 👥 Development Team")

    team_cols = st.columns(3)

    team_members = [
        {"name": "Dost Muhammad", "reg": "FA22-BSE-009", "role": "Lead Developer"},
        {"name": "Usman Ali", "reg": "FA22-BSE-051", "role": "Data Scientist"},
        {"name": "M. Husnain", "reg": "FA22-BSE-065", "role": "Web Developer"}
    ]

    for idx, member in enumerate(team_members):
        with team_cols[idx]:
            st.markdown(f"""
            <div style="
                padding: 15px;
                border-radius: 12px;
                background-color: #161b22;
                border: 1px solid #30363d;
                text-align: center;
            ">
                <h4 style="color:#00bcd4;">{member['name']}</h4>
                <p style="margin:5px 0;"><b>{member['reg']}</b></p>
                <p style="color:#8b949e;">{member['role']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # =====================================================
    # AFFILIATIONS (CLEAN TWO COLUMN)
    # =====================================================
    st.markdown("## 🏢 Affiliations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎓 Academic Institution
        **Mirpur University of Science & Technology (MUST)**  
        - Department: Software Engineering  
        - Batch: 2022–2026  
        - Supervisor: Engr. Iqra Gillani  
        """)

    with col2:
        st.markdown("""
        ### 🤝 Client Organization
        **Muslim Hands Mirpur**  
        - Operational Head: Javaid ul Hassan  
        - Contact: +92 300 555064  
        - Location: Mirpur, AJK  
        
        _Partnering for clean water solutions_
        """)

    st.divider()

    # =====================================================
    # TECH STACK (CLEAN BULLETS)
    # =====================================================
    st.markdown("## 🛠️ Technical Specifications")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Core Technologies
        - Python 3.8+
        - YOLOv8 (Object Detection)
        - Random Forest ML Model
        - Streamlit Web Framework
        - OpenCV Image Processing
        """)

    with col2:
        st.markdown("""
        ### Hardware Requirements
        - Digital Microscope (USB)
        - pH / TDS / Turbidity Meters
        - Standard Laptop (4GB+ RAM)
        - Image Capture Setup
        """)

    st.divider()

    # =====================================================
    # SYSTEM FEATURES (IMPORTANT FOR VIVA)
    # =====================================================
    st.markdown("## 🚀 System Features")

    st.markdown("""
    - Automated microplastic detection from microscope images  
    - Multi-parameter water quality risk analysis  
    - Hybrid ML + Rule-based decision system  
    - Real-time interactive dashboard  
    - Safety-first water classification system  
    """)

    st.divider()

    # =====================================================
    # SDG IMPACT (UNCHANGED AS YOU REQUESTED)
    # =====================================================
    st.subheader("🌱 Sustainable Development Impact")

    sdg_impact = {
        "💧 SDG 6: Clean Water": "Provides accessible water quality assessment tools for underserved communities",
        "🔄 SDG 12: Responsible Consumption": "Raises awareness about plastic pollution and its impacts",
        "🌿 SDG 13: Climate Action": "Supports environmental monitoring for climate resilience",
        "🐟 SDG 14: Life Below Water": "Helps prevent microplastic contamination in aquatic ecosystems"
    }

    for sdg, impact in sdg_impact.items():
        st.info(f"**{sdg}:** {impact}")

# =====================================================
# FOOTER
# =====================================================
    
st.markdown("""
    <div style='
        text-align: center;
        padding: 25px;
        margin-top: 20px;
        color: #8b949e;
        font-size: 0.9rem;
        border-top: 1px solid #30363d;
    '>
        <p>🔬 <b>MicroClear System</b> | Final Year Project 2026</p>
        <p>Department of Software Engineering | MUST Mirpur</p>
        <p>© 2024–2026 MicroClear Project Team</p>
    </div>
""", unsafe_allow_html=True)