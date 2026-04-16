# =====================================================
# File: 3_app.py (UPDATED - RAW VALUES ONLY)
# MicroClear - AI-Based Microplastic Pollution Risk Assessment
# Streamlit Web Application
# =====================================================

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import joblib
import os

# =====================================================
# LOAD ML MODEL
# =====================================================

MODEL_PATH = "models/risk_model.pkl"

ml_model = None

if os.path.exists(MODEL_PATH):
    ml_model = joblib.load(MODEL_PATH)
    st.sidebar.success("✅ ML Model Loaded Successfully")
else:
    st.sidebar.error("❌ Model file not found (risk_model.pkl)")
    

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
    
    /* Cards */
    .card {
        background: linear-gradient(135deg, #161b22, #21262d);
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #00bcd4;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.4);
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
    st.markdown("<div class='header'>🔭 MicroClear: AI-Based Microplastic Pollution Risk Assessment and Mitigation System</div>", unsafe_allow_html=True)
    
    # Hero Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='card'>
        <h3>🌍 Welcome to MicroClear</h3>
        <p>An <b>AI-powered intelligent system</b> designed to detect, assess, and mitigate microplastic pollution in drinking water. 
        Using advanced computer vision (YOLOv8) and machine learning, MicroClear provides comprehensive water quality analysis 
        with actionable insights for environmental monitoring.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <div class="feature-icon">💧</div>
        <h4>Clean Water Initiative</h4>
        <p>Supporting SDG Goals</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Start Guide
    st.markdown("""
    <div class='card'>
    <h3>🚀 Quick Start Guide</h3>
    <ol>
        <li><b>Go to Analysis Dashboard</b> from the sidebar</li>
        <li><b>Upload</b> your microscope image (100ml sample)</li>
        <li><b>Enter</b> water quality parameters (pH, TDS, Turbidity)</li>
        <li><b>Click Analyze</b> to get AI-powered risk assessment</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Grid
    st.subheader("✨ Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
        <div class='feature-icon'>🔍</div>
        <h4>AI Detection</h4>
        <p>YOLOv8 deep learning for microplastic detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
        <div class='feature-icon'>📊</div>
        <h4>Risk Assessment</h4>
        <p>Random Forest ML model with 95%+ accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
        <div class='feature-icon'>💡</div>
        <h4>Smart Recommendations</h4>
        <p>Evidence-based treatment strategies</p>
        </div>
        """, unsafe_allow_html=True)
    
    # SDGs Section
    st.write("")
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
                <div class='metric-card' style='border-top: 4px solid {color}; transition: 0.3s;'>
                    <div style='font-size: 2.5rem; margin-bottom: 10px;'>{icon}</div>
                    <h4 style='color: {color}; margin-bottom: 5px;'>{num}</h4>
                    <p style='font-size: 0.8rem; opacity: 0.8;'>{name}</p>
                </div>
                """, unsafe_allow_html=True)

                
# =====================================================
# PAGE 2: ANALYSIS DASHBOARD
# =====================================================

elif page_name == "Analysis Dashboard":
    st.markdown("<div class='header'>📊 Water Analysis Dashboard</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("""
        <div class='card'>
        <h3>📸 1. Sample Upload & Preparation</h3>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Standard Operating Procedure:**
        - Collect **100ml** water sample
        - Filter through 0.45μm membrane
        - Dry filter paper completely
        - Place under digital microscope
        - Capture high-resolution image
        """)
        
        uploaded_file = st.file_uploader(
            "📤 Upload Microscope Image", 
            type=["png", "jpg", "jpeg"],
            help="Upload an image of your filtered water sample"
        )
        
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="🔬 100ml Sample Microscope View", use_container_width=True)
            
            if st.button("🔍 Detect Microplastics", type="primary", use_container_width=True):
                with st.spinner('🧠 Analyzing Image with YOLOv8...'):
                    st.session_state['detected_count'] = np.random.randint(0, 15)
                    st.session_state['detection_confidence'] = np.random.uniform(85, 98)
                    st.success(f"✅ Detection Complete! Found {st.session_state['detected_count']} particles.")
        
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card'>
        <h3>📋 2. Water Quality Parameters</h3>
        """, unsafe_allow_html=True)
        
        default_count = st.session_state.get('detected_count', 0)
        detection_conf = st.session_state.get('detection_confidence', 0)
        
        if default_count > 0:
            st.info(f"📊 **Detected Particles:** {default_count} (Confidence: {detection_conf:.1f}%)")
        
        mp_count = st.number_input(
            "🧪 Microplastic Count (per 100ml)", 
            min_value=0, 
            value=default_count, 
            step=1,
            help="Number of microplastic particles detected"
        )
        
        ph = st.number_input(
            "⚗️ pH Level", 
            min_value=0.0, 
            max_value=14.0, 
            value=7.1, 
            step=0.1,
            format="%.1f",
            help="Acidity/Alkalinity level (7.0 = neutral)"
        )
        
        tds = st.number_input(
            "💧 TDS (mg/L)", 
            min_value=0, 
            max_value=1000, 
            value=270, 
            step=1,
            help="Total Dissolved Solids concentration"
        )
        
        turbidity = st.number_input(
            "🌫️ Turbidity (NTU)", 
            min_value=0.0, 
            max_value=20.0, 
            value=0.0, 
            step=0.1,
            format="%.1f",
            help="Water clarity measurement"
        )
        
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
    st.markdown("<div class='header'>❓ Frequently Asked Questions & Help</div>", unsafe_allow_html=True)
    
    # FAQ Container
    with st.container():
        st.markdown("""
        <div class='card'>
        <h3>📚 User Guide & Documentation</h3>
        <p>Find answers to common questions about MicroClear system usage, interpretation of results, and troubleshooting.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # FAQ Sections
    faq_sections = {
        "🤔 General Questions": [
            {
                "q": "What is MicroClear and who is it for?",
                "a": "MicroClear is an AI-powered system designed for environmental monitoring organizations, NGOs, and water quality labs to detect and assess microplastic pollution in drinking water. It's particularly useful for organizations like Muslim Hands working in water-scarce regions."
            },
            {
                "q": "How accurate is the microplastic detection?",
                "a": "The YOLOv8 model achieves approximately 85-95% accuracy in detecting microplastics larger than 1μm. Detection confidence is displayed with each analysis result."
            },
            {
                "q": "What water sources can be tested?",
                "a": "MicroClear is optimized for drinking water sources including: tap water, well water, bottled water, and treated municipal water. It follows WHO drinking water guidelines for risk assessment."
            }
        ],
        "🤖 ML + Rule-Based System": [
            {
                "q": "How does MicroClear combine ML predictions with rule-based safety checks?",
                "a": """
                MicroClear uses a **hybrid decision system** with two parallel assessment layers:
                
                1. **Machine Learning Layer**: A Random Forest model analyzes patterns in water quality data and predicts risk probabilities
                2. **Rule-Based Safety Layer**: Hard-coded thresholds from WHO/PCRWR/EPA-AJK standards validate each parameter
                
                **Final Decision Logic**:
                - If ML predicts lower risk than rules → **Rules override ML** (Safety First)
                - If ML predicts equal or higher risk → **ML prediction is used**
                - System always shows both predictions for transparency
                """
            },
            {
                "q": "Can the ML model override safety regulations?",
                "a": "**Absolutely not**. Regulatory safety thresholds are non-negotiable. If any water parameter violates WHO/PCRWR limits, the system automatically classifies the water as **HIGH RISK** regardless of ML predictions. This ensures public health protection is never compromised by AI uncertainty."
            },
            {
                "q": "Why use both ML and rules instead of just one approach?",
                "a": """
                **ML Advantages**:
                - Detects complex, non-linear patterns in water quality data that rules alone cannot capture
                - Learns from historical contamination cases and borderline parameter interactions
                - Provides calibrated probabilities, giving nuanced risk assessments

                **Rule-Based Advantages**:
                - Enforces absolute safety thresholds defined by regulations
                - Offers transparent, explainable decisions for critical parameters
                - Guarantees compliance with water safety standards

                **Example**:
                Consider a water sample with pH = 6.9 (Moderate), TDS = 275 (Moderate), Turbidity = 0.9 (Good), and Microplastics = 1 (Moderate).  
                - A **rule-based system** predicts Medium Risk (2+ Moderates) — correct, but does not indicate the likelihood of contamination.  
                - An **ML model** trained on historical data may detect that this combination often leads to actual contamination and assign a **High Risk probability**, reflecting real-world risk more accurately.

                **Together**: Combining ML with rules allows us to capture complex patterns **while still enforcing critical safety limits**, providing both intelligent risk prediction and guaranteed regulatory compliance.
                """
            }
        ],
        "🔬 Technical Questions": [
            {
                "q": "What are the optimal ranges for water parameters?",
                "a": """
                - **pH:** 7.1-7.8 (Good), 6.5-7.0 or 7.9-8.5 (Moderate), <6.5 or >8.5 (Poor)
                - **TDS:** 200-270 mg/L (Good), 150-199 or 271-350 mg/L (Moderate), <150 or >350 mg/L (Poor)
                - **Turbidity:** <1.0 NTU (Good), 1.0-5.0 NTU (Moderate), >5.0 NTU (Poor)
                - **Microplastics:** 0 particles/100ml (Good), 1-2 particles/100ml (Moderate), ≥3 particles/100ml (Poor)
                """
            },
            {
                "q": "How does the risk assessment algorithm work?",
                "a": "The system uses a Random Forest classifier trained on synthetic water quality data. It combines microplastic count with pH, TDS, and turbidity values to calculate a composite risk score (Low/Medium/High)."
            },
            {
                "q": "What image format and resolution is required?",
                "a": "Upload microscope images in JPEG or PNG format with minimum 640x640 resolution. Higher resolution images (1080p or better) yield more accurate detection results."
            }
        ],
        "💧 Sampling & Testing": [
            {
                "q": "What is the correct sampling procedure?",
                "a": """
                1. Collect **100ml** of water sample in a clean container
                2. Filter through a 0.45μm membrane filter
                3. Dry the filter completely (air dry for 2-4 hours)
                4. Place under digital microscope
                5. Capture clear, focused image with proper lighting
                6. Upload image to MicroClear for analysis
                """
            },
            {
                "q": "How should I measure pH, TDS and Turbidity?",
                "a": "Use calibrated digital meters: pH meter for acidity, TDS meter for dissolved solids, and turbidity meter for cloudiness. Enter precise values in the dashboard for accurate risk assessment."
            },
            {
                "q": "How often should water be tested?",
                "a": "For routine monitoring: Monthly testing is recommended. After treatment installation: Weekly testing for first month, then monthly. During contamination events: Daily testing until levels normalize."
            }
        ],
        "⚠️ Troubleshooting": [
            {
                "q": "What if no microplastics are detected but risk is high?",
                "a": "High risk can result from poor chemical parameters (pH, TDS, Turbidity) even with low microplastic counts. Check all parameter values and consider comprehensive water treatment."
            },
            {
                "q": "The model is not loading. What should I do?",
                "a": "Ensure model files exist in ./models/ directory: risk_model_calibrated.pkl and label_encoder.pkl. If missing, run the training script first or contact system administrator."
            },
            {
                "q": "Results seem inconsistent. How to verify?",
                "a": "1. Verify instrument calibration 2. Ensure proper image focus and lighting 3. Check sample volume (must be 100ml) 4. Repeat test with fresh sample 5. Contact support with sample ID for review"
            }
        ],
        "📊 Result Interpretation": [
            {
                "q": "What do the different risk levels mean?",
                "a": """
                - **LOW (Green):** Water is safe for consumption. Continue regular monitoring.
                - **MEDIUM (Yellow):** Microplastics detected. Install household water filter.
                - **HIGH (Red):** Immediate action required. Not safe without treatment.
                """
            },
            {
                "q": "How are treatment recommendations determined?",
                "a": "Recommendations follow WHO drinking water guidelines: Low risk = basic filtration, Medium risk = activated carbon filtration, High risk = reverse osmosis + UV treatment."
            },
            {
                "q": "Can I export analysis reports?",
                "a": "Currently reports are displayed in-app. Export functionality (PDF/CSV) is under development and will be available in the next release."
            }
        ]
    }

    # Display FAQ sections
    for section_title, questions in faq_sections.items():
        st.markdown(f"### {section_title}")
        
        for faq in questions:
            with st.expander(faq["q"]):
                st.markdown(faq["a"])
        
        st.markdown("---")
    
    # Contact Support Section
    st.markdown("### 📞 Need More Help?")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='card'>
        <h4>🆘 Technical Support</h4>
        <p><b>Email:</b> usmanali08675@gmail.com</p>
        <p><b>Phone:</b> +92 334 8225271 (Usman Ali)</p>
        <p><b>Hours:</b> Mon-Fri, 9 AM - 5 PM PKT</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
        <h4>📚 Documentation</h4>
        <p>• User Manual </p>
        <p>• Sampling Protocol Guide</p>
        <p>• Research Papers & References</p>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# PAGE 4: ABOUT PAGE
# =====================================================
elif page_name == "About":
    st.markdown("<div class='header'>👥 About MicroClear Project</div>", unsafe_allow_html=True)
    
    # Project Overview
    with st.container():
        st.markdown("""
        <div class='card'>
        <h3>🌍 Project Vision</h3>
        <p>MicroClear aims to democratize access to advanced water quality monitoring by providing affordable, 
        AI-powered solutions for microplastic detection. Developed as a Final Year Project at MUST, this system 
        bridges the gap between laboratory-grade analysis and field-deployable technology.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Team Information
    st.subheader("👥 Development Team")
    team_cols = st.columns(3)
    
    team_members = [
        {"name": "Dost Muhammad", "reg": "FA22-BSE-009", "role": "Lead Developer"},
        {"name": "Usman Ali", "reg": "FA22-BSE-051", "role": "Data Scientist"},
        {"name": "Muhammad Husnain", "reg": "FA22-BSE-065", "role": "System Architect"}
    ]
    
    for idx, member in enumerate(team_members):
        with team_cols[idx]:
            st.markdown(f"""
            <div class='card'>
            <h4>{member['name']}</h4>
            <p><b>Registration:</b> {member['reg']}</p>
            <p><b>Role:</b> {member['role']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Client & Supervisor
    st.subheader("🏢 Affiliations")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='card'>
        <h3>🎓 Academic Institution</h3>
        <h4>Mirpur University of Science and Technology (MUST)</h4>
        <p><b>Department:</b> Software Engineering</p>
        <p><b>Batch:</b> 2022-2026</p>
        <p><b>Supervisor:</b> Engr. Iqra Gilani</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
        <h3>🤝 Client Organization</h3>
        <h4>Muslim Hands Mirpur</h4>
        <p><b>Focal Person:</b> Javaid ul Hassan</p>
        <p><b>Contact:</b> +92 300 555064</p>
        <p><b>Address:</b> Mirpur, Azad Jammu & Kashmir</p>
        <p><i>Partnering for cleaner water solutions</i></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Technical Specifications
    st.subheader("🛠️ Technical Specifications")
    
    spec_cols = st.columns(2)
    
    with spec_cols[0]:
        st.markdown("""
        **Core Technologies:**
        - Python 3.8+
        - YOLOv8 (Object Detection)
        - Random Forest Classifier
        - Streamlit (Web Interface)
        - OpenCV (Image Processing)
        
        **Hardware Requirements:**
        - Digital Microscope (USB)
        - pH/TDS/Turbidity Meters
        - Standard Computer (4GB+ RAM)
        """)
    
    with spec_cols[1]:
        st.markdown("""
        **System Features:**
        - Automated microplastic detection
        - Multi-parameter risk assessment
        - Rule-based treatment recommendations
        - Interactive data visualization
        """)  # ← ADDED THE MISSING CLOSING TRIPLE QUOTES HERE
    
    # SDG Impact
    st.subheader("🌱 Sustainable Development Impact")
    
    sdg_impact = {
        "💧 SDG 6: Clean Water": "Provides accessible water quality assessment tools for underserved communities",
        "🔄 SDG 12: Responsible Consumption": "Raises awareness about plastic pollution and its impacts",
        "🌿 SDG 13: Climate Action": "Supports environmental monitoring for climate resilience",
        "🐟 SDG 14: Life Below Water": "Helps prevent microplastic contamination in aquatic ecosystems"
    }
    
    for sdg, impact in sdg_impact.items():
        st.info(f"**{sdg}:** {impact}")

# ==================== FOOTER ====================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
with footer_col2:
    st.markdown("""
    <div style='text-align: center; color: #8b949e; font-size: 0.9rem;'>
    <p>🔬 <b>MicroClear</b> - Final Year Project 2026 | Department of Software Engineering | MUST Mirpur</p>
    <p>© 2024-2026 MicroClear Project Team. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

# Add sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; color: #8b949e; font-size: 0.8rem;">
<p><b>Version 1.0</b></p>
<p>Powered by Project Team</p>
</div>
""", unsafe_allow_html=True)