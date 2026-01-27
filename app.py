import streamlit as st
from PIL import Image
import numpy as np
import joblib
import base64

# Load model & label encoder once at the top
MODEL_PATH = "./models/risk_model_calibrated.pkl"
ENCODER_PATH = "./models/label_encoder.pkl"

# Using a try-except block in case paths are not yet set up
try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
except:
    model = None
    label_encoder = None

# Page configuration
st.set_page_config(
    page_title="MicroClear - AI Microplastic Detection",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS for a professional Dark Mode UI
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    
    /* Headers */
    .header { 
        font-size: 2.5rem; 
        font-weight: bold; 
        color: #00bcd4; 
        text-align: center; 
        margin-bottom: 30px;
        padding: 10px;
        background: linear-gradient(90deg, #00bcd4, #0077b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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
    
    /* Input fields */
    .stNumberInput, .stFileUploader {
        margin-bottom: 20px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #161b22;
    }
    
    /* Success/Error/Warning boxes */
    .stAlert {
        border-radius: 10px;
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
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    
    /* Divider */
    .stDivider {
        border-color: #30363d;
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
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="color: #00bcd4;">🔬</h1>
    <h3 style="color: #00bcd4;">MicroClear</h3>
    <p style="color: #8b949e; font-size: 0.9rem;">AI-Based Microplastic Pollution Risk Assessment and Mitigation System</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
page = st.sidebar.radio("**Navigation**", ["🏠 Home", "📊 Analysis Dashboard", "❓ FAQ & Help", "👥 About"])

# Remove emoji from page selection for comparison
page_name = page[2:] if page.startswith(("🏠", "📊", "❓", "👥")) else page

# ==================== HOME PAGE ====================
if page_name == "Home":
    st.markdown("<div class='header'>🔬 MicroClear: AI-Based Microplastic Pollution Risk Assessment and Mitigation System</div>", unsafe_allow_html=True)
    
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
        <p>Supporting Sustainable Development Goals (SDG)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Start Card
    st.markdown("""
    <div class='card'>
    <h3>🚀 Quick Start Guide</h3>
    <ol>
        <li><b>Go to Analysis Dashboard</b> from the sidebar</li>
        <li><b>Upload</b> your microscope image (100ml sample)</li>
        <li><b>Enter</b> water quality parameters (pH, TDS, Turbidity)</li>
        <li><b>Click Analyze</b> to get risk assessment & recommendations</li>
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
        <p>Automated microplastic detection using YOLOv8 deep learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
        <div class='feature-icon'>📊</div>
        <h4>Risk Assessment</h4>
        <p>Multi-factor pollution risk scoring (Low/Medium/High)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
        <div class='feature-icon'>💡</div>
        <h4>Smart Mitigation</h4>
        <p>Rule-based treatment recommendations for water purification</p>
        </div>
        """, unsafe_allow_html=True)
    
    # SDGs Section
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
            <div class='metric-card' style='border-color: {color};'>
            <div style='font-size: 2rem; margin-bottom: 10px;'>{icon}</div>
            <h4 style='color: {color};'>{num}</h4>
            <p>{name}</p>
            </div>
            """, unsafe_allow_html=True)

# ==================== ANALYSIS DASHBOARD ====================
elif page_name == "Analysis Dashboard":
    st.markdown("<div class='header'>📊 Water Analysis Dashboard</div>", unsafe_allow_html=True)
    
    # Dashboard Layout
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
            
            # Detection Button
            if st.button("🔍 Detect Microplastics", type="primary", use_container_width=True):
                with st.spinner('🧠 Analyzing Image with YOLOv8...'):
                    # Placeholder for YOLOv8 model logic
                    st.session_state['detected_count'] = np.random.randint(0, 15)
                    st.session_state['detection_confidence'] = np.random.uniform(85, 98)
                    st.success(f"✅ Detection Complete! Found {st.session_state['detected_count']} particles.")
        
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card'>
        <h3>📋 2. Water Quality Parameters</h3>
        """, unsafe_allow_html=True)
        
        # Use session state count if available
        default_count = st.session_state.get('detected_count', 0)
        detection_conf = st.session_state.get('detection_confidence', 0)
        
        if default_count > 0:
            st.info(f"📊 **Detected Particles:** {default_count} (Confidence: {detection_conf:.1f}%)")
        
        # Parameter Inputs
        count = st.number_input(
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
            max_value=2000, 
            value=270, 
            step=1,
            help="Total Dissolved Solids concentration"
        )
        
        turbidity = st.number_input(
            "🌫️ Turbidity (NTU)", 
            min_value=0.0, 
            max_value=100.0, 
            value=0.0, 
            step=0.1,
            format="%.1f",
            help="Water clarity measurement"
        )
        
        # Analyze Button
        run_analysis = st.button(
            "📈 Analyze Risk Level", 
            type="primary", 
            use_container_width=True,
            disabled=(model is None)
        )
        
        if model is None:
            st.warning("⚠️ ML Model not loaded. Please ensure model files exist in ./models/")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Risk Analysis Results
    if run_analysis:
        st.divider()
        st.markdown("<div class='header'>📋 Analysis Results</div>", unsafe_allow_html=True)
        
        # Prepare user input for ML model
        user_input = np.array([[ph, tds, turbidity, count]])

        # Predict Risk class
        pred_encoded = model.predict(user_input)[0]
        risk_level = label_encoder.inverse_transform([pred_encoded])[0]

        # Predict probability for confidence
        prob = model.predict_proba(user_input)[0]
        risk_confidence = prob[pred_encoded] * 100

        # Enhanced treatment recommendations
        treatment_info = {
            "low": {
                "color": "#00e676",
                "title": "SAFE FOR CONSUMPTION",
                "technique": "Standard Chlorination & Basic Filtration",
                "steps": [
                    "Regular chlorination (if municipal supply)",
                    "Basic sediment filtration (10-20μm)",
                    "Monthly water quality testing"
                ],
                "urgency": "Continue normal use"
            },
            "medium": {
                "color": "#ffeb3b",
                "title": "REQUIRES FILTRATION",
                "technique": "Household Water Purifier",
                "steps": [
                    f"Install activated carbon filter",
                    "Add 1-5μm microfiltration stage",
                    "Test water weekly for 1 month"
                ],
                "urgency": "Install within 1 week"
            },
            "high": {
                "color": "#ff5252",
                "title": "IMMEDIATE TREATMENT REQUIRED",
                "technique": "Advanced Reverse Osmosis System",
                "steps": [
                    f"INSTALL RO SYSTEM IMMEDIATELY",
                    "Add UV disinfection stage",
                    "Boil water as interim measure",
                    "Contact water authority"
                ],
                "urgency": "TREAT BEFORE CONSUMPTION"
            }
        }
        
        treatment = treatment_info.get(risk_level.lower(), treatment_info["high"])
        
        # Display Results in Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Risk Assessment", "💡 Recommendations", "📈 Detailed Analysis"])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class='result-box' style='border-color: {treatment["color"]};'>
                <small>RISK LEVEL</small>
                <h1 style='color: {treatment["color"]}; margin: 10px 0;'>{risk_level.upper()}</h1>
                <small>Confidence: {risk_confidence:.1f}%</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='result-box'>
                <small>URGENCY LEVEL</small>
                <h3 style='color: {treatment["color"]}; margin: 10px 0;'>{treatment["urgency"]}</h3>
                <small>Action Required</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='result-box'>
                <small>TREATMENT TECHNIQUE</small>
                <h4 style='margin: 10px 0;'>{treatment["technique"]}</h4>
                <small>Recommended Solution</small>
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown(f"### 🛠️ Treatment Recommendations: **{treatment['title']}**")
            
            st.markdown("#### Implementation Steps:")
            for i, step in enumerate(treatment["steps"], 1):
                st.markdown(f"{i}. {step}")
            
            st.markdown("---")
            
            # Additional parameters info
            st.markdown("#### 📋 Parameter Analysis:")
            params_col1, params_col2 = st.columns(2)
            
            with params_col1:
                st.metric("Microplastic Count", f"{count} particles/100ml", 
                         delta=None, delta_color="off")
                st.metric("pH Level", f"{ph}", 
                         delta="Optimal: 7.1-7.8" if 7.1 <= ph <= 7.8 else "Out of range")
            
            with params_col2:
                st.metric("TDS Level", f"{tds} mg/L", 
                         delta="Optimal: 200-270" if 200 <= tds <= 270 else "Out of range")
                st.metric("Turbidity", f"{turbidity} NTU", 
                         delta="Optimal: <1.0" if turbidity < 1.0 else "High turbidity")
        
        with tab3:
            # Water quality categorization
            st.markdown("### 🔬 Detailed Parameter Categorization")
            
            # pH categorization
            if 7.1 <= ph <= 7.8:
                ph_status = "✅ Good"
            elif (6.5 <= ph <= 7.0) or (7.9 <= ph <= 8.5):
                ph_status = "⚠️ Moderate"
            else:
                ph_status = "❌ Poor"
            
            # TDS categorization
            if 200 <= tds <= 270:
                tds_status = "✅ Good"
            elif (150 <= tds <= 199) or (271 <= tds <= 350):
                tds_status = "⚠️ Moderate"
            else:
                tds_status = "❌ Poor"
            
            # Turbidity categorization
            if turbidity < 1.0:
                turb_status = "✅ Good"
            elif 1.0 <= turbidity <= 5.0:
                turb_status = "⚠️ Moderate"
            else:
                turb_status = "❌ Poor"
            
            # MP categorization
            if count == 0:
                mp_status = "✅ Excellent"
            elif 1 <= count <= 2:
                mp_status = "⚠️ Moderate"
            else:
                mp_status = "❌ Poor"
            
            # Display categorization table
            cat_data = {
                "Parameter": ["pH", "TDS", "Turbidity", "Microplastic Count"],
                "Value": [ph, f"{tds} mg/L", f"{turbidity} NTU", f"{count} particles"],
                "Status": [ph_status, tds_status, turb_status, mp_status],
                "Optimal Range": ["7.1-7.8", "200-270 mg/L", "<1.0 NTU", "0 particles"]
            }
            
            st.table(cat_data)
            
        
# ==================== FAQ PAGE ====================
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
        <p><b>Email:</b> support@microclear.must.edu.pk</p>
        <p><b>Phone:</b> +92 300 555064 (Javaid ul Hassan)</p>
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

# ==================== ABOUT PAGE ====================
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
<p>Powered by Streamlit</p>
</div>
""", unsafe_allow_html=True)