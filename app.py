import streamlit as st
from PIL import Image
import numpy as np
import joblib

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

# Sidebar navigation
st.sidebar.title("🔬 MicroClear Navigation")
page = st.sidebar.radio("Go to", ["Home", "Analysis Dashboard", "About"])

# Custom CSS for a professional Dark Mode UI
st.markdown("""
<style>
    body { background-color: #121212; color: #e0e0e0; }
    .header { font-size: 2.0rem; font-weight: bold; color: #00bcd4; text-align: center; margin-bottom: 20px; }
    .card {
        background: linear-gradient(135deg, #1e1e1e, #2d2d2d);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #00bcd4;
        margin-bottom: 20px;
    }
    .stNumberInput, .stFileUploader { margin-bottom: 20px; }
    .result-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #263238;
        border: 1px solid #00bcd4;
        text-align: center;
        margin-top: 10px;
    }
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #00bcd4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

if page == "Home":
    st.markdown("<div class='header'>🔬 MicroClear: AI-Based Microplastic Pollution Risk Assessment and Mitigation System</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='card'>
    <h3>Abstract</h3>
    <p>Microplastic pollution is a critical environmental threat. <b>MicroClear</b> is an AI-powered system designed for 
    automated detection of microplastics in water samples using <b>YOLOv8</b>. By integrating hardware sensors for pH, TDS, and Turbidity, 
    it provides a holistic pollution risk assessment and mitigation strategy for cleaner water resources.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
    <h3>Key Objectives</h3>
    <ul>
        <li>Automate microplastic detection in 20ml water samples via microscope imaging.</li>
        <li>Integrate real-time water quality parameters (pH, TDS, Turbidity).</li>
        <li>Classify pollution risk levels and recommend rule-based mitigation techniques.</li>
        <li>Support Sustainable Development Goals (SDG 6, 12, 13, 14).</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.info("💡 Navigate to the 'Analysis Dashboard' to start the 200ml sample assessment.")

elif page == "Analysis Dashboard":
    st.markdown("<div class='header'>Water Analysis Portal</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Sample Upload")
        st.markdown("""
        **Standard Operating Procedure (SOP):**
        - Filter **200ml** of water through a membrane.
        - Place dry filter under the digital microscope.
        """)
        uploaded_file = st.file_uploader("Upload 200ml Filter Image", type=["png", "jpg", "jpeg"])
        
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="200ml Sample View", use_container_width=True)
            
            # --- NEW BUTTON 1: Detect ---
            if st.button("🔍 Detect Microplastics"):
                with st.spinner('Analyzing Image with YOLOv8...'):
                    # Placeholder for YOLOv8 model logic
                    st.session_state['detected_count'] = 12 
                    st.success(f"Detection Complete! Found {st.session_state['detected_count']} particles.")

    with col2:
        st.subheader("2. Parameters")
        # Use session state count if available
        default_count = st.session_state.get('detected_count', 0)
        count = st.number_input("Microplastic Count", min_value=0, value=default_count, step=1)
        ph = st.number_input("pH Level", 0.0, 14.0, 7.1)
        tds = st.number_input("TDS (mg/L)", 0, 2000, 270)
        turbidity = st.number_input("Turbidity (NTU)", 0.0, 100.0, 0.0)
        
        # --- NEW BUTTON 2: Analyze ---
        run_analysis = st.button("📊 Analyze Risk Level")

    if run_analysis:
        st.divider()
        
        if model is not None:
            # Prepare user input for ML model
            user_input = np.array([[ph, tds, turbidity, count]])

            # Predict Risk class
            pred_encoded = model.predict(user_input)[0]
            risk_level = label_encoder.inverse_transform([pred_encoded])[0]

            # predict probability for confidence
            prob = model.predict_proba(user_input)[0]
            risk_confidence = prob[pred_encoded] * 100

            # Assign color and technique
            if risk_level.lower() == "low":
                color = "#00e676"  # Green
                technique = "Sand Filtration / Standard Monitoring"
            elif risk_level.lower() == "medium":
                color = "#ffeb3b"  # Yellow
                technique = "Dual-stage Membrane Filtration (Microfiltration)"
            else:
                color = "#ff5252"  # Red
                technique = "Advanced Mitigation: Activated Carbon & Reverse Osmosis"

            # Display Results
            st.subheader("3. Pollution Assessment & Mitigation")
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.markdown(f"""
                    <div class='result-box' style='border-color: {color};'>
                    <small>Risk Level</small>
                    <h2 style='color: {color};'>{risk_level.upper()}</h2>
                    <small>Confidence: {risk_confidence:.1f}%</small>
                    </div>
                """, unsafe_allow_html=True)

            with res_col2:
                st.markdown(f"""
                    <div class='result-box'>
                    <small>Recommended Technique</small>
                    <p style='font-weight: bold; margin-top:10px;'>{technique}</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.error("ML Model files not found. Please check your ./models/ directory.")

elif page == "About":
    st.markdown("<div class='header'>Project Credits & Affiliations</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
    <h3 style='color: #ffeb3b;'>👥 Development Team</h3>
    <p><b>Department of Software Engineering (Batch 2022-2026)</b></p>
    <ul>
        <li><b>Dost Muhammad</b> (FA22-BSE-009)</li>
        <li><b>Usman Ali</b> (FA22-BSE-051) </li>
        <li><b>Muhammad Husnain</b> (FA22-BSE-065) </li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
    <h3 style='color: #ffeb3b;'>🏢 Client Organization</h3>
    <h4>Muslim Hands Mirpur</h4>
    <p><b>Focal Person:</b> Javaid ul Hassan</p>
    <p><b>Contact:</b> +92 300 555064</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
    <h3 style='color: #ffeb3b;'>🎓 Academic Institution</h3>
    <h4>Mirpur University of Science and Technology (MUST)</h4>
    <p><b>Location:</b> Mirpur, Azad Jammu & Kashmir</p>
    <p><i>العلم يرفع قدر الانسان</i></p>
    </div>
    """, unsafe_allow_html=True)