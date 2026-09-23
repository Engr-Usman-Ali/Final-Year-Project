# =====================================================
# File: app.py
# MicroClear - AI-Based Microplastic Pollution Risk
# Assessment and Mitigation System
# Streamlit Web Application
# =====================================================

import os

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# =====================================================
# PAGE CONFIG  (must be the FIRST Streamlit command)
# =====================================================

st.set_page_config(
    page_title="MicroClear | AI-Based Microplastic Pollution Risk Assessment",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================
# LOAD BOTH MODELS (YOLO + ML) - cached so they load once
# =====================================================


@st.cache_resource(show_spinner=False)
def load_models():
    """Load YOLO and ML models once and reuse them across reruns."""

    try:
        # Using raw string or forward slashes for cross-platform compatibility
        yolo_model = YOLO("models/microplastic_yolo_roboflow_640_run2/best.pt")
        yolo_status = True
    except Exception:
        yolo_model = None
        yolo_status = False

    MODEL_PATH = "models/result_randomforest/risk_model.pkl"
    if os.path.exists(MODEL_PATH):
        ml_model = joblib.load(MODEL_PATH)
        ml_status = True
    else:
        ml_model = None
        ml_status = False

    return yolo_model, ml_model, yolo_status, ml_status


yolo_model, ml_model, yolo_ok, ml_ok = load_models()

if "models_notified" not in st.session_state:
    if yolo_ok and ml_ok:
        st.toast("✅ YOLO + ML models loaded successfully")
    elif yolo_ok and not ml_ok:
        st.toast("⚠️ YOLO loaded, ML model missing")
    elif not yolo_ok and ml_ok:
        st.toast("⚠️ ML loaded, YOLO model missing")
    else:
        st.toast("❌ Both models failed to load")
    st.session_state.models_notified = True


# =====================================================
# HELPER: render HTML safely inside st.markdown
# =====================================================

def html(block: str):
    cleaned = "\n".join(line.strip() for line in block.strip().splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


# =====================================================
# PARAMETER CATEGORIZATION (WHO / PCRWR)
# =====================================================

def categorize_parameter(value, param):

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
# 6.9 VISUALIZATION AND CHART RENDERING (NFR-05)
# =====================================================

def plot_parameter_status(param_name, value, category):
    """
    Generates a horizontal bar chart showing the status of a single parameter.
    Matches the 'Good', 'Moderate', 'Poor' color scheme.
    Never saves to disk (Data Privacy NFR-05).
    """
    colors = {
        'Good': '#3DDC97',      # Matches --good in CSS
        'Moderate': '#FFC24B',  # Matches --mod in CSS
        'Poor': '#FF6B6B'       # Matches --poor in CSS
    }
    
    # 1. Set dark theme for this specific figure
    plt.style.use('dark_background')
    
    # 2. Create figure with transparent background
    fig, ax = plt.subplots(figsize=(5, 0.8))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    # 3. Plot the bar
    ax.barh([param_name], [1], color=colors.get(category, '#93AEBB'))
    
    # 4. Styling
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 5. Title with white text and proper padding
    ax.set_title(f'{param_name}: {value} ({category})', fontsize=10, loc='left', pad=10, color='#E4F1F5')
    
    # 6. Remove borders
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    plt.tight_layout()
    return fig


# =====================================================
# ML PREDICTION
# =====================================================

def ml_predict(ph, tds, turbidity, mp_count):

    if ml_model is None:
        return "Unknown", 0.0

    input_data = pd.DataFrame([{
        "pH": ph,
        "TDS": tds,
        "Turbidity": turbidity,
        "MP_Count": mp_count,
    }])

    prediction = ml_model.predict(input_data)[0]

    if hasattr(ml_model, "predict_proba"):
        probs = ml_model.predict_proba(input_data)[0]
        confidence = max(probs) * 100
    else:
        confidence = 0.0

    return prediction, confidence


# =====================================================
# RULE BASED SAFETY SYSTEM
# =====================================================

def rule_based_risk(ph, tds, turbidity, mp_count):

    pH_cat = categorize_parameter(ph, "pH")
    tds_cat = categorize_parameter(tds, "TDS")
    turbidity_cat = categorize_parameter(turbidity, "Turbidity")
    mp_cat = categorize_parameter(mp_count, "MP_Count")

    categories = [pH_cat, tds_cat, turbidity_cat, mp_cat]

    if "Poor" in categories:
        return "High"

    if categories.count("Moderate") >= 2:
        return "Medium"

    return "Low"


# =====================================================
# FINAL RISK ENGINE (ML-first + safety override)
# =====================================================

def final_risk_engine(ph, tds, turbidity, mp_count):

    ml_risk, ml_conf = ml_predict(ph, tds, turbidity, mp_count)
    rule_risk = rule_based_risk(ph, tds, turbidity, mp_count)

    override_applied = False

    if ml_risk == "Unknown":
        final_risk = rule_risk

    elif ml_risk == rule_risk:
        final_risk = ml_risk

    elif rule_risk == "Medium" and ml_risk == "Low":
        final_risk = "Medium"
        override_applied = True

    elif rule_risk == "High" and ml_risk in ["Low", "Medium"]:
        final_risk = "High"
        override_applied = True

    else:
        final_risk = ml_risk

    return {
        "final_risk": final_risk,
        "ml_risk": ml_risk,
        "ml_confidence": ml_conf,
        "rule_risk": rule_risk,
        "override_applied": override_applied,
        "categories": {
            "pH": categorize_parameter(ph, "pH"),
            "TDS": categorize_parameter(tds, "TDS"),
            "Turbidity": categorize_parameter(turbidity, "Turbidity"),
            "MP_Count": categorize_parameter(mp_count, "MP_Count"),
        },
    }


# =====================================================
# GLOBAL STYLING
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Nunito+Sans:wght@400;600;700&display=swap');

:root {
    --bg: #061621;
    --bg2: #041019;
    --surface: #0F2E3E;
    --surface2: #0B2433;
    --line: #1E4759;
    --text: #E4F1F5;
    --muted: #93AEBB;
    --aqua: #2ED3C3;
    --aqua-bright: #5CE6D6;
    --aqua-deep: #0E8FA2;
    --sky: #4CC9F0;
    --amber: #FFC24B;
    --good: #3DDC97;
    --mod: #FFC24B;
    --poor: #FF6B6B;
}

/* ---------- Base ---------- */
.stApp {
    background:
        radial-gradient(1300px 620px at 85% -12%, rgba(46, 211, 195, 0.16), transparent 60%),
        radial-gradient(900px 520px at -12% 18%, rgba(76, 201, 240, 0.13), transparent 55%),
        radial-gradient(900px 700px at 55% 118%, rgba(14, 143, 162, 0.16), transparent 60%),
        linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%);
    background-attachment: fixed;
    color: var(--text);
    font-family: 'Nunito Sans', 'Segoe UI', sans-serif;
}
h1, h2, h3, h4 { font-family: 'Sora', 'Segoe UI', sans-serif !important; letter-spacing: -0.01em; }

/* ---------- Remove sidebar + Streamlit chrome ---------- */
section[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
#MainMenu, footer { display: none !important; }

.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 1200px;
    padding-top: 6.8rem;
    padding-bottom: 3rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* =====================================================
   TOP NAVBAR
   ===================================================== */
div[data-testid="stHorizontalBlock"]:has(.mc-nav-marker) {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    box-sizing: border-box;
    z-index: 100000;
    padding: 12px max(2rem, calc((100vw - 1200px) / 2 + 2rem));
    align-items: center;
    gap: 1rem;
    border-radius: 0;
    background: rgba(6, 22, 33, 0.92);
    border-bottom: 1px solid var(--line);
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.38);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
}
div[data-testid="stHorizontalBlock"]:has(.mc-nav-marker) {
    justify-content: space-between !important;
    flex-wrap: nowrap !important;
}
div[data-testid="stHorizontalBlock"]:has(.mc-nav-marker) > div:first-child {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
}
div[data-testid="stHorizontalBlock"]:has(.mc-nav-marker) > div:last-child {
    flex: 0 1 auto !important;
    width: auto !important;
    min-width: 0 !important;
}
.mc-nav-brand { display: flex; align-items: center; gap: 12px; }
.mc-nav-logo {
    flex: 0 0 40px; width: 40px; height: 40px; border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.35rem; line-height: 1;
    background: linear-gradient(145deg, var(--aqua), var(--aqua-deep));
    box-shadow: 0 6px 18px rgba(46, 211, 195, 0.32);
}
.mc-nav-title { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.15rem; color: #fff; line-height: 1.1; }
.mc-nav-sub { font-size: 0.72rem; color: var(--muted); margin-top: 3px; }

div[data-testid="stHorizontalBlock"]:has(.mc-nav-marker) div[role="radiogroup"] {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    justify-content: flex-end;
    align-items: center;
}
div[data-testid="stHorizontalBlock"]:has(.mc-nav-marker) div[role="radiogroup"] > label {
    margin: 0;
    padding: 12px 16px 10px 16px;
    border-radius: 8px 8px 0 0;
    border-bottom: 3px solid transparent;
    cursor: pointer;
    background: transparent;
    transition: background 0.18s ease, border-color 0.18s ease;
}
div[data-testid="stHorizontalBlock"]:has(.mc-nav-marker) div[role="radiogroup"] > label > div:first-child { display: none; }
div[data-testid="stHorizontalBlock"]:has(.mc-nav-marker) div[role="radiogroup"] > label p {
    font-weight: 700; font-size: 0.95rem; color: #B7CCD5; white-space: nowrap; margin: 0;
    transition: color 0.18s ease;
}
div[data-testid="stHorizontalBlock"]:has(.mc-nav-marker) div[role="radiogroup"] > label:hover {
    background: rgba(46, 211, 195, 0.07);
}
div[data-testid="stHorizontalBlock"]:has(.mc-nav-marker) div[role="radiogroup"] > label:hover p { color: #fff; }
div[data-testid="stHorizontalBlock"]:has(.mc-nav-marker) div[role="radiogroup"] > label:has(input:checked) {
    border-bottom-color: var(--aqua);
    background: linear-gradient(180deg, transparent, rgba(46, 211, 195, 0.12));
}
div[data-testid="stHorizontalBlock"]:has(.mc-nav-marker) div[role="radiogroup"] > label:has(input:checked) p { color: var(--aqua-bright); }

@media (max-width: 640px) {
    div[data-testid="stHorizontalBlock"]:has(.mc-nav-marker) {
        padding: 8px 1rem;
        gap: 0.2rem;
        flex-wrap: wrap !important;
        justify-content: center !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.mc-nav-marker) > div:first-child,
    div[data-testid="stHorizontalBlock"]:has(.mc-nav-marker) > div:last-child {
        flex: 1 1 100% !important;
        width: 100% !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.mc-nav-marker) div[role="radiogroup"] { justify-content: center; gap: 0; }
    div[data-testid="stHorizontalBlock"]:has(.mc-nav-marker) div[role="radiogroup"] > label { padding: 8px 10px 6px 10px; }
    div[data-testid="stHorizontalBlock"]:has(.mc-nav-marker) div[role="radiogroup"] > label p { font-size: 0.85rem; }
    .mc-nav-sub { display: none; }
    .block-container, [data-testid="stMainBlockContainer"] { padding-top: 9.5rem !important; }
}

/* ---------- Inner page header ---------- */
.header {
    font-family: 'Sora', sans-serif;
    font-size: 1.95rem;
    font-weight: 700;
    margin: 0 0 22px 0;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--line);
    position: relative;
    color: #FFFFFF;
    letter-spacing: -0.015em;
}
.header::after {
    content: "";
    position: absolute; left: 0; bottom: -1px;
    width: 110px; height: 3px; border-radius: 3px;
    background: linear-gradient(90deg, var(--aqua), var(--sky), transparent);
}

/* ---------- Buttons ---------- */
.stButton > button {
    width: 100%;
    border-radius: 10px;
    font-weight: 700;
    padding: 0.64rem 1.2rem;
    background: rgba(15, 46, 62, 0.6);
    color: var(--text);
    border: 1px solid var(--line);
    transition: all 0.18s ease;
}
.stButton > button p { color: inherit; font-weight: 700; }
.stButton > button:hover {
    border-color: var(--aqua);
    color: var(--aqua-bright);
    box-shadow: 0 0 0 1px rgba(46, 211, 195, 0.25), 0 8px 22px rgba(46, 211, 195, 0.12);
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, var(--aqua), var(--aqua-deep));
    color: #04222A;
    border: none;
    box-shadow: 0 6px 18px rgba(46, 211, 195, 0.25);
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, var(--aqua-bright), var(--aqua));
    color: #04222A;
    box-shadow: 0 10px 30px rgba(46, 211, 195, 0.42);
    transform: translateY(-1px);
}
.stButton > button:disabled { opacity: 0.45; }

/* ---------- Streamlit components ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--line);
    border-radius: 14px;
    background: linear-gradient(160deg, rgba(15, 46, 62, 0.85), rgba(11, 36, 51, 0.9));
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(46, 211, 195, 0.45);
    box-shadow: 0 12px 34px rgba(46, 211, 195, 0.08);
}
[data-testid="stMetric"] {
    background: linear-gradient(160deg, var(--surface), var(--surface2));
    border: 1px solid var(--line);
    padding: 16px 18px;
    border-radius: 12px;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
[data-testid="stMetric"]:hover { border-color: rgba(46, 211, 195, 0.5); transform: translateY(-2px); }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: var(--surface2); }

.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-weight: 700; padding: 10px 14px; border-radius: 8px 8px 0 0; color: #B7CCD5; }
.stTabs [data-baseweb="tab"]:hover { color: var(--aqua); }
.stTabs [aria-selected="true"] { color: var(--aqua) !important; background: rgba(46, 211, 195, 0.07); }
.stTabs [data-baseweb="tab-highlight"] {
    background: linear-gradient(90deg, var(--aqua), var(--sky)) !important;
    height: 3px; border-radius: 3px;
}

/* ---------- Dashboard: progress stepper ---------- */
.mc-stepper { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 0 0 22px 0; }
.mc-st {
    display: flex; align-items: center; gap: 12px;
    padding: 12px 16px; border-radius: 12px;
    background: var(--surface2); border: 1px solid var(--line);
    transition: all 0.2s ease;
}
.mc-st-n {
    flex: 0 0 30px; width: 30px; height: 30px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Sora', sans-serif; font-weight: 700; font-size: 0.88rem;
    background: rgba(147, 174, 187, 0.15); color: var(--muted);
}
.mc-st-t { font-weight: 700; font-size: 0.93rem; color: var(--muted); }
.mc-st.done { border-color: rgba(61, 220, 151, 0.45); }
.mc-st.done .mc-st-n { background: var(--good); color: #04222A; }
.mc-st.done .mc-st-t { color: #CFE0E6; }
.mc-st.active { border-color: var(--aqua); background: rgba(46, 211, 195, 0.10); box-shadow: 0 0 0 3px rgba(46, 211, 195, 0.12); }
.mc-st.active .mc-st-n { background: linear-gradient(135deg, var(--aqua), var(--aqua-deep)); color: #04222A; }
.mc-st.active .mc-st-t { color: #fff; }

/* ---------- Results ---------- */
.result-box {
    padding: 24px 20px;
    border-radius: 14px;
    background: linear-gradient(160deg, var(--surface), var(--surface2));
    border: 1px solid var(--line);
    border-top-width: 4px;
    text-align: center;
    margin-top: 12px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.32);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.result-box:hover { transform: translateY(-3px); box-shadow: 0 14px 38px rgba(0, 0, 0, 0.45); }

/* ---------- Home: hero ---------- */
.mc-brand { display: flex; align-items: center; gap: 14px; margin-bottom: 10px; width: 100%; }
.mc-logo {
    flex: 0 0 52px; width: 52px; height: 52px; border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.75rem; line-height: 1;
    background: linear-gradient(145deg, var(--aqua), var(--aqua-deep));
    box-shadow: 0 8px 24px rgba(46, 211, 195, 0.32);
}
.mc-brand-text { display: flex; flex-direction: column; justify-content: center; min-width: 0; flex: 1 1 auto; }
.mc-wordmark { font-family: 'Sora', sans-serif; font-size: 1.5rem; font-weight: 700; color: #fff; line-height: 1.15; letter-spacing: -0.01em; }
.mc-tagline { font-size: 0.85rem; color: var(--muted); margin-top: 4px; line-height: 1.4; }
.mc-title {
    font-family: 'Sora', sans-serif;
    font-size: clamp(1.9rem, 3.3vw, 2.75rem);
    font-weight: 700; line-height: 1.14; letter-spacing: -0.02em;
    margin: 18px 0 16px 0;
    background: linear-gradient(90deg, #FFFFFF 0%, #B9EBE5 55%, var(--aqua) 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.mc-intro { font-size: 1.08rem; line-height: 1.65; color: #B7CCD5; max-width: 58ch; margin: 0 0 14px 0; }
.mc-official { font-size: 0.9rem; color: var(--muted); max-width: 62ch; margin: 0 0 20px 0; line-height: 1.55; }

.mc-report {
    background: linear-gradient(165deg, #123A4C, #0B2433);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 22px 22px 18px 22px;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.36), 0 0 0 1px rgba(46, 211, 195, 0.12);
}
.mc-report-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.mc-report-title { font-family: 'Sora', sans-serif; font-weight: 600; font-size: 1rem; color: #fff; }
.mc-row { display: flex; justify-content: space-between; align-items: center; padding: 11px 0; border-top: 1px solid rgba(30, 71, 89, 0.9); font-size: 0.97rem; }
.mc-row .k { color: var(--muted); }
.mc-row .v { font-weight: 700; color: #fff; margin-right: 10px; }
.mc-row .r { display: flex; align-items: center; }
.mc-action {
    margin-top: 10px; padding: 12px 14px; border-radius: 10px;
    background: linear-gradient(135deg, rgba(255, 194, 75, 0.10), rgba(232, 149, 47, 0.06));
    border: 1px solid rgba(255, 194, 75, 0.35);
    font-size: 0.9rem; line-height: 1.5; color: #F5DBA0;
}
.mc-note { font-size: 0.78rem; color: var(--muted); margin-top: 10px; }

.pill { display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; border: 1px solid transparent; }
.pill-good { background: rgba(61, 220, 151, 0.15); color: var(--good); border-color: rgba(61, 220, 151, 0.38); }
.pill-mod  { background: rgba(255, 194, 75, 0.15); color: var(--mod);  border-color: rgba(255, 194, 75, 0.38); }
.pill-poor { background: rgba(255, 107, 107, 0.15); color: var(--poor); border-color: rgba(255, 107, 107, 0.38); }

.mc-strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 34px 0 6px 0; }
.mc-strip div {
    padding: 16px; border-radius: 12px;
    background: linear-gradient(160deg, var(--surface), var(--surface2));
    border: 1px solid var(--line);
    font-weight: 600; font-size: 0.95rem; text-align: center;
    transition: border-color 0.2s ease, transform 0.2s ease, color 0.2s ease, box-shadow 0.2s ease;
}
.mc-strip div:hover { border-color: var(--aqua); color: var(--aqua); transform: translateY(-3px); box-shadow: 0 10px 26px rgba(46, 211, 195, 0.14); }

.mc-h2 { font-family: 'Sora', sans-serif; font-size: 1.55rem; font-weight: 700; color: #fff; margin: 54px 0 8px 0; }
.mc-lead { color: var(--muted); font-size: 1.02rem; line-height: 1.6; max-width: 66ch; margin: 0 0 22px 0; }

.mc-why { display: grid; grid-template-columns: repeat(3, 1fr); gap: 26px; }
.mc-why > div { border-left: 3px solid var(--aqua); padding: 2px 0 2px 16px; }
.mc-why b { display: block; font-family: 'Sora', sans-serif; font-size: 1.02rem; margin-bottom: 6px; color: #fff; }
.mc-why span { color: #B7CCD5; line-height: 1.6; font-size: 0.97rem; }
.mc-bottomline {
    margin-top: 24px; padding: 16px 20px; border-radius: 12px;
    background: linear-gradient(135deg, rgba(46, 211, 195, 0.10), rgba(76, 201, 240, 0.05));
    border: 1px solid rgba(46, 211, 195, 0.35);
    font-size: 1.02rem; line-height: 1.6;
}

.mc-steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.mc-step {
    padding: 18px; border-radius: 14px;
    background: linear-gradient(160deg, var(--surface), var(--surface2));
    border: 1px solid var(--line);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.mc-step:hover { transform: translateY(-3px); border-color: rgba(46, 211, 195, 0.5); box-shadow: 0 12px 30px rgba(46, 211, 195, 0.12); }
.mc-step .n {
    width: 32px; height: 32px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, var(--aqua), var(--aqua-deep)); color: #04222A;
    font-family: 'Sora', sans-serif; font-weight: 700; font-size: 0.92rem; margin-bottom: 10px;
}
.mc-step b { display: block; font-size: 1.02rem; margin-bottom: 4px; color: #fff; }
.mc-step span { color: #B7CCD5; font-size: 0.93rem; line-height: 1.55; }

/* ---------- Home: detector illustration ---------- */
.mc-detect { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; align-items: center; }
.mc-detect-copy ul { list-style: none; padding: 0; margin: 0; }
.mc-detect-copy li { padding: 4px 0 4px 16px; border-left: 3px solid var(--aqua); margin-bottom: 16px; color: #B7CCD5; line-height: 1.55; font-size: 0.98rem; }
.mc-detect-copy li b { display: block; color: #fff; font-family: 'Sora', sans-serif; font-size: 1rem; margin-bottom: 2px; }
.mc-scope-wrap { display: flex; flex-direction: column; align-items: center; gap: 12px; }
.mc-scope {
    position: relative; width: 100%; max-width: 360px; aspect-ratio: 1 / 1;
    border-radius: 50%; overflow: hidden;
    background: radial-gradient(circle at 34% 28%, #1B5A70 0%, #0C2C3D 62%, #081D2A 100%);
    border: 6px solid var(--line);
    box-shadow: 0 0 0 2px rgba(46, 211, 195, 0.25), 0 24px 50px rgba(0, 0, 0, 0.45), inset 0 0 60px rgba(0, 0, 0, 0.5);
}
.mc-scope::before {
    content: ""; position: absolute; inset: 0;
    background-image:
        linear-gradient(rgba(255, 255, 255, 0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.04) 1px, transparent 1px);
    background-size: 28px 28px;
}
.mc-scope .sp { position: absolute; width: 3px; height: 3px; border-radius: 50%; background: rgba(255, 255, 255, 0.28); }
.mc-scope .mp { position: absolute; width: 0; height: 0; }
.mc-scope .dot { position: absolute; left: 0; top: 0; transform: translate(-50%, -50%); display: block; }
.mc-scope .bx { position: absolute; border: 2px solid var(--good); border-radius: 4px; display: block; }
.mc-scope .bx em {
    position: absolute; left: -2px; top: -17px;
    font-style: normal; font-size: 10px; font-weight: 700; line-height: 15px;
    padding: 0 5px; background: var(--good); color: #04222A; border-radius: 3px 3px 3px 0; white-space: nowrap;
}
.mc-scope .scan {
    position: absolute; left: 0; right: 0; top: 6%; height: 3px;
    background: linear-gradient(90deg, transparent, var(--aqua-bright), transparent);
    box-shadow: 0 0 18px var(--aqua);
    animation: mcscan 4.5s ease-in-out infinite;
}
@keyframes mcscan { 0% { top: 6%; opacity: 0; } 12% { opacity: 1; } 88% { opacity: 1; } 100% { top: 94%; opacity: 0; } }
.mc-caption { font-size: 0.8rem; color: var(--muted); text-align: center; }
@media (prefers-reduced-motion: reduce) { .mc-scope .scan { animation: none; display: none; } }

.mc-layers { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.mc-layer {
    padding: 20px; border-radius: 14px;
    background: linear-gradient(160deg, var(--surface), var(--surface2));
    border: 1px solid var(--line);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.mc-layer:hover { transform: translateY(-3px); }
.mc-l1:hover { box-shadow: 0 12px 32px rgba(76, 201, 240, 0.20); }
.mc-l2:hover { box-shadow: 0 12px 32px rgba(46, 211, 195, 0.22); }
.mc-l3:hover { box-shadow: 0 12px 32px rgba(255, 194, 75, 0.20); }
.mc-layer .ico { font-size: 1.7rem; }
.mc-layer h4 { margin: 8px 0 2px 0; font-size: 1.1rem; color: #fff; }
.mc-layer .tech { font-weight: 700; font-size: 0.88rem; margin-bottom: 12px; }
.mc-l1 .tech { color: var(--sky); }
.mc-l2 .tech { color: var(--aqua); }
.mc-l3 .tech { color: var(--amber); }
.mc-layer p { margin: 0 0 8px 0; font-size: 0.93rem; line-height: 1.5; color: #B7CCD5; }
.mc-layer p b { color: #fff; }
.mc-l1 { border-top: 4px solid var(--sky); }
.mc-l2 { border-top: 4px solid var(--aqua); }
.mc-l3 { border-top: 4px solid var(--amber); }

.mc-table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 14px; }
.mc-table { width: 100%; border-collapse: collapse; font-size: 0.95rem; background: var(--surface2); }
.mc-table th {
    text-align: left; padding: 13px 16px;
    background: linear-gradient(135deg, #123A4C, #0F3042); color: #fff;
    font-family: 'Sora', sans-serif; font-weight: 600; font-size: 0.9rem;
}
.mc-table td { padding: 13px 16px; border-top: 1px solid var(--line); color: #CFE0E6; }
.mc-table td:first-child { font-weight: 700; color: #fff; }
.mc-rule-note { margin-top: 12px; font-size: 0.93rem; color: var(--muted); line-height: 1.6; }

.mc-chips { display: flex; flex-wrap: wrap; gap: 10px; }
.mc-chip {
    padding: 10px 18px; border-radius: 999px;
    background: linear-gradient(160deg, var(--surface), var(--surface2));
    border: 1px solid var(--line); font-size: 0.95rem; font-weight: 600;
    transition: border-color 0.2s ease, transform 0.2s ease, color 0.2s ease, box-shadow 0.2s ease;
}
.mc-chip:hover { border-color: var(--aqua); color: var(--aqua); transform: translateY(-2px); box-shadow: 0 8px 22px rgba(46, 211, 195, 0.12); }
.mc-chip.plan { border-style: dashed; color: #B7CCD5; }

.metric-card {
    background: linear-gradient(160deg, var(--surface), var(--surface2));
    padding: 20px 12px; border-radius: 14px; text-align: center; border: 1px solid var(--line);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 14px 30px rgba(0, 0, 0, 0.42); }
.metric-card h4 { margin: 6px 0 2px 0; font-family: 'Sora', sans-serif; }
.metric-card p { margin: 0; font-size: 0.88rem; color: #B7CCD5; }

.team-card {
    padding: 20px; border-radius: 14px;
    background: linear-gradient(160deg, var(--surface), var(--surface2));
    border: 1px solid var(--line); text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}
.team-card:hover { transform: translateY(-3px); border-color: rgba(46, 211, 195, 0.5); box-shadow: 0 12px 30px rgba(46, 211, 195, 0.12); }
.team-card h4 { color: var(--aqua); margin: 0 0 6px 0; }
.team-card p { margin: 4px 0; }

.mc-footer { text-align: center; padding: 26px 10px 8px 10px; margin-top: 56px; color: var(--muted); font-size: 0.88rem; border-top: 1px solid var(--line); line-height: 1.7; }
.mc-footer b { color: #fff; }

/* ---------- Responsive ---------- */
@media (max-width: 900px) {
    .mc-strip { grid-template-columns: repeat(2, 1fr); }
    .mc-why, .mc-steps, .mc-layers, .mc-detect { grid-template-columns: 1fr; }
    .mc-stepper { grid-template-columns: repeat(2, 1fr); }
    .mc-title { font-size: 1.7rem; }
    .header { font-size: 1.5rem; }
    .mc-wordmark { font-size: 1.2rem; }
    .mc-tagline { font-size: 0.8rem; }
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# TOP NAVBAR (no sidebar)
# =====================================================

NAV_HOME = "🏠 Home"
NAV_DASH = "🧪 Analysis"
NAV_FAQ = "💬 FAQ & Help"
NAV_ABOUT = "👥 About"
PAGES = [NAV_HOME, NAV_DASH, NAV_FAQ, NAV_ABOUT]

PAGE_NAMES = {
    NAV_HOME: "Home",
    NAV_DASH: "Analysis Dashboard",
    NAV_FAQ: "FAQ & Help",
    NAV_ABOUT: "About",
}


def go_to(target):
    """Callback used by in-page buttons to switch pages."""
    st.session_state["nav"] = target


nav_brand, nav_links = st.columns([1, 1.7])

with nav_brand:
    html("""
    <div class="mc-nav-brand">
        <span class="mc-nav-marker"></span>
        <div class="mc-nav-logo">💧</div>
        <div>
            <div class="mc-nav-title">MicroClear</div>
            <div class="mc-nav-sub">AI-Based Microplastic Pollution Risk Assessment and Mitigation System</div>
        </div>
    </div>
    """)

with nav_links:
    page = st.radio(
        "Navigation",
        PAGES,
        key="nav",
        horizontal=True,
        label_visibility="collapsed",
    )

page_name = PAGE_NAMES[page]


# =====================================================
# PAGE 1: HOME
# =====================================================

if page_name == "Home":

    hero_left, hero_right = st.columns([1.15, 0.85], gap="large")

    with hero_left:
        html("""
        <div class="mc-brand">
            <div class="mc-logo">💧</div>
            <div class="mc-brand-text">
                <div class="mc-wordmark">MicroClear</div>
                <div class="mc-tagline">AI-Based Microplastic Pollution Risk Assessment and Mitigation System</div>
            </div>
        </div>
        <h1 class="mc-title">Know what's in your water before anyone drinks it.</h1>
        <p class="mc-intro">
            MicroClear counts microplastic particles in a microscope image, combines the count with pH, TDS
            and turbidity readings, and classifies drinking water as Low, Medium or High risk. Every result
            comes with a treatment recommendation.
        </p>
        <p class="mc-official">
            Built with YOLOv8 computer vision and a Random Forest model, aligned with WHO and PCRWR
            parameter standard values.
        </p>
        """)
        b1, b2, _ = st.columns([1, 1, 0.8])
        with b1:
            st.button("Start analysis", type="primary", on_click=go_to, args=(NAV_DASH,), key="cta_start")
        with b2:
            st.button("Read the guide", on_click=go_to, args=(NAV_FAQ,), key="cta_guide")

    with hero_right:
        html("""
        <div class="mc-report">
            <div class="mc-report-head">
                <span class="mc-report-title">💧 Example water report</span>
                <span class="pill pill-mod">Medium risk</span>
            </div>
            <div class="mc-row"><span class="k">Microplastics</span><span class="r"><span class="v">4 particles / 100 ml</span><span class="pill pill-mod">Moderate</span></span></div>
            <div class="mc-row"><span class="k">pH</span><span class="r"><span class="v">7.4</span><span class="pill pill-good">Good</span></span></div>
            <div class="mc-row"><span class="k">TDS</span><span class="r"><span class="v">310 mg/L</span><span class="pill pill-mod">Moderate</span></span></div>
            <div class="mc-row"><span class="k">Turbidity</span><span class="r"><span class="v">0.6 NTU</span><span class="pill pill-good">Good</span></span></div>
            <div class="mc-action">
                <b>Recommended:</b> add an activated carbon filter and a 1-5 μm microfiltration stage within one week.
            </div>
            <div class="mc-note">Illustrative values, not a real sample.</div>
        </div>
        """)

    html("""
    <div class="mc-strip">
        <div>🎯 YOLOv8 particle detection</div>
        <div>🧠 Random Forest risk model</div>
        <div>🛡️ Rule-based safety check</div>
        <div>📋 WHO and PCRWR parameter standard values</div>
    </div>
    """)

    html("""
    <div class="mc-h2">Why this project matters</div>
    <p class="mc-lead">Microplastics turn up in drinking water around the world, and most people have no practical way to check for them.</p>
    <div class="mc-why">
        <div>
            <b>Found in drinking water worldwide</b>
            <span>Long-term exposure may affect human health and aquatic ecosystems.</span>
        </div>
        <div>
            <b>Lab testing is slow and costly</b>
            <span>Sending samples away takes time and money that many agencies do not have.</span>
        </div>
        <div>
            <b>Rural areas are left out</b>
            <span>Communities far from a laboratory rarely get their water tested at all.</span>
        </div>
    </div>
    <div class="mc-bottomline">
        MicroClear gives a fast, low-cost first assessment using a digital microscope, three basic meters and a laptop.
    </div>
    """)

    html("""
    <div class="mc-h2">How an assessment works</div>
    <p class="mc-lead">Six steps from a water sample to a treatment plan.</p>
    <div class="mc-steps">
        <div class="mc-step"><div class="n">1</div><b>Prepare the sample</b><span>Filter 100 ml of water through a 0.45 μm membrane and let the filter dry.</span></div>
        <div class="mc-step"><div class="n">2</div><b>Capture an image</b><span>Photograph the filter under the digital microscope and upload the image.</span></div>
        <div class="mc-step"><div class="n">3</div><b>Detect particles</b><span>YOLOv8 finds microplastic particles and counts them with confidence scores.</span></div>
        <div class="mc-step"><div class="n">4</div><b>Enter meter readings</b><span>Add the pH, TDS and turbidity values measured for the same sample.</span></div>
        <div class="mc-step"><div class="n">5</div><b>Assess the risk</b><span>The Random Forest model predicts a risk level and the safety rules check it.</span></div>
        <div class="mc-step"><div class="n">6</div><b>Act on the result</b><span>Get the final risk level with a treatment plan and its urgency.</span></div>
    </div>
    """)

    # ---------------- DETECTOR ILLUSTRATION ----------------
    html("""
    <div class="mc-h2">What the detector sees</div>
    <p class="mc-lead">YOLOv8 scans the filter image and marks every particle it recognises as microplastic.</p>
    <div class="mc-detect">
        <div class="mc-detect-copy">
            <ul>
                <li><b>Every particle gets a box and a score</b>Each detection shows how confident the model is.</li>
                <li><b>Weak detections are ignored</b>Only detections at 50% confidence or higher are counted.</li>
                <li><b>The count feeds the risk model</b>The particle total goes straight into the risk assessment, so it is never typed in by hand.</li>
            </ul>
        </div>
        <div class="mc-scope-wrap">
            <div class="mc-scope">
                <i class="sp" style="top:14%;left:46%"></i><i class="sp" style="top:20%;left:72%"></i>
                <i class="sp" style="top:33%;left:18%"></i><i class="sp" style="top:44%;left:80%"></i>
                <i class="sp" style="top:58%;left:40%"></i><i class="sp" style="top:76%;left:66%"></i>
                <i class="sp" style="top:82%;left:30%"></i><i class="sp" style="top:88%;left:52%"></i>
                <div class="mp" style="top:27%;left:34%">
                    <i class="dot" style="width:30px;height:8px;border-radius:4px;background:#4CC9F0;transform:translate(-50%,-50%) rotate(25deg)"></i>
                    <b class="bx" style="left:-24px;top:-18px;width:48px;height:36px"><em>MP 0.93</em></b>
                </div>
                <div class="mp" style="top:40%;left:66%">
                    <i class="dot" style="width:13px;height:13px;border-radius:50%;background:#FF6B6B"></i>
                    <b class="bx" style="left:-16px;top:-16px;width:32px;height:32px"><em>MP 0.88</em></b>
                </div>
                <div class="mp" style="top:64%;left:30%">
                    <i class="dot" style="width:34px;height:7px;border-radius:4px;background:#3DDC97;transform:translate(-50%,-50%) rotate(-35deg)"></i>
                    <b class="bx" style="left:-24px;top:-22px;width:48px;height:44px"><em>MP 0.91</em></b>
                </div>
                <div class="mp" style="top:70%;left:64%">
                    <i class="dot" style="width:11px;height:11px;border-radius:3px;background:#FFC24B;transform:translate(-50%,-50%) rotate(20deg)"></i>
                    <b class="bx" style="left:-15px;top:-15px;width:30px;height:30px"><em>MP 0.76</em></b>
                </div>
                <div class="scan"></div>
            </div>
            <div class="mc-caption">Illustration of YOLOv8 output on a filter membrane.</div>
        </div>
    </div>
    """)

    html("""
    <div class="mc-h2">Three layers behind every result</div>
    <p class="mc-lead">The machine learning model makes the prediction. The safety rules can raise the result when the readings are dangerous, and they can never lower it.</p>
    <div class="mc-layers">
        <div class="mc-layer mc-l1">
            <div class="ico">🎯</div>
            <h4>Detect</h4>
            <div class="tech">Computer vision, YOLOv8</div>
            <p><b>Takes in:</b> a microscope image of the filtered sample.</p>
            <p><b>Produces:</b> particle count, bounding boxes and detection confidence.</p>
        </div>
        <div class="mc-layer mc-l2">
            <div class="ico">🧠</div>
            <h4>Predict</h4>
            <div class="tech">Machine learning, Random Forest</div>
            <p><b>Takes in:</b> pH, TDS, turbidity and microplastic count.</p>
            <p><b>Produces:</b> a Low, Medium or High risk level with a confidence score.</p>
        </div>
        <div class="mc-layer mc-l3">
            <div class="ico">🛡️</div>
            <h4>Verify</h4>
            <div class="tech">Safety rules, WHO and PCRWR</div>
            <p><b>Takes in:</b> the same readings, compared against fixed standard values.</p>
            <p><b>Produces:</b> an override when the model underestimates the risk.</p>
        </div>
    </div>
    """)

    html("""
    <div class="mc-h2">How each reading is judged</div>
    <p class="mc-lead">Every parameter is rated Good, Moderate or Poor against WHO and PCRWR parameter standard values before the safety check runs.</p>
    <div class="mc-table-wrap">
    <table class="mc-table">
        <thead>
            <tr><th>Parameter</th><th>Good</th><th>Moderate</th><th>Poor</th></tr>
        </thead>
        <tbody>
            <tr><td>pH</td><td><span class="pill pill-good">7.1 to 7.8</span></td><td><span class="pill pill-mod">6.5 to 7.0 or 7.9 to 8.5</span></td><td><span class="pill pill-poor">Outside 6.5 to 8.5</span></td></tr>
            <tr><td>TDS (mg/L)</td><td><span class="pill pill-good">200 to 270</span></td><td><span class="pill pill-mod">150 to 199 or 271 to 350</span></td><td><span class="pill pill-poor">Below 150 or above 350</span></td></tr>
            <tr><td>Turbidity (NTU)</td><td><span class="pill pill-good">Below 1</span></td><td><span class="pill pill-mod">1 to 4.9</span></td><td><span class="pill pill-poor">5 or more</span></td></tr>
            <tr><td>Microplastics (per 100 ml)</td><td><span class="pill pill-good">0</span></td><td><span class="pill pill-mod">1 to 5</span></td><td><span class="pill pill-poor">More than 5</span></td></tr>
        </tbody>
    </table>
    </div>
    <div class="mc-rule-note">One Poor reading sets the rule-based level to High. Two or more Moderate readings set it to Medium. Otherwise it stays Low.</div>
    """)

    html("""
    <div class="mc-h2">Who can use it</div>
    <div class="mc-chips">
        <span class="mc-chip">🏫 Environmental research labs</span>
        <span class="mc-chip">💧 Water quality agencies</span>
        <span class="mc-chip">🤝 NGOs such as Muslim Hands</span>
        <span class="mc-chip">🏠 Household water testing</span>
        <span class="mc-chip">🎓 Academic projects</span>
    </div>
    <div class="mc-h2">What comes next</div>
    <div class="mc-chips">
        <span class="mc-chip plan">📱 Mobile app</span>
        <span class="mc-chip plan">☁️ Cloud monitoring dashboard</span>
        <span class="mc-chip plan">📄 PDF report export</span>
        <span class="mc-chip plan">📡 Real-time water quality tracking</span>
        <span class="mc-chip plan">🧬 Detection trained on larger datasets</span>
    </div>
    <div class="mc-h2">Sustainable Development Goals</div>
    """)

    sdg_cols = st.columns(4)
    sdgs = [
        ("💧", "SDG 6", "Clean Water and Sanitation", "#2ED3C3"),
        ("♻️", "SDG 12", "Responsible Consumption", "#FFC24B"),
        ("🌍", "SDG 13", "Climate Action", "#3DDC97"),
        ("🐟", "SDG 14", "Life Below Water", "#4CC9F0"),
    ]
    for idx, (icon, num, name, color) in enumerate(sdgs):
        with sdg_cols[idx]:
            html(f"""
            <div class="metric-card" style="border-top: 4px solid {color};">
                <div style="font-size: 2rem;">{icon}</div>
                <h4 style="color: {color};">{num}</h4>
                <p>{name}</p>
            </div>
            """)


# =====================================================
# PAGE 2: ANALYSIS DASHBOARD
# =====================================================

elif page_name == "Analysis Dashboard":

    html("<div class='header'>🧪 Water Analysis Dashboard</div>")

    # Placeholder so the progress stepper appears at the top
    stepper_slot = st.container()

    # -------------------------------------------------
    # Initialize detection state
    # -------------------------------------------------

    def reset_detection():
        st.session_state["detection_completed"] = False
        st.session_state["detected_count"] = 0
        st.session_state["detection_confidence"] = 0.0
        st.session_state["detected_image"] = None
        st.session_state["original_image"] = None

    if "detection_completed" not in st.session_state:
        reset_detection()

    # =================================================
    # TWO COLUMNS
    # =================================================

    col1, col2 = st.columns([1, 1], gap="large")

    # =================================================
    # COLUMN 1: IMAGE UPLOAD + YOLO DETECTION
    # =================================================

    with col1:

        with st.container(border=True):

            st.markdown("### 📸 1. Upload & preparation")

            st.markdown("""
            **Standard laboratory procedure**
            - Collect 100 ml water sample
            - Filter using 0.45 μm membrane
            - Dry filter paper properly
            - Capture microscope image
            """)

            uploaded_file = st.file_uploader(
                "Upload microscope image *",
                type=["png", "jpg", "jpeg"],
                help="Upload a clear microscope image of the water sample.",
            )

            if uploaded_file:

                # A different image was uploaded -> old detection is stale
                file_sig = (uploaded_file.name, uploaded_file.size)
                if st.session_state.get("file_sig") != file_sig:
                    st.session_state["file_sig"] = file_sig
                    reset_detection()

                img = Image.open(uploaded_file).convert("RGB")

                st.success(f"✅ {uploaded_file.name} uploaded ({img.width} × {img.height} px)")
                st.caption("The original and detected images appear side by side in the detailed report after the analysis.")

                st.write("")

                if st.button(
                    "🔍 Detect microplastics",
                    type="primary",
                    use_container_width=True,
                ):

                    if yolo_model is None:

                        st.error("YOLO model file not found. Please check the model path.")

                    else:

                        with st.spinner("Running YOLOv8 detection..."):

                            try:

                                # SRS: confidence threshold >= 0.5
                                results = yolo_model(np.array(img), imgsz=640, conf=0.5)
                                result = results[0]
                                boxes = result.boxes

                                detected_count = len(boxes)

                                confidences = [float(box.conf[0]) for box in boxes]
                                avg_conf = float(np.mean(confidences) * 100) if confidences else 0.0

                                # result.plot() returns a BGR array -> convert to RGB for display
                                detected_image = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

                                st.session_state["detected_count"] = detected_count
                                st.session_state["detection_confidence"] = avg_conf
                                st.session_state["detected_image"] = detected_image
                                st.session_state["original_image"] = np.array(img)
                                st.session_state["detection_completed"] = True

                                st.success(
                                    f"Detection complete: {detected_count} particles found "
                                    f"(confidence {avg_conf:.1f}%)"
                                )

                            except Exception as e:

                                st.session_state["detection_completed"] = False
                                st.error(f"❌ YOLO detection error: {e}")

            else:

                st.info("Upload a microscope image to start the analysis.")

                st.session_state["file_sig"] = None
                reset_detection()

    # =================================================
    # COLUMN 2: WATER QUALITY PARAMETERS
    # =================================================

    with col2:

        with st.container(border=True):

            st.markdown("### 📋 2. Water quality parameters")

            detection_done = st.session_state["detection_completed"]
            default_count = st.session_state["detected_count"]
            detection_conf = st.session_state["detection_confidence"]

            if detection_done:
                st.success(
                    f"YOLO detection completed: {default_count} microplastic particles detected"
                )
                st.info(f"Detection confidence: {detection_conf:.1f}%")
            else:
                st.info("⚠️ First upload an image and run YOLOv8 detection.")

            st.markdown("#### 💧 Water quality parameters")

            ph = st.number_input(
                "pH level *",
                min_value=0.0,
                max_value=14.0,
                value=7.1,
                step=0.1,
                format="%.1f",
                disabled=not detection_done,
                help="Enter the pH value of the water sample.",
            )

            tds = st.number_input(
                "TDS (mg/L) *",
                min_value=0.0,
                max_value=1000.0,
                value=270.0,
                step=0.1,
                format="%.1f",
                disabled=not detection_done,
                help="Enter the Total Dissolved Solids (TDS) value in mg/L.",
            )

            turbidity = st.number_input(
                "Turbidity (NTU) *",
                min_value=0.0,
                max_value=20.0,
                value=1.0,
                step=0.1,
                format="%.1f",
                disabled=not detection_done,
                help="Enter the turbidity value of the water sample in NTU.",
            )

            run_analysis = st.button(
                "📈 Analyze risk level",
                type="primary",
                use_container_width=True,
                disabled=not detection_done,
            )

    # -------------------------------------------------
    # Progress stepper (rendered into the top slot)
    # -------------------------------------------------

    if run_analysis:
        current_step = 5
    elif st.session_state["detection_completed"]:
        current_step = 3
    elif uploaded_file:
        current_step = 2
    else:
        current_step = 1

    step_labels = ["Upload image", "Detect particles", "Enter readings", "Analyze risk"]
    step_items = ""
    for i, label in enumerate(step_labels, 1):
        if i < current_step:
            cls, mark = "done", "✓"
        elif i == current_step:
            cls, mark = "active", str(i)
        else:
            cls, mark = "todo", str(i)
        step_items += f'<div class="mc-st {cls}"><span class="mc-st-n">{mark}</span><span class="mc-st-t">{label}</span></div>'

    with stepper_slot:
        html(f'<div class="mc-stepper">{step_items}</div>')

    # =================================================
    # ANALYSIS RESULTS
    # =================================================

    if run_analysis:

        st.write("")

        html("<div class='header'>📋 Analysis Results</div>")

        try:

            # Microplastic count comes ONLY from YOLO
            mp_count = st.session_state["detected_count"]

            assessment = final_risk_engine(ph, tds, turbidity, mp_count)

            risk_level = assessment["final_risk"]
            ml_risk = assessment["ml_risk"]
            ml_confidence = assessment["ml_confidence"]
            override_applied = assessment["override_applied"]

        except Exception as e:

            st.error(f"❌ Error in risk engine: {e}")
            st.stop()

        treatment_info = {
            "Low": {
                "color": "#3DDC97",
                "title": "SAFE FOR CONSUMPTION",
                "technique": "Standard Chlorination & Basic Filtration",
                "steps": [
                    "Continue regular chlorination (if municipal supply)",
                    "Use basic sediment filter (10-20μm)",
                    "Monthly water quality testing recommended",
                ],
                "urgency": "Continue normal use",
            },
            "Medium": {
                "color": "#FFC24B",
                "title": "FILTRATION RECOMMENDED",
                "technique": "Household Water Purifier with Carbon Filter",
                "steps": [
                    "Install activated carbon filter system",
                    "Add 1-5μm microfiltration stage",
                    "Test water weekly for 1 month",
                    "Consider UV disinfection",
                ],
                "urgency": "Install filtration within 1 week",
            },
            "High": {
                "color": "#FF6B6B",
                "title": "IMMEDIATE TREATMENT REQUIRED",
                "technique": "Advanced Reverse Osmosis + UV System",
                "steps": [
                    "🚨 INSTALL RO SYSTEM IMMEDIATELY",
                    "Add UV disinfection stage",
                    "Boil water as interim measure",
                    "Contact local water authority",
                    "Do NOT consume without treatment",
                ],
                "urgency": "TREAT BEFORE CONSUMPTION",
            },
        }
        treatment = treatment_info[risk_level]

        tab1, tab2, tab3 = st.tabs(["📊 Risk Assessment", "💡 Recommendations", "📈 Detailed Report"])

        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                override_text = ""
                if assessment["override_applied"]:
                    override_text = "<br><small style='color: #FFC24B;'>⚠️ Safety override applied</small>"

                html(f"""
                <div class='result-box' style='border-top-color: {treatment['color']};'>
                <small>FINAL RISK LEVEL</small>
                <h1 style='color: {treatment['color']}; margin: 10px 0;'>{risk_level.upper()}</h1>
                <small>ML predicted: {ml_risk} ({ml_confidence:.1f}%)</small>
                {override_text}
                </div>
                """)

            with col2:
                html(f"""
                <div class='result-box' style='border-top-color: {treatment['color']};'>
                <small>URGENCY LEVEL</small>
                <h3 style='color: {treatment['color']}; margin: 10px 0;'>{treatment['urgency']}</h3>
                <small>Action required</small>
                </div>
                """)

            with col3:
                html(f"""
                <div class='result-box' style='border-top-color: {treatment['color']};'>
                <small>TREATMENT METHOD</small>
                <h4 style='margin: 10px 0;'>{treatment['technique']}</h4>
                <small>Recommended solution</small>
                </div>
                """)

        with tab2:
            st.markdown(f"### 🛠️ Treatment plan: **{treatment['title']}**")
            st.markdown("#### Implementation steps")
            for i, step in enumerate(treatment['steps'], 1):
                st.markdown(f"{i}. {step}")

            st.markdown("---")
            st.markdown("#### 📋 Parameter summary")
            params_col1, params_col2 = st.columns(2)
            with params_col1:
                st.metric("Microplastic count", f"{mp_count} particles/100ml",
                          delta="Detected" if mp_count > 0 else "None detected")
                st.metric("pH level", f"{ph}",
                          delta="Optimal" if 7.1 <= ph <= 7.8 else "Out of range",
                          delta_color="normal" if 7.1 <= ph <= 7.8 else "inverse")
            with params_col2:
                st.metric("TDS level", f"{tds} mg/L",
                          delta="Optimal" if 200 <= tds <= 270 else "Out of range",
                          delta_color="normal" if 200 <= tds <= 270 else "inverse")
                st.metric("Turbidity", f"{turbidity} NTU",
                          delta="Clear" if turbidity < 1 else "Cloudy",
                          delta_color="normal" if turbidity < 1 else "inverse")

        with tab3:
            st.markdown("### 🔬 Detailed report")

            st.markdown("#### 🤖 ML model information")
            st.write("**Algorithm:** Random Forest Classifier (Raw Values + Engineered Features)")
            st.write(f"**ML prediction:** {ml_risk}")
            st.write(f"**ML confidence:** {ml_confidence:.1f}%")
            st.write(f"**Rule-based assessment:** {assessment['rule_risk']}")
            st.write(f"**Safety override applied:** {'⚠️ Yes - ML underestimated risk' if assessment['override_applied'] else '✅ No - ML prediction used'}")

            st.write("") 
            html('<div style="height: 14px;"></div>')

            if st.session_state.get("detected_image") is not None:
                st.markdown("#### 🖼️ Sample image vs. detection")
                img_left, img_right = st.columns(2, gap="medium")
                with img_left:
                    original_img = st.session_state.get("original_image")
                    if original_img is not None:
                        st.image(
                            original_img,
                            caption="Original microscope image",
                            width=320,
                        )
                with img_right:
                    st.image(
                        st.session_state["detected_image"],
                        caption=f"YOLOv8 detection: {mp_count} particles found",
                        width=320,
                    )
            st.write("") 
            st.markdown("#### 📊 Parameter categorization")
            cats = assessment['categories']
            icon = {"Good": "✅", "Moderate": "⚠️", "Poor": "❌"}
            cat_data = {
                "Parameter": ["pH", "TDS", "Turbidity", "Microplastic count"],
                "Value": [ph, f"{tds} mg/L", f"{turbidity} NTU", f"{mp_count} particles"],
                "Category": [cats['pH'], cats['TDS'], cats['Turbidity'], cats['MP_Count']],
                "Status": [icon[cats['pH']], icon[cats['TDS']], icon[cats['Turbidity']], icon[cats['MP_Count']]],
                "Parameter standard values": ["7.1-7.8", "200-270 mg/L", "<1 NTU", "0 particles"],
            }
            st.table(cat_data)
            
            st.write("") 

            # =====================================================
            # 6.9 VISUALIZATION AND CHART RENDERING
            # =====================================================
            st.markdown("#### 📈 Visual Status Indicators")
            st.caption("Visual representation of each parameter relative to its safety bands.")
            
            # Generate charts for all 4 parameters in 2 columns
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # pH Chart
                fig_ph = plot_parameter_status("pH", ph, cats['pH'])
                st.pyplot(fig_ph, use_container_width=True)
                plt.close(fig_ph)
                
                # Turbidity Chart
                fig_turb = plot_parameter_status("Turbidity", f"{turbidity} NTU", cats['Turbidity'])
                st.pyplot(fig_turb, use_container_width=True)
                plt.close(fig_turb)

            with col_chart2:
                # TDS Chart
                fig_tds = plot_parameter_status("TDS", f"{tds} mg/L", cats['TDS'])
                st.pyplot(fig_tds, use_container_width=True)
                plt.close(fig_tds)

                # MP Count Chart
                fig_mp = plot_parameter_status("Microplastics", f"{mp_count} particles", cats['MP_Count'])
                st.pyplot(fig_mp, use_container_width=True)
                plt.close(fig_mp)

            st.write("") 
            st.markdown("#### 📚 Reference standard values")
            st.info("""
            **WHO and PCRWR parameter standard values applied:**
            - Pakistan Council of Research in Water Resources (PCRWR)
            - Environmental Protection Agency, Azad Jammu & Kashmir (EPA-AJK)
            - World Health Organization (WHO) Drinking Water Guidelines
            - Muslim Hands International WASH Program
            """)


# =====================================================
# PAGE 3: FAQ PAGE
# =====================================================

elif page_name == "FAQ & Help":
    html("<div class='header'>💬 FAQ & System Documentation</div>")

    st.markdown("""
    ### 📚 MicroClear user guide

    This section provides **technical documentation, system behavior explanation, and troubleshooting help**
    for using the MicroClear AI-Based Microplastic Pollution Risk Assessment and Mitigation System.
    """)

    st.divider()

    faq_sections = {
        "🤔 General overview": [
            {
                "q": "What is MicroClear?",
                "a": "MicroClear is an AI-based Microplastic Pollution Risk Assessment and Mitigation System that detects microplastics in water using YOLOv8 and evaluates water safety using machine learning and rule-based logic."
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

        "🤖 AI & decision system": [
            {
                "q": "How does the hybrid system work?",
                "a": """
                MicroClear uses a 3-layer decision pipeline:

                1️⃣ YOLOv8 detects microplastics from microscope images  
                2️⃣ ML model predicts risk (Low / Medium / High)  
                3️⃣ Rule-based system validates WHO and PCRWR parameter standard values  

                Final output is generated using **ML-first + safety override logic**.
                """
            },
            {
                "q": "Why combine ML and rule-based logic?",
                "a": """
                - ML captures hidden patterns in water contamination  
                - Rules ensure strict WHO and PCRWR parameter standard values  
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

        "🎯 Detection system": [
            {
                "q": "How accurate is microplastic detection?",
                "a": "YOLOv8 model achieves approximately 85–95% accuracy depending on image quality, lighting, and microscope resolution."
            },
            {
                "q": "What images are required?",
                "a": "Microscope images in JPG/PNG format, ideally 640x640 or higher resolution for better detection performance."
            }
        ],

        "💧 Sampling & parameters": [
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
                "a": "pH, TDS, Turbidity, and Microplastic Count are used for final risk prediction, judged against WHO and PCRWR parameter standard values."
            }
        ],

        "⚠️ Troubleshooting": [
            {
                "q": "Why is risk HIGH even with low microplastics?",
                "a": "Because chemical parameters (pH, TDS, turbidity) may violate the WHO and PCRWR parameter standard values."
            },
            {
                "q": "Model not loading?",
                "a": "Ensure YOLO and ML model files exist in correct directory and paths are properly set."
            }
        ],

        "📊 Result interpretation": [
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
                "a": "Yes, treatment suggestions are generated based on WHO and PCRWR parameter standard values."
            }
        ]
    }

    for section_title, questions in faq_sections.items():
        st.markdown(f"## {section_title}")
        for faq in questions:
            with st.expander(f"❓ {faq['q']}"):
                st.markdown(faq["a"])
        st.markdown("")

    st.divider()

    st.markdown("## 📞 Support & documentation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🆘 Technical support

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
    html("<div class='header'>👥 About the MicroClear project</div>")

    st.markdown("## 🌍 Project vision")

    st.markdown("""
    **MicroClear — AI-Based Microplastic Pollution Risk Assessment and Mitigation System** is a
    Final Year environmental monitoring system designed to detect microplastic contamination in
    drinking water using **computer vision and machine learning**.

    The system aims to bridge the gap between:
    - 🧪 Traditional laboratory testing (accurate but expensive)
    - 📱 Field-level rapid testing (fast but less reliable)

    👉 By combining **YOLOv8 + Machine Learning + Rule-based safety system**, MicroClear provides a
    scalable and intelligent solution for real-world water quality assessment, aligned with **WHO and
    PCRWR parameter standard values**.
    """)

    st.divider()

    st.markdown("## 🎯 Project objectives")

    st.markdown("""
    - Detect microplastics using AI-based image processing  
    - Analyze water quality using multiple chemical parameters  
    - Classify water into risk levels (Low, Medium, High)  
    - Ensure safety using WHO and PCRWR parameter standard values  
    - Provide treatment recommendations for users  
    """)

    st.divider()

    st.markdown("## ⚙️ System overview")

    st.markdown("""
    MicroClear is built on a **hybrid intelligence architecture**:

    ### 🎯 1. Computer vision layer
    - YOLOv8 object detection model  
    - Identifies microplastic particles in microscope images  
    - Outputs bounding boxes and confidence scores  

    ### 🧠 2. Machine learning layer
    - Random Forest classification model  
    - Uses water quality parameters:
        - pH  
        - TDS  
        - Turbidity  
        - Microplastic count  

    ### 🛡️ 3. Safety rule engine
    - Ensures compliance with WHO and PCRWR parameter standard values  
    - Overrides ML if unsafe conditions are detected  
    """)

    st.divider()

    st.markdown("## 👥 Development team")

    team_cols = st.columns(3)

    team_members = [
        {"name": "Dost Muhammad", "reg": "FA22-BSE-009", "role": "Lead Developer"},
        {"name": "Usman Ali", "reg": "FA22-BSE-051", "role": "Data Scientist"},
        {"name": "M. Husnain", "reg": "FA22-BSE-065", "role": "Web Developer"},
    ]

    for idx, member in enumerate(team_members):
        with team_cols[idx]:
            html(f"""
            <div class="team-card">
                <h4>{member['name']}</h4>
                <p><b>{member['reg']}</b></p>
                <p style="color:#93AEBB;">{member['role']}</p>
            </div>
            """)

    st.divider()

    st.markdown("## 🏢 Affiliations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎓 Academic institution
        **Mirpur University of Science & Technology (MUST)**  
        - Department: Software Engineering  
        - Batch: 2022–2026  
        - Supervisor: Engr. Iqra Gillani  
        """)

    with col2:
        st.markdown("""
        ### 🤝 Client organization
        **Muslim Hands Mirpur**  
        - Operational Head: Javaid ul Hassan  
        - Contact: +92 300 555064  
        - Location: Mirpur, AJK  

        _Partnering for clean water solutions_
        """)

    st.divider()

    st.markdown("## 🛠️ Technical specifications")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Core technologies
        - Python 3.8+
        - YOLOv8 (Object Detection)
        - Random Forest ML Model
        - Streamlit Web Framework
        - OpenCV Image Processing
        """)

    with col2:
        st.markdown("""
        ### Hardware requirements
        - Digital Microscope (USB)
        - pH / TDS / Turbidity Meters
        - Standard Laptop (4GB+ RAM)
        - Image Capture Setup
        """)

    st.divider()

    st.markdown("## 🚀 System features")

    st.markdown("""
    - Automated microplastic detection from microscope images  
    - Multi-parameter water quality risk analysis  
    - Hybrid ML + Rule-based decision system  
    - Real-time interactive dashboard  
    - Safety-first water classification system  
    """)

    st.divider()

    st.subheader("🌱 Sustainable development impact")

    sdg_impact = {
        "💧 SDG 6: Clean Water": "Provides accessible water quality assessment tools for underserved communities",
        "♻️ SDG 12: Responsible Consumption": "Raises awareness about plastic pollution and its impacts",
        "🌍 SDG 13: Climate Action": "Supports environmental monitoring for climate resilience",
        "🐟 SDG 14: Life Below Water": "Helps prevent microplastic contamination in aquatic ecosystems",
    }

    for sdg, impact in sdg_impact.items():
        st.info(f"**{sdg}:** {impact}")


# =====================================================
# FOOTER
# =====================================================

html("""
<div class="mc-footer">
    <b>💧 MicroClear</b> — AI-Based Microplastic Pollution Risk Assessment and Mitigation System<br>
    Final Year Project 2026 &nbsp;•&nbsp; Department of Software Engineering &nbsp;•&nbsp; MUST Mirpur<br>
    © 2022–2026 MicroClear Project Team
</div>
""")