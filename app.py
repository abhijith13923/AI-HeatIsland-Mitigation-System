"""
app.py
──────
Streamlit frontend for Heat Island Severity Score Predictor.
Dark themed, manual input features.
"""

import streamlit as st
import pandas as pd
import joblib
import os
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------
# 🎨 CONFIG & STYLES
# -------------------------------------------------
st.set_page_config(
    page_title="Heat Island Severity Predictor",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    .stApp {
        background-color: #0c0f14;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Metrics and Cards */
    .metric-card, .result-box, .input-container {
        background: #151821;
        border-radius: 16px;
        border: 1px solid #2d313b;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -2px rgba(0, 0, 0, 0.4);
        padding: 24px;
        transition: all 0.3s ease-in-out;
    }
    
    .metric-card:hover, .input-container:hover {
        transform: translateY(-2px);
        border-color: #10b981;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.2);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #10b981;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 8px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Result Box specific */
    .result-box {
        text-align: center;
        background: linear-gradient(145deg, #151821, #0a0c10);
    }
    
    .severity-text {
        font-size: 4.5rem;
        font-weight: 900;
        margin: 15px 0;
        letter-spacing: 2px;
        text-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    
    .sev-0 { color: #10b981 !important; }
    .sev-1 { color: #facc15 !important; }
    .sev-2 { color: #fb923c !important; }
    .sev-3 { color: #ef4444 !important; }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #10b981, #3b82f6) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        box-shadow: 0 4px 14px 0 rgba(16, 185, 129, 0.3) !important;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #059669, #2563eb) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1lcbmhc {
        background: #080a0e;
        border-right: 1px solid #2d313b;
    }
    
    /* Custom Headers */
    .gradient-text {
        background: linear-gradient(90deg, #10b981, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    .sub-gradient-text {
        background: linear-gradient(90deg, #34d399, #93c5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# 📦 MODEL LOADER
# -------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists("models/uhi_model.keras") or not os.path.exists("models/scaler.pkl"):
        return None, None
    model = tf.keras.models.load_model("models/uhi_model.keras")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

SEVERITY_MAP = {
    0: ("SAFE / LOW", "sev-0", "Optimal urban planning - minimal heat island effect"),
    1: ("MODERATE", "sev-1", "Some heat accumulation - consider more green spaces"),
    2: ("HIGH", "sev-2", "Significant heat island - urgent action recommended"),
    3: ("EXTREME", "sev-3", "Critical situation - immediate intervention required")
}

# -------------------------------------------------
# 🖥️ UI MAIN
# -------------------------------------------------
with st.sidebar:
    # Sidebar Header
    st.markdown('<div style="text-align: center; margin-top: 20px;">', unsafe_allow_html=True)
    st.markdown('<span style="font-size: 4rem;">🏙️</span>', unsafe_allow_html=True)
    st.markdown('<h1 class="gradient-text" style="font-size: 2.2rem; margin-bottom: 0;">EcoSense</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; margin-top: 0;">AI Heat Severity Intelligence</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    gemini_key_input = os.getenv("GEMINI_API_KEY", "")
    
    st.markdown("---")
    
    # Dataset Reference Info
    st.markdown("### 📊 Active Model Parameters")
    st.markdown("""
    <div style="background: rgba(16, 185, 129, 0.1); border-left: 3px solid #10b981; padding: 15px; border-radius: 4px;">
        <span style="font-size: 0.85rem; color: #94a3b8;">Features Analyzed:</span><br>
        <span style="color: #6ee7b7; font-weight: 600;">🌡️ Urban Temp</span><br>
        <span style="color: #6ee7b7; font-weight: 600;">🌳 Rural Temp</span><br>
        <span style="color: #6ee7b7; font-weight: 600;">💧 Humidity</span><br>
        <span style="color: #6ee7b7; font-weight: 600;">🌬️ Wind Speed</span><br>
        <span style="color: #6ee7b7; font-weight: 600;">☁️ Cloud Cover</span><br>
        <span style="color: #6ee7b7; font-weight: 600;">🔥 UHI Intensity</span>
    </div>
    """, unsafe_allow_html=True)

# Main Title Area
st.markdown("""
<div style="text-align: center; margin-bottom: 40px; margin-top: 20px;">
    <h1 class="gradient-text" style="font-size: 3.5rem; margin-bottom: 5px;">Urban Heat Dashboard</h1>
    <p style="color: #94a3b8; font-size: 1.2rem; max-width: 600px; margin: 0 auto;">Simulate and forecast urban heat island severity risk using neural network predictions</p>
</div>
""", unsafe_allow_html=True)


# Layout using grid for inputs
st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.markdown("<h3 class='sub-gradient-text' style='margin-top: 0;'>🎛️ Configure Environmental Parameters</h3>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748b; font-size: 0.9rem; margin-bottom: 30px;'>Manually adjust the environmental conditions to predict the resulting heat severity.</p>", unsafe_allow_html=True)

# First Row of Inputs
col1, col2, col3 = st.columns(3)
with col1:
    u_temp = st.slider("🌡️ Urban Temperature (°C)", min_value=15.0, max_value=50.0, value=35.0, step=0.1)
with col2:
    r_temp = st.slider("🌳 Rural Temperature (°C)", min_value=15.0, max_value=50.0, value=30.0, step=0.1)
with col3:
    humidity = st.slider("💧 Relative Humidity (%)", min_value=10.0, max_value=100.0, value=65.0, step=1.0)

st.markdown("<br>", unsafe_allow_html=True)

# Second Row of Inputs
col4, col5, col6 = st.columns(3)
with col4:
    wind_speed = st.slider("🌬️ Wind Speed (m/s)", min_value=0.0, max_value=25.0, value=4.5, step=0.1)
with col5:
    clouds = st.slider("☁️ Cloud Cover (%)", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
with col6:
    default_uhi = max(0.0, u_temp - r_temp)
    uhi_intensity = st.number_input("🔥 UHI Intensity (°C)", value=round(default_uhi, 2), step=0.1, help="Difference between Urban and Rural temps")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Prediction execution button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_btn = st.button("🚀 PREDICT SEVERITY SCORE", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)


# -------------------------------------------------
#PREDICTION RESULTS
# -------------------------------------------------
if predict_btn:
    # 1. Run the Prediction First
    model, scaler = load_model()
    if model and scaler:
        features = pd.DataFrame(
            [[float(u_temp), float(r_temp), float(humidity), float(wind_speed), float(clouds), float(uhi_intensity)]], 
            columns=["urban_temp", "rural_temp", "humidity", "wind_speed", "clouds", "uhi_intensity"]
        )
        features_scaled = scaler.transform(features)
        prediction_prob = model.predict(features_scaled)
        prediction = int(np.argmax(prediction_prob, axis=1)[0])
        sev_label, sev_class, sev_desc = SEVERITY_MAP.get(prediction, ("UNKNOWN", "sev-0", "N/A"))

        # 2. Get Gemini's Opinion BEFORE rendering the UI
        ai_opinion = "API Key required for deep analysis."
        if gemini_key_input:
            try:
                genai.configure(api_key=gemini_key_input)
                # Using a faster model for the dashboard summary
                gen_model = genai.GenerativeModel('gemini-flash-latest')
                response = gen_model.generate_content(f"In 2 sentences, explain why a UHI intensity of {uhi_intensity}°C with {humidity}% humidity is dangerous and suggest one urban fix.")
                ai_opinion = response.text
            except Exception as e:
                ai_opinion = f"AI service error: {str(e)}"

        # 3. Now render the UI with the AI opinion injected
        import html
        ai_opinion_html = html.escape(ai_opinion.strip()).replace('\n', '<br>')
        
        r1, r2 = st.columns([1.2, 1])
        
        with r1:
            st.markdown(f"""
                <div class="result-box" style="border-radius: 20px; border: 1px solid #2d313b; padding: 40px; position: relative; overflow: hidden; height: 100%;">
                    <div style="position: absolute; top: -20px; right: -20px; font-size: 10rem; opacity: 0.05;">🌡️</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 700; letter-spacing: 2px;">NEURAL NETWORK FORECAST</div>
                    <div class="severity-text {sev_class}" style="font-size: 3.5rem; margin: 10px 0;">Level {prediction}: {sev_label}</div>
                    <p style="color: #cbd5e1; font-size: 1.1rem; margin-bottom: 20px;">{sev_desc}</p>
                    
                    <div style="background: rgba(168, 85, 247, 0.1); border-left: 3px solid #a855f7; padding: 15px; border-radius: 8px; text-align: left;">
                        <p style="color: #a855f7; font-weight: bold; font-size: 0.8rem; margin: 0 0 5px 0; text-transform: uppercase;">✨ Gemini Recommendation</p>
                        <p style="color: #e2e8f0; font-size: 0.9rem; line-height: 1.5; margin: 0; font-style: italic;">"{ai_opinion_html}"</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        with r2:
            st.markdown('<div class="metric-card" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown("<h3 style='margin-top:0; margin-bottom: 20px; color: #e2e8f0; font-size: 1.2rem;'>Probabilistic Confidence Map</h3>", unsafe_allow_html=True)
            
            # Map probabilities to a simple Streamlit progress bar based viz
            probs = prediction_prob[0] * 100
            classes = ["SAFE (0)", "MODERATE (1)", "HIGH (2)", "EXTREME (3)"]
            colors = ["#10b981", "#facc15", "#fb923c", "#ef4444"]
            
            for i in range(len(classes)):
                st.markdown(f"""
                    <div style="margin-bottom: 15px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span style="color: #94a3b8; font-weight: 600; font-size: 0.9rem;">{classes[i]}</span>
                            <span style="color: {colors[i]}; font-weight: 700;">{probs[i]:.1f}%</span>
                        </div>
                        <div style="background-color: #1e293b; border-radius: 8px; height: 12px; width: 100%; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, {colors[i]}80, {colors[i]}); height: 100%; width: {probs[i]}%; border-radius: 8px;"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Optional AI Recommendation engine
        if gemini_key_input:
            with st.spinner("🤖 Consulting AI Analyst for mitigation strategies..."):
                try:
                    genai.configure(api_key=gemini_key_input)
                    
                    # Dynamically query available models for this specific API key
                    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                    if not available_models:
                        raise Exception("Your API key does not have access to any content generation models.")
                        
                    # Auto-select the best available model (prefer any flash, then pro, then whatever exists)
                    model_to_use = next((m for m in available_models if 'flash' in m), 
                                      next((m for m in available_models if 'pro' in m), available_models[0]))
                                      
                    gen_model = genai.GenerativeModel(model_to_use)
                    
                    prompt = f"""
                    You are an expert environmental consultant analyzing an Urban Heat Island situation based on neural network outputs.
                    Current environmental parameters provided by the simulation:
                    - Urban Temperature: {u_temp}°C
                    - Rural Temperature: {r_temp}°C
                    - Humidity: {humidity}%
                    - Wind Speed: {wind_speed} m/s
                    - Cloud Cover: {clouds}%
                    - Model Predicted Severity: Level {prediction} / 3 ({sev_label})
                    
                    Please provide:
                    1. A short analysis of why these meteorological metrics might have driven the model to this conclusion.
                    2. Three urgent and actionable bullet point recommendations to modify these parameters through urban planning.
                    
                    Tone: Professional, Data-driven, Solutions-oriented. No generic fluff. Markdown formatted.
                    """
                    response = gen_model.generate_content(prompt)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 🧠 Generative AI Consultation")
                    st.markdown(f"""
                        <div style="background: #151821; padding: 30px; border-radius: 16px; border-left: 4px solid #a855f7; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
                        {response.text}
                        </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"GenAI Integration Error: {str(e)}")
                    
    else:
        st.error("⚠️ Environment error: 'models/uhi_model.keras' and/or 'models/scaler.pkl' are not present.")
else:
    # Empty State Call to Action
    st.markdown("""
        <div style="background: #151821; border-radius: 16px; padding: 40px; text-align: center; margin-top: 30px; border: 1px dashed #334155;">
            <span style="font-size: 4rem; opacity: 0.6; display: block; margin-bottom: 20px;">🎛️</span>
            <h3 style="color: #94a3b8; font-weight: normal;">Adjust the parameters above and execute the simulation</h3>
            <p style="color: #64748b; font-size: 0.95rem;">The neural network will evaluate the scenario against historical temperature anomalies.</p>
        </div>
    """, unsafe_allow_html=True)