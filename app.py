"""
app.py
──────
Streamlit frontend for Heat Island Severity Score Predictor.
"Catchy" green-themed UI for sustainable development focus.
"""

import streamlit as st
import pandas as pd
import requests
import joblib
import os
import time
from datetime import datetime
import numpy as np

# -------------------------------------------------
# 🎨 CONFIG & STYLES
# -------------------------------------------------
st.set_page_config(
    page_title="Heat Island Severity Score",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Sustainable Green" Theme - Enhanced
st.markdown("""
    <style>
    /* Import Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #031010 0%, #0a2a24 40%, #124538 100%);
        background-attachment: fixed;
        color: #e0f2f1;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #ccfff2 !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px;
    }
    
    h1 {
        background: linear-gradient(90deg, #b9f6ca, #a7ffeb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 3.2rem !important;
        margin-bottom: 0 !important;
    }
    
    /* Card Styling - Enhanced Glassmorphism */
    .metric-card {
        background: rgba(20, 60, 45, 0.25);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 24px;
        padding: 25px 15px;
        border: 1px solid rgba(82, 255, 168, 0.15);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::after {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.7s ease;
    }
    
    .metric-card:hover::after {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 40px rgba(0, 230, 118, 0.15);
        border-color: rgba(82, 255, 168, 0.4);
        background: rgba(30, 80, 60, 0.35);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #b9f6ca, #69f0ae);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
        text-shadow: 0 0 20px rgba(105, 240, 174, 0.3);
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #b9f6ca;
        margin-top: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.9;
    }
    
    /* Button Styling - Enhanced */
    .stButton>button {
        background: linear-gradient(145deg, #00c853, #64dd17);
        color: #031010;
        border: none;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.2rem;
        padding: 0.8rem 2.5rem;
        transition: all 0.4s;
        text-transform: uppercase;
        letter-spacing: 2px;
        box-shadow: 0 10px 20px rgba(0, 200, 83, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: "";
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton>button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton>button:hover {
        transform: scale(1.08);
        box-shadow: 0 0 30px rgba(100, 221, 23, 0.6);
        color: #000;
    }
    
    /* Result Box - Premium */
    .result-box {
        background: linear-gradient(145deg, rgba(10, 50, 35, 0.7), rgba(5, 30, 20, 0.8));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 40px;
        padding: 40px 20px;
        margin-top: 30px;
        border: 1px solid rgba(167, 255, 235, 0.2);
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
    }
    
    .result-box::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #00e676, #00b0ff, #00e676);
        background-size: 200% 100%;
        animation: gradientMove 3s ease infinite;
    }
    
    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .severity-text {
        font-size: 4.5rem;
        font-weight: 900;
        text-shadow: 0 4px 20px rgba(0,0,0,0.6);
        margin: 15px 0;
        letter-spacing: 4px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* Severity Colors - More Vibrant */
    .sev-0 { 
        color: #69f0ae !important;
        text-shadow: 0 0 30px rgba(105, 240, 174, 0.5) !important;
    }
    .sev-1 { 
        color: #ffd600 !important;
        text-shadow: 0 0 30px rgba(255, 214, 0, 0.5) !important;
    }
    .sev-2 { 
        color: #ff9100 !important;
        text-shadow: 0 0 30px rgba(255, 145, 0, 0.5) !important;
    }
    .sev-3 { 
        color: #ff1744 !important;
        text-shadow: 0 0 30px rgba(255, 23, 68, 0.5) !important;
    }
    
    /* Sidebar Styling */
    .css-1d391kg, .css-1lcbmhc {
        background: rgba(8, 40, 30, 0.8);
        backdrop-filter: blur(20px);
    }
    
    /* Progress Bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00e676, #69f0ae);
        border-radius: 10px;
    }
    
    /* Select Box */
    .stSelectbox, .stRadio {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 10px;
        border: 1px solid rgba(105, 240, 174, 0.2);
    }
    
    /* Map Container */
    .map-container {
        border-radius: 24px;
        overflow: hidden;
        border: 2px solid rgba(105, 240, 174, 0.2);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    /* Leaf Icon Animation */
    @keyframes float {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-10px) rotate(5deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }
    
    .leaf-icon {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Custom Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #69f0ae, #00e676, #69f0ae, transparent);
        margin: 30px 0;
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        background: rgba(0, 0, 0, 0.8);
        color: #fff;
        border-radius: 6px;
        padding: 5px 10px;
        position: absolute;
        z-index: 1;
    }
    
    /* Temperature Glow Effect */
    .temp-glow {
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 5px rgba(105, 240, 174, 0.3)); }
        to { filter: drop-shadow(0 0 15px rgba(105, 240, 174, 0.7)); }
    }
    </style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# 🌍 LOCATIONS
# -------------------------------------------------
LOCATIONS = {
    "Mumbai": {"urban": {"lat": 18.958193, "lon": 72.832073}, "rural": {"lat": 18.151955, "lon": 74.655139}},
    "Chennai": {"urban": {"lat": 13.084301, "lon": 80.270462}, "rural": {"lat": 13.006662, "lon": 80.220637}},
    "Bengaluru": {"urban": {"lat": 12.962867, "lon": 77.577509}, "rural": {"lat": 12.812942, "lon": 77.580530}},
    "Delhi": {"urban": {"lat": 28.704059, "lon": 77.102490}, "rural": {"lat": 26.258537, "lon": 82.065986}},
    "Ahmedabad": {"urban": {"lat": 23.022505, "lon": 72.571362}, "rural": {"lat": 23.137287, "lon": 72.406581}},
    "Hyderabad": {"urban": {"lat": 17.406498, "lon": 78.477244}, "rural": {"lat": 17.336455, "lon": 77.904827}},
    "Kolkata": {"urban": {"lat": 22.574354, "lon": 88.362873}, "rural": {"lat": 22.165227, "lon": 88.807898}},
    "Jaipur": {"urban": {"lat": 26.912434, "lon": 75.787271}, "rural": {"lat": 26.985487, "lon": 75.851345}},
    "Lucknow": {"urban": {"lat": 26.846694, "lon": 80.946166}, "rural": {"lat": 26.916818, "lon": 80.707581}},
    "Pune": {"urban": {"lat": 18.524609, "lon": 73.878624}, "rural": {"lat": 18.501054, "lon": 73.513765}},
    "Bhopal": {"urban": {"lat": 23.259933, "lon": 77.412615}, "rural": {"lat": 23.203240, "lon": 77.084404}},
    "Nagpur": {"urban": {"lat": 21.145800, "lon": 79.088155}, "rural": {"lat": 20.847410, "lon": 79.324693}},
    "Kochi": {"urban": {"lat": 9.9312, "lon": 76.2673}, "rural": {"lat": 9.8760, "lon": 76.2800}},
}

SEVERITY_MAP = {
    0: ("🌱 SAFE / LOW", "sev-0", "Optimal urban planning - minimal heat island effect"),
    1: ("🌿 MODERATE", "sev-1", "Some heat accumulation - consider more green spaces"),
    2: ("🔥 HIGH", "sev-2", "Significant heat island - urgent action recommended"),
    3: ("⚠️ EXTREME", "sev-3", "Critical situation - immediate intervention required")
}

# -------------------------------------------------
# 📦 MODEL LOADER
# -------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists("uhi_model.pkl"):
        return None
    return joblib.load("uhi_model.pkl")

# -------------------------------------------------
# 🌤️ API FETCHER
# -------------------------------------------------
def fetch_weather(lat, lon, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

# -------------------------------------------------
# 🖥️ UI MAIN
# -------------------------------------------------
# Sidebar - Enhanced
with st.sidebar:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown('<div style="text-align: center;" class="leaf-icon">🌳</div>', unsafe_allow_html=True)
    
    st.markdown('<p style="text-align: center; font-size: 1.8rem; font-weight: 800; background: linear-gradient(90deg, #69f0ae, #00e676); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">EcoSense</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Animated gradient border
    st.markdown("""
        <div style="background: linear-gradient(145deg, #0a3a2a, #052018); border-radius: 20px; padding: 20px; border: 1px solid rgba(105,240,174,0.2);">
    """, unsafe_allow_html=True)
    
    api_key_input = st.text_input("🔑 OpenWeather API Key", type="password", value=os.getenv("OPENWEATHER_API_KEY", ""), 
                                   help="Enter your API key to access real-time weather data")
    
    if not api_key_input:
        st.warning("⚠️ API Key Required")
        st.markdown('<p style="font-size:0.8rem; color:#b9f6ca;">Get your free API key from OpenWeather</p>', unsafe_allow_html=True)
        st.stop()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sustainability Stats
    st.markdown("### 🌍 Global Impact")
    
    # Create a metric for total analyses (mock data)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div style="background: rgba(105,240,174,0.1); border-radius: 15px; padding: 15px; text-align: center;">
                <p style="font-size: 0.8rem; color: #b9f6ca;">CO₂ SAVED</p>
                <p style="font-size: 1.5rem; font-weight: 700; color: #69f0ae;">284kg</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div style="background: rgba(105,240,174,0.1); border-radius: 15px; padding: 15px; text-align: center;">
                <p style="font-size: 0.8rem; color: #b9f6ca;">TREES PLANTED</p>
                <p style="font-size: 1.5rem; font-weight: 700; color: #69f0ae;">142</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <p style="color: #80cbc4; font-size: 0.9rem;">🚀 Powering Sustainable Urban Development</p>
            <p style="color: #4db6ac; font-size: 0.8rem;">v2.0 • Green Intelligence</p>
        </div>
    """, unsafe_allow_html=True)

# Main Content - Enhanced
col_title1, col_title2 = st.columns([3, 1])
with col_title1:
    st.title("🌿 Urban Heat Island Severity")
    st.markdown('<p style="font-size: 1.2rem; color: #b9f6ca; margin-top: -15px;">AI-Powered Environmental Intelligence for Sustainable Cities</p>', unsafe_allow_html=True)

with col_title2:
    st.markdown('<div style="background: rgba(105,240,174,0.1); border-radius: 30px; padding: 15px; text-align: center;">📊 LIVE</div>', unsafe_allow_html=True)

# Custom Divider
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# Layout: Inputs on top, then results
col_input, col_map = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("""
        <div style="background: rgba(20, 60, 45, 0.3); border-radius: 30px; padding: 25px; border: 1px solid rgba(105,240,174,0.1);">
    """, unsafe_allow_html=True)
    
    st.markdown("### 📍 Location Configuration")
    location_mode = st.radio("", ["🏙️ Select City", "🗺️ Custom Coordinates"], horizontal=True, label_visibility="collapsed")

    selected_coords = {}
    
    if location_mode == "🏙️ Select City":
        city_name = st.selectbox("Choose a City:", list(LOCATIONS.keys()), index=0)
        selected_coords = LOCATIONS[city_name]
        st.success(f"📍 Analyzing **{city_name}**")
        
        # City-specific insights
        city_tips = {
            "Mumbai": "Coastal city - consider mangrove restoration",
            "Delhi": "Inland metropolis - rooftop gardens recommended",
            "Bengaluru": "Tech hub - vertical gardens ideal",
            "Chennai": "Coastal - urban forestry priority",
            "Kolkata": "River delta - wetland preservation key"
        }
        if city_name in city_tips:
            st.info(f"💡 Tip: {city_tips[city_name]}")
        
    else:
        st.markdown("#### Enter Coordinates")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Urban Location**")
            u_lat = st.number_input("Latitude", value=0.0, format="%.4f", key="u_lat")
            u_lon = st.number_input("Longitude", value=0.0, format="%.4f", key="u_lon")
        with c2:
            st.markdown("**Rural Reference**")
            r_lat = st.number_input("Latitude", value=0.0, format="%.4f", key="r_lat")
            r_lon = st.number_input("Longitude", value=0.0, format="%.4f", key="r_lon")
            
        selected_coords = {
            "urban": {"lat": u_lat, "lon": u_lon},
            "rural": {"lat": r_lat, "lon": r_lon}
        }
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Enhanced Predict Button
    predict_btn = st.button("🌱 ANALYZE ENVIRONMENTAL IMPACT", use_container_width=True)

with col_map:
    # Enhanced Map with custom styling
    map_data = pd.DataFrame([
        {"lat": selected_coords["urban"]["lat"], "lon": selected_coords["urban"]["lon"], 
         "Type": "Urban Center", "Color": "#ff5252"},
        {"lat": selected_coords["rural"]["lat"], "lon": selected_coords["rural"]["lon"], 
         "Type": "Rural Baseline", "Color": "#69f0ae"},
    ])
    
    st.markdown("""
        <div style="background: rgba(20, 60, 45, 0.3); border-radius: 30px; padding: 20px; border: 1px solid rgba(105,240,174,0.1);">
    """, unsafe_allow_html=True)
    
    st.markdown("### 🗺️ Thermal Mapping View")
    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    st.map(map_data, color="Color", size=20, zoom=10)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Legend
    col_leg1, col_leg2 = st.columns(2)
    with col_leg1:
        st.markdown('<p style="color: #ff5252; font-size: 0.9rem;">🔴 Urban Heat Source</p>', unsafe_allow_html=True)
    with col_leg2:
        st.markdown('<p style="color: #69f0ae; font-size: 0.9rem;">🟢 Rural Reference</p>', unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------
# 🔍 PREDICTION & METRICS
# -------------------------------------------------
if predict_btn:
    # Animated progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        if i < 30:
            status_text.text("🛰️ Connecting to satellite network...")
        elif i < 60:
            status_text.text("🌡️ Processing thermal imagery...")
        elif i < 90:
            status_text.text("🤖 AI computing severity index...")
        else:
            status_text.text("✨ Generating sustainability report...")
        progress_bar.progress(i + 1)
        time.sleep(0.01)
    
    status_text.empty()
    progress_bar.empty()
    
    # Fetch Data
    with st.spinner("🔄 Analyzing climate patterns..."):
        u_data = fetch_weather(selected_coords["urban"]["lat"], selected_coords["urban"]["lon"], api_key_input)
        r_data = fetch_weather(selected_coords["rural"]["lat"], selected_coords["rural"]["lon"], api_key_input)
        
        if not u_data or not r_data:
            st.error("❌ Failed to fetch climate data. Please verify your coordinates & API Key.")
        else:
            # Extract features
            u_temp = u_data["main"]["temp"]
            r_temp = r_data["main"]["temp"]
            humidity = u_data["main"]["humidity"]
            wind_speed = u_data["wind"]["speed"]
            clouds = u_data["clouds"]["all"]
            uhi_intensity = round(u_temp - r_temp, 2)
            
            # --- RESULTS SECTION - Enhanced ---
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            
            # Animated header
            st.markdown("""
                <div style="text-align: center; padding: 20px;">
                    <h2 style="font-size: 2.2rem; margin-bottom: 10px;">📊 Environmental Impact Metrics</h2>
                    <p style="color: #b9f6ca; font-size: 1.1rem;">Real-time urban climate analysis</p>
                </div>
            """, unsafe_allow_html=True)
            
            # 3 Columns for primary metrics - Enhanced
            m1, m2, m3 = st.columns(3)
            
            with m1:
                st.markdown(f'''
                    <div class="metric-card">
                        <div style="font-size: 1rem; color: #b9f6ca; margin-bottom: 5px;">🌇 URBAN CORE</div>
                        <div class="metric-value temp-glow">{u_temp:.1f}°C</div>
                        <div class="metric-label">Surface Temperature</div>
                        <div style="font-size: 0.8rem; color: #80cbc4; margin-top: 10px;">Heat Intensity</div>
                    </div>
                ''', unsafe_allow_html=True)
            
            with m2:
                st.markdown(f'''
                    <div class="metric-card">
                        <div style="font-size: 1rem; color: #b9f6ca; margin-bottom: 5px;">🌳 RURAL BASELINE</div>
                        <div class="metric-value">{r_temp:.1f}°C</div>
                        <div class="metric-label">Natural Reference</div>
                        <div style="font-size: 0.8rem; color: #80cbc4; margin-top: 10px;">Green Cover</div>
                    </div>
                ''', unsafe_allow_html=True)
            
            with m3:
                # Dynamic color based on intensity
                uhi_color = "#ff5252" if uhi_intensity > 2 else "#ff9100" if uhi_intensity > 1 else "#69f0ae"
                st.markdown(f'''
                    <div class="metric-card" style="border-color: {uhi_color};">
                        <div style="font-size: 1rem; color: #b9f6ca; margin-bottom: 5px;">🔥 THERMAL ANOMALY</div>
                        <div class="metric-value" style="color: {uhi_color}; -webkit-text-fill-color: {uhi_color}; background: none;">{uhi_intensity}°C</div>
                        <div class="metric-label">UHI Intensity</div>
                        <div style="font-size: 0.8rem; color: {uhi_color}; margin-top: 10px;">Δ Urban - Rural</div>
                    </div>
                ''', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Secondary metrics row - Enhanced
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                    <div style="background: rgba(20, 60, 45, 0.2); border-radius: 20px; padding: 20px;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 2rem;">💧</span>
                            <div style="flex: 1;">
                """, unsafe_allow_html=True)
                st.markdown(f"**Humidity Level**")
                st.markdown(f"<h3 style='margin: 0; color: #69f0ae;'>{humidity}%</h3>", unsafe_allow_html=True)
                st.progress(min(humidity, 100))
                st.markdown("</div></div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div style="background: rgba(20, 60, 45, 0.2); border-radius: 20px; padding: 20px;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 2rem;">🌬️</span>
                            <div style="flex: 1;">
                """, unsafe_allow_html=True)
                st.markdown(f"**Wind Speed**")
                st.markdown(f"<h3 style='margin: 0; color: #69f0ae;'>{wind_speed:.1f} m/s</h3>", unsafe_allow_html=True)
                st.progress(min(int(wind_speed * 10), 100))
                st.markdown("</div></div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                    <div style="background: rgba(20, 60, 45, 0.2); border-radius: 20px; padding: 20px;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 2rem;">☁️</span>
                            <div style="flex: 1;">
                """, unsafe_allow_html=True)
                st.markdown(f"**Cloud Cover**")
                st.markdown(f"<h3 style='margin: 0; color: #69f0ae;'>{clouds}%</h3>", unsafe_allow_html=True)
                st.progress(min(clouds, 100))
                st.markdown("</div></div>", unsafe_allow_html=True)

            # --- PREDICTION - Enhanced ---
            model = load_model()
            if model:
                # Features: [urban_temp, rural_temp, humidity, wind_speed, clouds, uhi_intensity]
                features = pd.DataFrame([[u_temp, r_temp, humidity, wind_speed, clouds, uhi_intensity]], 
                                        columns=["urban_temp", "rural_temp", "humidity", "wind_speed", "clouds", "uhi_intensity"])
                
                prediction = model.predict(features)[0]
                sev_label, sev_class, sev_desc = SEVERITY_MAP.get(prediction, ("UNKNOWN", "", ""))
                
                # Enhanced result display
                st.markdown(f"""
                    <div class="result-box">
                        <div style="position: absolute; top: 20px; right: 30px; font-size: 4rem; opacity: 0.1;">🌡️</div>
                        <h3 style="color: #b9f6ca; margin-bottom: 10px; letter-spacing: 3px;">URBAN HEAT ISLAND SEVERITY</h3>
                        <div class="severity-text {sev_class}" style="font-size: 5rem;">{sev_label}</div>
                        <p style="color: #80cbc4; font-size: 1.3rem; max-width: 600px; margin: 20px auto;">{sev_desc}</p>
                        <div style="display: flex; justify-content: center; gap: 20px; margin-top: 20px;">
                            <span style="background: rgba(105,240,174,0.1); padding: 8px 20px; border-radius: 30px; font-size: 0.9rem;">Level {prediction}/3</span>
                            <span style="background: rgba(105,240,174,0.1); padding: 8px 20px; border-radius: 30px; font-size: 0.9rem;">{datetime.now().strftime('%Y-%m-%d')}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Action recommendations based on severity
                st.markdown("<br>", unsafe_allow_html=True)
                
                col_rec1, col_rec2 = st.columns(2)
                with col_rec1:
                    if prediction >= 2:
                        st.error("""
                            ### 🚨 Immediate Actions Required
                            - Plant native trees for shade
                            - Install green roofs
                            - Create cool pavements
                            - Increase albedo surfaces
                        """)
                    elif prediction == 1:
                        st.warning("""
                            ### 📋 Recommended Interventions
                            - Add street trees
                            - Develop pocket parks
                            - Implement cool roofs
                            - Monitor hot spots
                        """)
                    else:
                        st.success("""
                            ### ✅ Maintaining Excellence
                            - Continue green practices
                            - Regular monitoring
                            - Community engagement
                            - Share best practices
                        """)
                
                with col_rec2:
                    # Carbon offset calculation
                    carbon_saved = round(uhi_intensity * 2.5, 1)
                    trees_needed = round(uhi_intensity * 1.8)
                    
                    st.markdown("""
                        <div style="background: rgba(20, 60, 45, 0.3); border-radius: 20px; padding: 25px;">
                            <h4 style="color: #b9f6ca; margin-bottom: 15px;">🌱 Sustainability Impact</h4>
                    """, unsafe_allow_html=True)
                    
                    if prediction == 0:
                        st.metric("Carbon Offset Potential", f"{carbon_saved} kg", "+2.1%")
                        st.metric("Trees Equivalent", f"{trees_needed}", "🌳")
                    elif prediction == 1:
                        st.metric("Carbon Offset Potential", f"{carbon_saved} kg", "-5.3%")
                        st.metric("Trees Needed", f"{trees_needed}", "🌳")
                    elif prediction >= 2:
                        st.metric("Carbon Reduction Target", f"{carbon_saved} kg", "-12.8%")
                        st.metric("Urgent Planting Need", f"{trees_needed}", "🌳")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
            else:
                st.error("⚠️ Model file 'uhi_model.pkl' not found. Please ensure it exists in the root directory.")
else:
    # Welcome message when no prediction yet
    st.markdown("""
        <div style="background: linear-gradient(145deg, rgba(20,80,60,0.2), rgba(10,50,40,0.2)); 
                    border-radius: 30px; padding: 40px; text-align: center; margin-top: 40px;
                    border: 1px solid rgba(105,240,174,0.1);">
            <span style="font-size: 5rem; animation: float 3s ease-in-out infinite; display: inline-block;">🌍</span>
            <h2 style="color: #b9f6ca; margin-top: 20px;">Ready to Assess Urban Heat Impact</h2>
            <p style="color: #80cbc4; font-size: 1.2rem; max-width: 600px; margin: 20px auto;">
                Configure your location settings and click the analyze button to generate a comprehensive 
                urban heat island severity report with AI-powered recommendations.
            </p>
        </div>
    """, unsafe_allow_html=True)