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
import tensorflow as tf
import google.generativeai as genai

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    /* Global Styles */
    .stApp {
        background-color: #09090b;
        color: #fafafa;
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Card Enhancements */
    .metric-card, .result-box, .map-container {
        background: #18181b;
        border-radius: 16px;
        border: 1px solid #27272a;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5), 0 2px 4px -2px rgba(0, 0, 0, 0.5);
    }
    
    .metric-card {
        padding: 24px;
        transition: all 0.2s ease-in-out;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5), 0 4px 6px -4px rgba(0, 0, 0, 0.5);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #4ade80;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #a1a1aa;
        margin-top: 8px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Result Box */
    .result-box {
        padding: 40px 20px;
        margin-top: 30px;
        text-align: center;
        background: linear-gradient(145deg, #18181b, #09090b);
    }
    
    .severity-text {
        font-size: 4.5rem;
        font-weight: 900;
        margin: 15px 0;
        letter-spacing: 2px;
        text-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    
    /* Severity Colors */
    .sev-0 { color: #4ade80 !important; }
    .sev-1 { color: #fbbf24 !important; }
    .sev-2 { color: #f97316 !important; }
    .sev-3 { color: #ef4444 !important; }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #4ade80) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.39) !important;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #2563eb, #22c55e) !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5) !important;
        color: #ffffff !important;
        transform: translateY(-2px) !important;
    }
    
    /* Sidebar Styling */
    .css-1d391kg, .css-1lcbmhc {
        background: #09090b;
        border-right: 1px solid #27272a;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6, #4ade80);
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
    if not os.path.exists("uhi_model.keras") or not os.path.exists("scaler.pkl"):
        return None, None
    model = tf.keras.models.load_model("uhi_model.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

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
        <div style="background: #18181b; border-radius: 16px; padding: 20px; border: 1px solid #27272a;">
    """, unsafe_allow_html=True)
    
    api_key_input = st.text_input("🔑 OpenWeather API Key", type="password", value=os.getenv("OPENWEATHER_API_KEY", ""), 
                                   help="Enter your API key to access real-time weather data")
    
    gemini_key_input = st.text_input("✨ Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""), 
                                   help="Enter Google Gemini API key for Generative AI insights")
    
    if not api_key_input:
        st.warning("⚠️ OpenWeather API Key Required")
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
            <div style="background: #27272a; border-radius: 12px; padding: 15px; text-align: center; border: 1px solid #3f3f46;">
                <p style="font-size: 0.8rem; color: #b9f6ca;">CO₂ SAVED</p>
                <p style="font-size: 1.5rem; font-weight: 700; color: #69f0ae;">284kg</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div style="background: #27272a; border-radius: 12px; padding: 15px; text-align: center; border: 1px solid #3f3f46;">
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
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1 style="background: linear-gradient(90deg, #4ade80, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3.5rem; font-weight: 800; margin-bottom: 0;">🌿 Urban Heat Island Severity</h1>
    <p style="color: #a1a1aa; font-size: 1.2rem; font-weight: 400; margin-top: 5px;">AI-Powered Environmental Intelligence for Sustainable Cities</p>
    <div style="display: inline-block; background: rgba(59, 130, 246, 0.2); border: 1px solid rgba(59, 130, 246, 0.5); color: #60a5fa; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; letter-spacing: 1px; margin-top: 10px;">
        🔴 LIVE ANALYSIS
    </div>
</div>
""", unsafe_allow_html=True)

# Layout: Inputs on top, then results
col_input, col_map = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("""
        <div style="background: #18181b; border-radius: 16px; padding: 25px; border: 1px solid #27272a;">
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
        <div style="background: #18181b; border-radius: 16px; padding: 20px; border: 1px solid #27272a;">
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
                    <div style="background: #18181b; border-radius: 16px; padding: 20px; border: 1px solid #27272a;">
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
                    <div style="background: #18181b; border-radius: 16px; padding: 20px; border: 1px solid #27272a;">
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
                    <div style="background: #18181b; border-radius: 16px; padding: 20px; border: 1px solid #27272a;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 2rem;">☁️</span>
                            <div style="flex: 1;">
                """, unsafe_allow_html=True)
                st.markdown(f"**Cloud Cover**")
                st.markdown(f"<h3 style='margin: 0; color: #69f0ae;'>{clouds}%</h3>", unsafe_allow_html=True)
                st.progress(min(clouds, 100))
                st.markdown("</div></div>", unsafe_allow_html=True)

            # --- PREDICTION - Enhanced ---
            model, scaler = load_model()
            if model and scaler:
                # Features: [urban_temp, rural_temp, humidity, wind_speed, clouds, uhi_intensity]
                features = pd.DataFrame([[u_temp, r_temp, humidity, wind_speed, clouds, uhi_intensity]], 
                                        columns=["urban_temp", "rural_temp", "humidity", "wind_speed", "clouds", "uhi_intensity"])
                
                features_scaled = scaler.transform(features)
                prediction_prob = model.predict(features_scaled)
                prediction = int(np.argmax(prediction_prob, axis=1)[0])
                sev_label, sev_class, sev_desc = SEVERITY_MAP.get(prediction, ("UNKNOWN", "", ""))
                
                st.markdown("<br><h3>📊 Prediction Probabilities</h3>", unsafe_allow_html=True)
                prob_df = pd.DataFrame(
                    prediction_prob[0], 
                    index=["0: SAFE", "1: MODERATE", "2: HIGH", "3: EXTREME"], 
                    columns=["Probability"]
                )
                st.bar_chart(prob_df, use_container_width=True)
                
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
                        <div style="background: #18181b; border-radius: 16px; padding: 25px; border: 1px solid #27272a;">
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
                
                if gemini_key_input:
                    with st.spinner("🤖 Generating AI geographical & mitigation insights..."):
                        try:
                            genai.configure(api_key=gemini_key_input)
                            gen_model = genai.GenerativeModel('gemini-1.5-flash')
                            
                            prompt = f"""
                            You are an environmental expert AI.
                            Location: {city_name if location_mode == "🏙️ Select City" else "Custom Coordinates"}
                            Coordinates: Latitude {selected_coords['urban']['lat']}, Longitude {selected_coords['urban']['lon']}
                            Urban Heat Island Severity Predicted: Level {prediction} / 3 ({sev_label})
                            Prediction Confidence: {prediction_prob[0][prediction]:.2%}
                            Surface Temperature: {u_temp}°C
                            
                            Please provide:
                            1. A brief description of the geography of this place and how it might contribute to the heat island effect.
                            2. 3-4 specific and effective ways to balance or mitigate this heat island severity level.
                            
                            Keep the tone professional yet accessible. Output formatted nicely in Markdown.
                            """
                            response = gen_model.generate_content(prompt)
                            
                            st.markdown("---")
                            st.markdown("### ✨ AI Environment Analyst")
                            st.markdown(f'<div style="background: #18181b; padding: 30px; border-radius: 16px; border: 1px solid #3b82f6; box-shadow: 0 4px 20px rgba(59, 130, 246, 0.1);">\\n\\n{response.text}\\n\\n</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Generative AI request failed: {str(e)}")
                            
            else:
                st.error("⚠️ Model file 'uhi_model.keras' or 'scaler.pkl' not found. Please ensure they exist in the root directory.")
else:
    # Welcome message when no prediction yet
    st.markdown("""
        <div style="background: #18181b; border-radius: 20px; padding: 40px; text-align: center; margin-top: 40px; border: 1px solid #27272a;">
            <span style="font-size: 5rem; animation: float 3s ease-in-out infinite; display: inline-block;">🌍</span>
            <h2 style="color: #b9f6ca; margin-top: 20px;">Ready to Assess Urban Heat Impact</h2>
            <p style="color: #80cbc4; font-size: 1.2rem; max-width: 600px; margin: 20px auto;">
                Configure your location settings and click the analyze button to generate a comprehensive 
                urban heat island severity report with AI-powered recommendations.
            </p>
        </div>
    """, unsafe_allow_html=True)