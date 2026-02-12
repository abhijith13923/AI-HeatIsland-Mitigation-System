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

# -------------------------------------------------
# 🎨 CONFIG & STYLES
# -------------------------------------------------
st.set_page_config(
    page_title="Heat Island Severity Score",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Sustainable Green" Theme
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #051a1a 0%, #0e2f2a 50%, #1b4d3e 100%);
        background-attachment: fixed;
        color: #e0f2f1;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #a7ffeb !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Card Styling (Glassmorphism) */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(82, 255, 168, 0.2);
        backdrop-filter: blur(10px);
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(82, 255, 168, 0.15);
        border-color: rgba(82, 255, 168, 0.5);
    }
    .metric-value {
        font_size: 2.2rem;
        font-weight: 700;
        background: -webkit-linear-gradient(90deg, #69f0ae, #00e676);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font_size: 0.9rem;
        color: #b9f6ca;
        margin-top: 5px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #00c853 0%, #64dd17 100%);
        color: #003300;
        border: none;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(100, 221, 23, 0.6);
    }
    
    /* Result Styling */
    .result-box {
        background: rgba(0, 20, 10, 0.4);
        border-radius: 20px;
        padding: 30px;
        margin-top: 25px;
        text-align: center;
        border: 2px solid rgba(167, 255, 235, 0.1);
        position: relative;
        overflow: hidden;
    }
    .result-box::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 4px;
        background: linear-gradient(90deg, #00e676, #00b0ff);
    }
    .severity-text {
        font-size: 3.5rem;
        font-weight: 800;
        text-shadow: 0 4px 10px rgba(0,0,0,0.6);
        margin: 10px 0;
    }

    /* Severity Colors */
    .sev-0 { color: #69f0ae; } /* Low (Green/Safe) */
    .sev-1 { color: #ffff00; } /* Moderate (Yellow) */
    .sev-2 { color: #ffab40; } /* High (Orange) */
    .sev-3 { color: #ff5252; } /* Extreme (Red) */
    
    /* Map Container */
    .map-container {
        border-radius: 16px;
        overflow: hidden;
        border: 2px solid rgba(255,255,255,0.1);
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
    0: ("SAFE / LOW", "sev-0"),
    1: ("MODERATE", "sev-1"),
    2: ("HIGH", "sev-2"),
    3: ("EXTREME", "sev-3")
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
# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3209/3209931.png", width=60)
    st.title("Settings")
    
    st.markdown("---")
    api_key_input = st.text_input("🔑 OpenWeather API Key", type="password", value=os.getenv("OPENWEATHER_API_KEY", ""))
    
    st.caption("Enter key to fetch real-time climate data.")
    if not api_key_input:
        st.warning("API Key needed!")
        st.stop()
        
    st.markdown("### 🌍 About")
    st.info("Predicting Urban Heat Island intensity to promote sustainable city planning. Green cities are cool cities! 🌳")

# Main Content
st.title("🌿 Heat Island Severity Score")
st.markdown("##### AI-Powered Sustainability Intelligence")

# Layout: Inputs on top, then results
col_input, col_map = st.columns([1, 1], gap="medium")

with col_input:
    st.markdown("### 📍 Location Context")
    location_mode = st.radio("Choose Input Mode:", ["Select City", "Custom Coordinates"], horizontal=True)

    selected_coords = {}
    
    if location_mode == "Select City":
        city_name = st.selectbox("Choose a City:", list(LOCATIONS.keys()), index=0)
        selected_coords = LOCATIONS[city_name]
        st.success(f"Scanning: **{city_name}**")
        
    else:
        st.markdown("#### Enter Coordinates")
        c1, c2 = st.columns(2)
        with c1:
            u_lat = st.number_input("Urban Lat", value=0.0, format="%.4f")
            u_lon = st.number_input("Urban Lon", value=0.0, format="%.4f")
        with c2:
            r_lat = st.number_input("Rural Lat", value=0.0, format="%.4f")
            r_lon = st.number_input("Rural Lon", value=0.0, format="%.4f")
            
        selected_coords = {
            "urban": {"lat": u_lat, "lon": u_lon},
            "rural": {"lat": r_lat, "lon": r_lon}
        }
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🌱 ANALYZE SUSTAINABILITY SCORE", use_container_width=True)

with col_map:
    # Map Logic
    map_data = pd.DataFrame([
        {"lat": selected_coords["urban"]["lat"], "lon": selected_coords["urban"]["lon"], "Type": "Urban (Heat Source)", "Color": "#ff5252"}, # Red
        {"lat": selected_coords["rural"]["lat"], "lon": selected_coords["rural"]["lon"], "Type": "Rural (Ref Point)", "Color": "#69f0ae"},   # Green
    ])
    st.markdown("### 🗺️ Geospatial View")
    st.map(map_data, color="Color", size=100, zoom=10)


# -------------------------------------------------
# 🔍 PREDICTION & METRICS
# -------------------------------------------------
if predict_btn:
    with st.spinner("🛰️ Analyzing satellite data & calculating indices..."):
        # Fetch Data
        u_data = fetch_weather(selected_coords["urban"]["lat"], selected_coords["urban"]["lon"], api_key_input)
        r_data = fetch_weather(selected_coords["rural"]["lat"], selected_coords["rural"]["lon"], api_key_input)
        
        if not u_data or not r_data:
            st.error("❌ Failed to fetch climate data. Verify coordinates & API Key.")
        else:
            # Extract features
            u_temp = u_data["main"]["temp"]
            r_temp = r_data["main"]["temp"]
            humidity = u_data["main"]["humidity"]
            wind_speed = u_data["wind"]["speed"]
            clouds = u_data["clouds"]["all"]
            uhi_intensity = round(u_temp - r_temp, 2)
            
            # --- RESULTS ---
            st.markdown("---")
            st.subheader("📊 Environmental Impact Metrics")
            
            # 3 Columns for primary metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{u_temp}°C</div><div class="metric-label">🏙️ Urban Temp</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{r_temp}°C</div><div class="metric-label">🌳 Rural Temp</div></div>', unsafe_allow_html=True)
            with m3:
                # Colorize UHI intensity
                uhi_color = "#ff5252" if uhi_intensity > 2 else "#69f0ae"
                st.markdown(f'<div class="metric-card" style="border-color: {uhi_color};"><div class="metric-value" style="color: {uhi_color};">{uhi_intensity}°C</div><div class="metric-label">🔥 UHI Intensity</div></div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Secondary metrics row
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"**💧 Humidity:** {humidity}%")
                st.progress(min(humidity, 100))
            with c2:
                st.markdown(f"**🌬️ Wind Speed:** {wind_speed} m/s")
                st.progress(min(int(wind_speed * 10), 100))
            with c3:
                st.markdown(f"**☁️ Cloud Cover:** {clouds}%")
                st.progress(min(clouds, 100))

            # --- PREDICTION ---
            model = load_model()
            if model:
                # Features: [urban_temp, rural_temp, humidity, wind_speed, clouds, uhi_intensity]
                features = pd.DataFrame([[u_temp, r_temp, humidity, wind_speed, clouds, uhi_intensity]], 
                                        columns=["urban_temp", "rural_temp", "humidity", "wind_speed", "clouds", "uhi_intensity"])
                
                prediction = model.predict(features)[0]
                sev_label, sev_class = SEVERITY_MAP.get(prediction, ("UNKNOWN", ""))
                
                st.markdown(f"""
                    <div class="result-box">
                        <h3 style="color: #b9f6ca; margin-bottom: 0;">HEAT ISLAND SEVERITY SCORE</h3>
                        <div class="severity-text {sev_class}">{sev_label}</div>
                        <p style="color: #80cbc4; font-size: 1.1rem;">Impact Level: {prediction} / 3</p>
                    </div>
                """, unsafe_allow_html=True)
                
                if prediction >= 2:
                    st.warning("⚠️ High Urban Heat Island effect detected. Consider planting more trees and increasing green cover!")
                else:
                    st.success("✅ Sustainable temperature levels maintained. Good urban planning!")
            else:
                st.error("⚠️ Model file not found.")
