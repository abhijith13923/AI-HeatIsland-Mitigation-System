"""
app.py
──────
Streamlit frontend for UHI Severity Prediction.
Fetches real-time weather data from OpenWeather API and predicts UHI severity.
Enhanced UI with Map & Custom Locations.
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
    page_title="Urban Heat Island Predictor",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern, dark-themed UI
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: #ffffff;
    }
    
    /* Card Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font_size: 2.5rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #00d4ff, #00ffaa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font_size: 1rem;
        color: #cfcfcf;
        margin-top: 5px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Result Styling */
    .result-box {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 20px;
        padding: 30px;
        margin-top: 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .severity-text {
        font-size: 3.5rem;
        font-weight: 800;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Severity Colors */
    .sev-0 { color: #00ffaa; } /* Low */
    .sev-1 { color: #ffe600; } /* Moderate */
    .sev-2 { color: #ffaa00; } /* High */
    .sev-3 { color: #ff0055; } /* Extreme */
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
    "Kochi": {"urban": {"lat": 9.9312, "lon": 76.2673}, "rural": {"lat": 9.8760, "lon": 76.2800}},  # Added Kochi
}

SEVERITY_MAP = {
    0: ("NEGLIGIBLE / LOW", "sev-0"),
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
st.title("🏙️ UHI Severity Predictor")
st.markdown("### Real-time Urban Heat Island Intelligence")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key_input = st.text_input("OpenWeather API Key", type="password", value=os.getenv("OPENWEATHER_API_KEY", ""))
    
    st.info("ℹ️ Enter your OpenWeather API Key to fetch live data. If you have an environment variable `OPENWEATHER_API_KEY` set, it will be pre-filled.")
    
    if not api_key_input:
        st.warning("⚠️ API Key is required!")
        st.stop()

# Layout: 2 Columns (Inputs | Map)
col_left, col_right = st.columns([1, 1.5], gap="large")

with col_left:
    st.subheader("📍 Location Selection")
    location_mode = st.radio("Select Mode:", ["Predefined City", "Custom Coordinates"], horizontal=True)

    selected_coords = {}
    
    if location_mode == "Predefined City":
        city_name = st.selectbox("Select City", list(LOCATIONS.keys()), index=0)
        selected_coords = LOCATIONS[city_name]
        st.success(f"Selected: **{city_name}**")
        
    else:
        st.markdown("#### Custom Coordinates")
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

    predict_btn = st.button("🚀 Analyze & Predict", use_container_width=True)


# Map Logic
map_data = pd.DataFrame([
    {"lat": selected_coords["urban"]["lat"], "lon": selected_coords["urban"]["lon"], "Type": "Urban", "Color": "#FF0000"},
    {"lat": selected_coords["rural"]["lat"], "lon": selected_coords["rural"]["lon"], "Type": "Rural", "Color": "#00FF00"},
])

with col_right:
    st.subheader("🗺️ Geospatial View")
    st.map(map_data, color="Color", size=20, zoom=9)


# -------------------------------------------------
# 🔍 PREDICTION LOGIC
# -------------------------------------------------
if predict_btn:
    with st.spinner("Fetching live satellite weather data..."):
        # Fetch Data
        u_data = fetch_weather(selected_coords["urban"]["lat"], selected_coords["urban"]["lon"], api_key_input)
        r_data = fetch_weather(selected_coords["rural"]["lat"], selected_coords["rural"]["lon"], api_key_input)
        
        if not u_data or not r_data:
            st.error("❌ Failed to fetch weather data. Please check your API Key and coordinates.")
        else:
            # Extract features
            u_temp = u_data["main"]["temp"]
            r_temp = r_data["main"]["temp"]
            humidity = u_data["main"]["humidity"]
            wind_speed = u_data["wind"]["speed"]
            clouds = u_data["clouds"]["all"]
            uhi_intensity = round(u_temp - r_temp, 2)
            
            # --- RESULTS SECTION ---
            st.divider()
            st.subheader("📊 Live Environmental Metrics")
            
            # Row 1: Temperatures
            rm1, rm2, rm3 = st.columns(3)
            with rm1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{u_temp}°C</div><div class="metric-label">Urban Temp</div></div>', unsafe_allow_html=True)
            with rm2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{r_temp}°C</div><div class="metric-label">Rural Temp</div></div>', unsafe_allow_html=True)
            with rm3:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{uhi_intensity}°C</div><div class="metric-label">UHI Intensity</div></div>', unsafe_allow_html=True)
            
            st.write("") # Spacer
            
            # Row 2: Atmosphere
            rm4, rm5, rm6 = st.columns(3)
            with rm4:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{humidity}%</div><div class="metric-label">Humidity</div></div>', unsafe_allow_html=True)
            with rm5:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{wind_speed} m/s</div><div class="metric-label">Wind Speed</div></div>', unsafe_allow_html=True)
            with rm6:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{clouds}%</div><div class="metric-label">Cloud Cover</div></div>', unsafe_allow_html=True)

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
                        <h2 style="color: #bbb; margin-bottom: 10px;">PREDICTED SEVERITY</h2>
                        <div class="severity-text {sev_class}">{sev_label}</div>
                        <p style="color: #888; margin-top: 10px;">Severity Level: {prediction}</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error("⚠️ Model file (uhi_model.pkl) not found. Please train the model first.")
