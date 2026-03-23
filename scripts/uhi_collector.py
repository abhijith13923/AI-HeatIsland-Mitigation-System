import pandas as pd
import requests
import os
import time
from datetime import datetime

# -------------------------------------------------
# 🔐 API KEY (Notebook-safe)
# -------------------------------------------------
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

class UHIDataLoader:
    def __init__(self, csv_file="data/uhi_india_dataset.csv", api_key=None):
        self.csv_file = csv_file
        self.api_key = api_key
        if not self.api_key:
            raise RuntimeError("OpenWeather API key is missing")

        self.columns = [
            'timestamp', 'city', 'urban_temp', 'rural_temp',
            'humidity', 'wind_speed', 'clouds',
            'uhi_intensity', 'severity_label'
        ]

        # Create CSV only if it doesn't exist
        if not os.path.exists(self.csv_file):
            pd.DataFrame(columns=self.columns).to_csv(self.csv_file, index=False)
            print(f"📁 Created dataset: {os.path.abspath(self.csv_file)}")

    def _get_severity(self, intensity):
        if intensity < 1.0:
            return 0
        elif intensity < 2.5:
            return 1
        elif intensity < 4.5:
            return 2
        else:
            return 3

    def fetch_weather_by_coords(self, lat, lon):
        url = (
            "https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
        )

        try:
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                print(f"⚠️ API Error {response.status_code}: {response.text}")
                return None

            data = response.json()

            if not all(k in data for k in ("main", "wind", "clouds")):
                print("⚠️ Incomplete weather data")
                return None

            return data

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Network error: {e}")
            return None

    def log_data_pair(self, urban_info, rural_info):
        u_data = self.fetch_weather_by_coords(urban_info["lat"], urban_info["lon"])
        r_data = self.fetch_weather_by_coords(rural_info["lat"], rural_info["lon"])

        if not u_data or not r_data:
            print(f"⏭️ Skipping {urban_info['name']} (fetch failed)")
            return

        u_temp = u_data["main"]["temp"]
        r_temp = r_data["main"]["temp"]
        intensity = round(u_temp - r_temp, 2)

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "city": urban_info["name"],
            "urban_temp": u_temp,
            "rural_temp": r_temp,
            "humidity": u_data["main"]["humidity"],
            "wind_speed": u_data["wind"]["speed"],
            "clouds": u_data["clouds"]["all"],
            "uhi_intensity": intensity,
            "severity_label": self._get_severity(intensity)
        }

        df = pd.DataFrame([row], columns=self.columns)
        df.to_csv(self.csv_file, mode="a", header=False, index=False)

        print(
            f"✅ Logged {urban_info['name']} | "
            f"Urban {u_temp}°C | Rural {r_temp}°C | UHI {intensity}°C"
        )


# -------------------------------------------------
# 🌍 LOCATION PAIRS
# -------------------------------------------------
location_pairs = [
    ({"name": "Mumbai", "lat": 18.958193, "lon": 72.832073}, {"name": "Kanheri", "lat": 18.151955, "lon": 74.655139}),
    ({"name": "Chennai", "lat": 13.084301, "lon": 80.270462}, {"name": "Guindy", "lat": 13.006662, "lon": 80.220637}),
    ({"name": "Bengaluru", "lat": 12.962867, "lon": 77.577509}, {"name": "Bannerughatta", "lat": 12.812942, "lon": 77.580530}),
    ({"name": "Delhi", "lat": 28.704059, "lon": 77.102490}, {"name": "Sultanpur", "lat": 26.258537, "lon": 82.065986}),
    ({"name": "Ahmedabad", "lat": 23.022505, "lon": 72.571362}, {"name": "Thol", "lat": 23.137287, "lon": 72.406581}),
    ({"name": "Hyderabad", "lat": 17.406498, "lon": 78.477244}, {"name": "Vikarabad", "lat": 17.336455, "lon": 77.904827}),
    ({"name": "Kolkata", "lat": 22.574354, "lon": 88.362873}, {"name": "Gosaba", "lat": 22.165227, "lon": 88.807898}),
    ({"name": "Jaipur", "lat": 26.912434, "lon": 75.787271}, {"name": "Amber Palace", "lat": 26.985487, "lon": 75.851345}),
    ({"name": "Lucknow", "lat": 26.846694, "lon": 80.946166}, {"name": "Malihabad", "lat": 26.916818, "lon": 80.707581}),
    ({"name": "Pune", "lat": 18.524609, "lon": 73.878624}, {"name": "Mulshi", "lat": 18.501054, "lon": 73.513765}),
    ({"name": "Bhopal", "lat": 23.259933, "lon": 77.412615}, {"name": "Sehore", "lat": 23.203240, "lon": 77.084404}),
    ({"name": "Nagpur", "lat": 21.145800, "lon": 79.088155}, {"name": "Umred", "lat": 20.847410, "lon": 79.324693})
]

# -------------------------------------------------
# 🚀 EXECUTION
# -------------------------------------------------
loader = UHIDataLoader(api_key=OPENWEATHER_API_KEY)

print("🌍 Starting UHI batch fetch...\n")

for urban, rural in location_pairs:
    loader.log_data_pair(urban, rural)
    time.sleep(1.5)

print("\n✅ Batch update complete.")