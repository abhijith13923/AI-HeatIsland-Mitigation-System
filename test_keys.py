import os
import requests
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def test_keys():
    print("--- Testing API Keys ---")
    
    # 1. Test OpenWeather API
    openweather_key = os.getenv("OPENWEATHER_API_KEY")
    if not openweather_key:
        print("❌ OpenWeather API Key not found in .env")
    else:
        print("Testing OpenWeather API...")
        # Test with a simple query for Mumbai
        url = f"https://api.openweathermap.org/data/2.5/weather?q=Mumbai&appid={openweather_key}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("[SUCCESS] OpenWeather API Key is valid and working!")
            else:
                print(f"[ERROR] OpenWeather API Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[ERROR] OpenWeather API Request Failed: {e}")

    print("\n")

    # 2. Test Gemini API
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("[ERROR] Gemini API Key not found in .env")
    else:
        print("Testing Gemini API...")
        try:
            genai.configure(api_key=gemini_key)
            # Try to list models as a basic test of authentication
            models = [m.name for m in genai.list_models()]
            if models:
                print("[SUCCESS] Gemini API Key is valid and working!")
            else:
                print("[ERROR] Gemini API authenticated but no models found.")
        except Exception as e:
            print(f"[ERROR] Gemini API Error: {e}")

if __name__ == "__main__":
    test_keys()
