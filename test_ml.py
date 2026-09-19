import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

def test_model():
    print("Loading models...")
    model_path = "models/uhi_model.keras"
    scaler_path = "models/scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Missing model files!")
        return
        
    try:
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        # Dummy data (matching the app.py inputs)
        features = pd.DataFrame(
            [[35.0, 30.0, 65.0, 4.5, 25.0, 5.0]], 
            columns=["urban_temp", "rural_temp", "humidity", "wind_speed", "clouds", "uhi_intensity"]
        )
        
        features_scaled = scaler.transform(features)
        prediction_prob = model.predict(features_scaled)
        prediction = int(np.argmax(prediction_prob, axis=1)[0])
        
        print(f"Prediction successful! Predicted Class: {prediction}")
        print(f"Probabilities: {prediction_prob}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_model()
