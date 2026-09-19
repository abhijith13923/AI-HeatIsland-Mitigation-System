# EcoSense: Urban Heat Island (UHI) Severity Predictor

EcoSense is an AI-powered dashboard that simulates and forecasts Urban Heat Island (UHI) severity risk using neural network predictions. It analyzes meteorological features and predicts a severity score indicating the intensity of the heat island effect, offering Generative AI recommendations for urban planning mitigation.

## 🚀 Features
- **Neural Network Forecast:** Uses a trained TensorFlow deep learning model to predict UHI severity (SAFE, MODERATE, HIGH, EXTREME) based on environmental parameters.
- **Generative AI Insights:** Integrates Google Gemini API to analyze the predicted scenario and recommend actionable urban fixes.
- **Interactive Dashboard:** Built with Streamlit, featuring a modern, dark-themed UI to manually configure environmental parameters and view probabilistic confidence maps.
- **Automated Data Collection:** Includes scripts to pull real-time urban and rural weather data via the OpenWeather API for continuous model updates.

## 🏗️ Project Architecture & Workflow

1. **Data Collection (`scripts/uhi_collector.py`)**
   - Fetches live weather data for pairs of urban and rural locations across India using the OpenWeather API.
   - Calculates UHI intensity (Urban Temp - Rural Temp) and assigns a severity label.
   - Appends the data to `data/uhi_india_dataset.csv`.

2. **Model Training (`scripts/train_nn.py`)**
   - Loads the UHI dataset and preprocesses the features.
   - Applies **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance in severity labels.
   - Trains a multi-class classification **TensorFlow/Keras Neural Network**.
   - Saves the trained model (`uhi_model.keras`) and standard scaler (`scaler.pkl`) to the `models/` directory.
   - Generates and saves evaluation metrics (accuracy curves, confusion matrix) in the `assets/` directory.

3. **Web Application (`app.py`)**
   - A **Streamlit** interface where users input variables like Urban/Rural temperatures, Humidity, Wind Speed, and Cloud Cover.
   - Uses the pre-trained neural network to output a probabilistic severity score.
   - Prompts the **Google Gemini API** for expert environmental consulting based on the predicted severity.

## 🛠️ Tech Stack & ML Libraries
- **Frontend / Dashboard:** `streamlit`
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning (Preprocessing & Resampling):** `scikit-learn` (StandardScaler, metrics), `imbalanced-learn` (SMOTE)
- **Deep Learning:** `tensorflow` (Keras Sequential API for the Neural Network)
- **Generative AI:** `google-generativeai` (Gemini API for mitigation recommendations)
- **Data Collection:** `requests` (OpenWeather API)
- **Visualizations:** `matplotlib`, `seaborn` (for training plots and confusion matrices)

## ⚙️ Setup & Installation

1. **Clone the repository** (or navigate to the project directory).
2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🔑 Environment Variables & API Keys
To run the full suite of tools, you'll need the following API keys. Create a `.env` file in the root directory and add:

```env
# Required ONLY if you want to fetch new data using scripts/uhi_collector.py
OPENWEATHER_API_KEY="your_openweather_api_key_here"

# Optional (Used in app.py for AI recommendations. You can also input this directly in the UI)
GEMINI_API_KEY="your_gemini_api_key_here"
```

## 🏃‍♂️ How to Run the App
To start the interactive Streamlit dashboard, run the following command from the root of the project:

```bash
streamlit run app.py
```
This will launch the app in your default web browser (typically at `http://localhost:8501`). You can interact with the sliders and hit "PREDICT SEVERITY SCORE" to test the model.
