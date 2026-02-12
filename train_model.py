"""
train_model.py
──────────────
Trains a Decision Tree Classifier on the UHI dataset and saves the model.
Designed to be run nightly via GitHub Actions (or manually).
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------------------
# 📂 DATA SOURCE
# -------------------------------------------------
# Pull latest CSV from GitHub raw URL so we always train on fresh data.
# Falls back to local file if the URL fetch fails.
GITHUB_CSV_URL = (
    "https://raw.githubusercontent.com/trinity1611/uhi-project/main/uhi_india_dataset.csv"
)
LOCAL_CSV_PATH = "uhi_india_dataset.csv"
MODEL_OUTPUT_PATH = "uhi_model.pkl"

# Features used for training (no timestamp, no city)
FEATURE_COLS = [
    "urban_temp", "rural_temp", "humidity",
    "wind_speed", "clouds", "uhi_intensity"
]
TARGET_COL = "severity_label"


def load_data():
    """Load the dataset from GitHub raw URL, falling back to local CSV."""
    try:
        print(f"Fetching CSV from GitHub ...")
        df = pd.read_csv(GITHUB_CSV_URL)
        print(f"Loaded {len(df)} rows from GitHub.")
    except Exception as e:
        print(f"GitHub fetch failed ({e}), trying local file ...")
        if not os.path.exists(LOCAL_CSV_PATH):
            raise FileNotFoundError(
                f"Neither GitHub URL nor local file '{LOCAL_CSV_PATH}' is available."
            )
        df = pd.read_csv(LOCAL_CSV_PATH)
        print(f"Loaded {len(df)} rows from local file.")
    return df


def preprocess(df):
    """Select features & target, drop rows with missing values."""
    df = df[FEATURE_COLS + [TARGET_COL]].dropna()
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values.astype(int)
    print(f"Dataset shape after cleaning: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def train_and_save(X, y):
    """Train Decision Tree, evaluate, and save model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    model = DecisionTreeClassifier(random_state=42, max_depth=10)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"Model saved to {os.path.abspath(MODEL_OUTPUT_PATH)}")

    return model, acc


# -------------------------------------------------
# 🚀 MAIN
# -------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    X, y = preprocess(df)
    train_and_save(X, y)
