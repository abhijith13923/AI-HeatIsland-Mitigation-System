import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import os

print("TensorFlow Version:", tf.__version__)

# Compute paths relative to this script so it runs from any working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, "data", "uhi_india_dataset.csv")
assets_dir = os.path.join(project_root, "assets")
models_dir = os.path.join(project_root, "models")
os.makedirs(assets_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# 1. Load Data
df = pd.read_csv(data_path)
FEATURE_COLS = ["urban_temp", "rural_temp", "humidity", "wind_speed", "clouds", "uhi_intensity"]
TARGET_COL = "severity_label"

df = df[FEATURE_COLS + [TARGET_COL]].dropna()
X = df[FEATURE_COLS].values
y = df[TARGET_COL].values.astype(int)

# 2. Train-Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Apply SMOTE to handle class imbalance
print("Original class distribution:", np.bincount(y_train))
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print("Resampled class distribution:", np.bincount(y_train_resampled))

# 4. Build Neural Network Model
num_classes = len(np.unique(y))
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_resampled.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. Train Model
history = model.fit(
    X_train_resampled, y_train_resampled,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_scaled, y_test),
    verbose=1
)

# 6. Evaluate & Plot Results
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

# Plot Accuracy & Loss
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()

plt.savefig(os.path.join(assets_dir, 'training_curves.png'))
print(f"Saved {os.path.join(assets_dir, 'training_curves.png')}")

# Confusion Matrix Plot
y_pred_prob = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[f"Sev {i}" for i in range(num_classes)], 
            yticklabels=[f"Sev {i}" for i in range(num_classes)])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(assets_dir, 'confusion_matrix.png'))
print(f"Saved {os.path.join(assets_dir, 'confusion_matrix.png')}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# 7. Save Model & Scaler
model.save(os.path.join(models_dir, "uhi_model.keras"))
joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
print(f"Saved UHI Model to {os.path.join(models_dir, 'uhi_model.keras')} and Scaler to {os.path.join(models_dir, 'scaler.pkl')}")
