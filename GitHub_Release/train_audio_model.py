import os
import librosa
import numpy as np
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================
# PATH CONFIGURATION
# =========================

# Project root = PythonProject2
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Audio folder
AUDIO_DIR = os.path.join(
    PROJECT_ROOT,
    "ASVspoof",
    "LA",
    "ASVspoof2019_LA_train",
    "flac"
)

# Protocol file (labels)
PROTOCOL_FILE = os.path.join(
    PROJECT_ROOT,
    "ASVspoof",
    "LA",
    "ASVspoof2019_LA_train",
    "protocol",
    "ASVspoof2019.LA.cm.train.trn.txt"
)

# Trained model save path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "audio_ai_model.pkl")

# =========================
# FEATURE EXTRACTION
# =========================

def extract_mfcc(file_path, n_mfcc=20):
    """
    Extract MFCC features from an audio file
    """
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)
    except Exception:
        return None

# =========================
# LOAD DATA
# =========================

def load_data(audio_dir, protocol_file, max_per_class=4000):
    """
    Load audio data and labels from ASVspoof protocol
    """
    X, y = [], []

    bonafide_count = 0
    spoof_count = 0

    print("🔍 Loading and processing audio files...")

    with open(protocol_file, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines):
        parts = line.strip().split()
        file_id = parts[1]
        label = parts[-1]

        audio_path = os.path.join(audio_dir, f"{file_id}.flac")

        if not os.path.exists(audio_path):
            continue

        # Limit samples per class
        if label == "bonafide" and bonafide_count >= max_per_class:
            continue
        if label != "bonafide" and spoof_count >= max_per_class:
            continue

        features = extract_mfcc(audio_path)
        if features is None:
            continue

        X.append(features)

        if label == "bonafide":
            y.append(1)  # Human
            bonafide_count += 1
        else:
            y.append(0)  # Spoof
            spoof_count += 1

    print(f"✅ Loaded {bonafide_count} bonafide + {spoof_count} spoof samples")
    return np.array(X), np.array(y)

# =========================
# MAIN TRAINING
# =========================

if __name__ == "__main__":

    # Load data
    X, y = load_data(AUDIO_DIR, PROTOCOL_FILE)

    # Train-test split
    print("📊 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train model
    print("🤖 Training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    print("🧪 Evaluating model...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1-score :", f1)

    # =========================
    # CONFUSION MATRIX
    # =========================

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix – Audio Deepfake Detection")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1], ["Spoof", "Bonafide"])
    plt.yticks([0, 1], ["Spoof", "Bonafide"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # Save model
    print("💾 Saving model...")
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved at: {MODEL_PATH}")
