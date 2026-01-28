import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import joblib
from skimage.feature import local_binary_pattern

from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
import joblib

# =========================
# CONFIG
# =========================

DATA_DIR = "data"
IMG_SIZE = (128, 128)
LBP_RADIUS = 1
LBP_POINTS = 8
LBP_METHOD = "uniform"

# =========================
# LBP FEATURE EXTRACTION
# =========================

def extract_lbp(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.equalizeHist(img)  # 🔥 MUST MATCH app.py

    lbp = local_binary_pattern(
        img,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method=LBP_METHOD
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_POINTS + 3),
        range=(0, LBP_POINTS + 2),
        density=True
    )

    return hist

# =========================
# LOAD DATA
# =========================

X, y = [], []

print("🔍 Extracting LBP features...")


for label, cls in enumerate(["real", "fake"]):
    class_dir = os.path.join(DATA_DIR, cls)

    for video_folder in tqdm(os.listdir(class_dir)):
        video_path = os.path.join(class_dir, video_folder)

        if not os.path.isdir(video_path):
            continue

        video_features = []

        for file in sorted(os.listdir(video_path)):
            if file.lower().endswith(".jpg"):
                frame_path = os.path.join(video_path, file)
                video_features.append(extract_lbp(frame_path))

        if len(video_features) > 0:
            # 🔥 Aggregate frames → ONE sample per video
            X.append(np.median(video_features, axis=0))
            y.append(label)


X = np.array(X)
y = np.array(y)

print(f"✅ Total samples: {len(y)}")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================
# TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# TRAIN SVM
# =========================

print("🤖 Training SVM classifier...")

model = SVC(
    kernel="linear",
    C=1,
    class_weight="balanced",
    probability=True   # 🔥 THIS FIXES predict_proba()
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nAccuracy:", acc)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

# =========================
# CONFUSION MATRIX PLOT
# =========================

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Confusion Matrix – LBP + SVM Video Deepfake Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["Real", "Fake"])
plt.yticks([0, 1], ["Real", "Fake"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.show()

# =========================
# SAVE MODEL
# =========================

joblib.dump(model, "../video_lbp_svm_model.pkl")
joblib.dump(scaler, "../video_lbp_scaler.pkl")

print("💾 Model saved as video_lbp_svm_model.pkl")
print("💾 Scaler saved as video_lbp_scaler.pkl")
