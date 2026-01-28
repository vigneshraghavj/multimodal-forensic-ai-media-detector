import streamlit as st
import os
import tempfile
import subprocess
import numpy as np
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import hashlib
import joblib
import sqlite3
import pandas as pd
from collections import Counter
from datetime import datetime

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata
from io import BytesIO

def is_blurry(face_img, threshold=40.0):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < threshold


from skimage.feature import local_binary_pattern
FACE_MODEL_PATH = "face_detection_yunet.onnx"

# ==================== CONFIG ====================
AUDIO_MODEL_PATH = "audio_ai_training/audio_ai_model.pkl"
VIDEO_MODEL_PATH = "video_lbp_svm_model.pkl"
DB_PATH = "forensic_results.db"

FACE_MODEL_PATH = "face_detection_yunet.onnx"

AUDIO_WEIGHT = 0.7
VIDEO_WEIGHT = 0.3

IMG_SIZE = (128, 128)
LBP_RADIUS = 1
LBP_POINTS = 8
LBP_METHOD = "uniform"

# ==================== UI ====================
st.set_page_config(layout="wide")
st.title("🕵️ Multimodal Forensic AI Media Detector")

# ==================== CASE DETAILS ====================
st.subheader("📁 Case Details")
case_id = st.text_input("Case ID")
investigator = st.text_input("Investigator Name")

# ==================== DATABASE ====================
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT,
    investigator TEXT,
    filename TEXT,
    md5 TEXT,
    sha256 TEXT,
    audio_result TEXT,
    audio_conf REAL,
    video_result TEXT,
    video_conf REAL,
    final_score REAL,
    timestamp TEXT
)
""")
conn.commit()

# ==================== HISTORY VIEWER ====================
def fetch_history():
    cursor.execute("""
        SELECT case_id, investigator, filename,
               audio_result, audio_conf,
               video_result, video_conf,
               final_score, timestamp
        FROM analysis_results
        ORDER BY timestamp DESC
    """)
    rows = cursor.fetchall()
    columns = [
        "Case ID", "Investigator", "Filename",
        "Audio Result", "Audio Confidence",
        "Video Result", "Video Confidence",
        "Final Score", "Timestamp"
    ]
    return pd.DataFrame(rows, columns=columns)

st.sidebar.title("📂 Forensic History")

if st.sidebar.button("View Analysis History"):
    history_df = fetch_history()
    if history_df.empty:
        st.sidebar.warning("No cases recorded yet.")
    else:
        st.subheader("🗂️ Case History")
        st.dataframe(history_df, use_container_width=True)
        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download History as CSV",
            csv,
            "forensic_case_history.csv",
            "text/csv"
        )

st.sidebar.markdown("### ⚠️ Database Management")

if st.sidebar.button("🗑️ Clear All Case History"):
    cursor.execute("DELETE FROM analysis_results")
    conn.commit()
    st.sidebar.success("All forensic history cleared successfully.")

# ==================== LOAD MODELS ====================
audio_model = joblib.load(AUDIO_MODEL_PATH)

video_model = joblib.load("video_lbp_svm_model.pkl")
video_scaler = joblib.load("video_lbp_scaler.pkl")

st.write("📌 Video model classes (0=Real, 1=Fake):", video_model.classes_)


face_detector = cv2.FaceDetectorYN.create(
    FACE_MODEL_PATH,
    "",
    (320, 320),
    score_threshold=0.75,
    nms_threshold=0.3,
    top_k=5000
)

# ==================== UTILITIES ====================
def compute_hashes(path):
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()

# ==================== AUDIO ====================
def extract_audio_features(audio_path, n_mfcc=20):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0), y, sr

def predict_audio(audio_path):
    features, y, sr = extract_audio_features(audio_path)
    pred = audio_model.predict([features])[0]
    prob = audio_model.predict_proba([features])[0]
    label = "AI-Generated Voice" if pred == 0 else "Human Voice"
    confidence = float(max(prob))
    return label, confidence, y, sr

# ==================== VIDEO ====================
def extract_lbp(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE)
    gray = cv2.equalizeHist(gray)

    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_POINTS + 3),
        range=(0, LBP_POINTS + 2),
        density=True
    )
    return hist

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fake_scores = []
    frame_id = 0

    while cap.isOpened() and frame_id < 200:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % 10 == 0:
            h, w = frame.shape[:2]
            face_detector.setInputSize((w, h))
            _, detections = face_detector.detect(frame)

            if detections is not None:
                for det in detections:
                    x, y, w_box, h_box = map(int, det[:4])
                    x, y = max(0, x), max(0, y)

                    face = frame[y:y+h_box, x:x+w_box]
                    if face.size == 0:
                        continue

                    features = extract_lbp(face)
                    features = video_scaler.transform([features])

                    prob = video_model.predict_proba(features)[0]
                    fake_index = list(video_model.classes_).index(1)
                    fake_scores.append(prob[fake_index])

        frame_id += 1

    cap.release()

    if len(fake_scores) == 0:
        return "VIDEO ANALYSIS INCONCLUSIVE", 0.0

    avg_fake = float(np.mean(fake_scores))

    if avg_fake > 0.6:
        return "LIKELY DEEPFAKE VIDEO", round(avg_fake, 2)
    elif avg_fake < 0.4:
        return "LIKELY REAL VIDEO", round(1 - avg_fake, 2)
    else:
        return "VIDEO UNCERTAIN", round(avg_fake, 2)





# ==================== AUDIO FROM VIDEO (FIXED) ====================
def extract_audio_from_video(video_path, out_audio):
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            out_audio
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        return False
    if not os.path.exists(out_audio):
        return False
    if os.path.getsize(out_audio) < 1024:
        return False

    return True

# ==================== CONFIDENCE GRAPH ====================
def show_confidence_graph(audio_c, video_c, final_c):
    df = pd.DataFrame({
        "Modality": ["Audio", "Video", "Final"],
        "Confidence": [audio_c, video_c, final_c]
    })
    st.subheader("📊 Confidence Analysis")
    st.bar_chart(df.set_index("Modality"))

# ==================== PDF ====================
def generate_pdf(data):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    text = c.beginText(40, 750)
    for k, v in data.items():
        text.textLine(f"{k}: {v}")
    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ==================== MAIN ====================
uploaded_file = st.file_uploader("Upload Audio or Video", type=["wav", "mp3", "mp4", "mov"])

if uploaded_file and case_id and investigator:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    md5, sha = compute_hashes(file_path)

    audio_label, audio_conf, y, sr = "N/A", 0.0, None, None
    video_label, video_conf = "N/A", 0.0

    audio_path = None
    if suffix in [".mp4", ".mov"]:
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        if extract_audio_from_video(file_path, temp_audio):
            audio_path = temp_audio
    else:
        audio_path = file_path

    if audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 1024:
        try:
            audio_label, audio_conf, y, sr = predict_audio(audio_path)

            st.subheader("Audio Waveform")
            fig, ax = plt.subplots(figsize=(8, 2.5))  # 👈 medium waveform
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set_title("Audio Waveform")
            st.pyplot(fig)

            st.subheader(" Spectrogram")
            fig, ax = plt.subplots(figsize=(8, 3.5))  # 👈 medium spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            librosa.display.specshow(
                librosa.power_to_db(S),
                sr=sr,
                x_axis="time",
                y_axis="mel",
                ax=ax
            )
            ax.set_title("Mel Spectrogram")
            plt.colorbar(ax.collections[0], ax=ax, format="%+2.0f dB")
            st.pyplot(fig)


        except Exception as e:
            st.error(f"Audio analysis failed: {e}")
            audio_label, audio_conf = "Audio Analysis Failed", 0.0
    else:
        audio_label, audio_conf = "No valid audio stream", 0.0

    if suffix in [".mp4", ".mov"]:
        video_label, video_conf = predict_video(file_path)

    final_score = round(
        (AUDIO_WEIGHT * audio_conf + VIDEO_WEIGHT * video_conf)
        / (AUDIO_WEIGHT + VIDEO_WEIGHT),
        2
    )

    cursor.execute("""
        INSERT INTO analysis_results
        (case_id, investigator, filename, md5, sha256,
         audio_result, audio_conf,
         video_result, video_conf,
         final_score, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        case_id, investigator, uploaded_file.name,
        md5, sha,
        audio_label, audio_conf,
        video_label, video_conf,
        final_score, datetime.now().isoformat()
    ))
    conn.commit()

    st.markdown(f"### 🎙 Audio Result: **{audio_label}** ({audio_conf:.2f})")
    st.markdown(f"### 🎥 Video Result: **{video_label}** ({video_conf:.2f})")
    st.markdown(f"## 🔍 Final Forensic Score: **{final_score:.2f}**")

    show_confidence_graph(audio_conf, video_conf, final_score)

    pdf = generate_pdf({
        "Case ID": case_id,
        "Investigator": investigator,
        "Filename": uploaded_file.name,
        "MD5": md5,
        "SHA256": sha,
        "Audio Result": audio_label,
        "Audio Confidence": audio_conf,
        "Video Result": video_label,
        "Video Confidence": video_conf,
        "Final Score": final_score
    })

    st.download_button("📄 Download Forensic PDF Report", pdf, "forensic_report.pdf")
else:
    st.info("Please enter Case ID, Investigator Name, and upload a file.")
