import os
import cv2
from tqdm import tqdm

# =========================
# CONFIG
# =========================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

VIDEO_DIR_REAL = os.path.join(BASE_DIR, "Celeb-DF-v2", "Celeb-real")
VIDEO_DIR_FAKE = os.path.join(BASE_DIR, "Celeb-DF-v2", "Celeb-synthesis")

OUTPUT_REAL = "data/real"
OUTPUT_FAKE = "data/fake"

FACE_MODEL_PATH = "face_detection_yunet.onnx"

MAX_FRAMES_PER_VIDEO = 10
FRAME_INTERVAL = 5

os.makedirs(OUTPUT_REAL, exist_ok=True)
os.makedirs(OUTPUT_FAKE, exist_ok=True)

# =========================
# LOAD YUNET FACE DETECTOR
# =========================

face_detector = cv2.FaceDetectorYN.create(
    FACE_MODEL_PATH,
    "",
    (320, 320),
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=5000
)

# =========================
# FACE EXTRACTION FUNCTION
# =========================

def extract_faces(video_dir, output_dir, max_videos=None):
    videos = sorted(os.listdir(video_dir))

    if max_videos is not None:
        videos = videos[:max_videos]

    for video in tqdm(videos):
        if not video.lower().endswith((".mp4", ".avi", ".mov")):
            continue

        video_path = os.path.join(video_dir, video)
        video_id = os.path.splitext(video)[0]

        out_video_dir = os.path.join(output_dir, video_id)
        os.makedirs(out_video_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        saved = 0

        while cap.isOpened() and saved < MAX_FRAMES_PER_VIDEO:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % FRAME_INTERVAL == 0:
                h, w = frame.shape[:2]
                face_detector.setInputSize((w, h))
                _, faces = face_detector.detect(frame)

                if faces is not None:
                    for f in faces:
                        x, y, w_box, h_box = map(int, f[:4])
                        x, y = max(0, x), max(0, y)

                        face = frame[y:y+h_box, x:x+w_box]
                        if face.size == 0:
                            continue

                        face = cv2.resize(face, (128, 128))
                        out_path = os.path.join(
                            out_video_dir, f"frame_{saved:04d}.jpg"
                        )
                        cv2.imwrite(out_path, face)
                        saved += 1

            frame_id += 1

        cap.release()

# =========================
# RUN EXTRACTION
# =========================

print("🔍 Extracting REAL video faces…")
extract_faces(VIDEO_DIR_REAL, OUTPUT_REAL, max_videos=600)

print("🔍 Extracting FAKE video faces…")
extract_faces(VIDEO_DIR_FAKE, OUTPUT_FAKE, max_videos=600)

print("✅ Face extraction completed successfully.")
