import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from pathlib import Path
from collections import Counter
from ultralytics import YOLO
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="Helmet Detection", layout="wide")

MODEL_PATH = "best.pt"  # <-- put your trained weights next to this file, or change path
CLASS_NAMES = {0: "Person", 1: "Helmet", 2: "No helmet"}
CLASS_COLORS = {0: (255, 165, 0), 1: (0, 200, 0), 2: (0, 0, 255)}  # BGR


@st.cache_resource
def load_model(path):
    return YOLO(path)


def draw_boxes(frame, result):
    boxes = result.boxes
    counts = Counter()
    if boxes is None:
        return frame, counts
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        label = f"{CLASS_NAMES.get(cls_id, cls_id)} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        counts[cls_id] += 1
    return frame, counts


def show_counts(counts):
    cols = st.columns(3)
    for i, cid in enumerate(sorted(CLASS_NAMES)):
        cols[i].metric(CLASS_NAMES[cid], counts.get(cid, 0))


st.title("Helmet Detection")

if not Path(MODEL_PATH).exists():
    st.error(f"Model weights not found at '{MODEL_PATH}'. Update MODEL_PATH at the top of app.py.")
    st.stop()

model = load_model(MODEL_PATH)

with st.sidebar:
    st.header("Settings")
    conf_threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)

tab_image, tab_video, tab_webcam = st.tabs(["Image", "Video", "Webcam"])

# ---------------- IMAGE TAB ----------------
with tab_image:
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="img")
    if uploaded_img:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        result = model.predict(frame, conf=conf_threshold, verbose=False)[0]
        annotated, counts = draw_boxes(frame.copy(), result)

        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        show_counts(counts)

# ---------------- VIDEO TAB ----------------
with tab_video:
    uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"], key="vid")
    if uploaded_vid:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_vid.read())

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        progress = st.progress(0, text="Processing video...")
        total_counts = Counter()
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = model.predict(frame, conf=conf_threshold, verbose=False)[0]
            annotated, counts = draw_boxes(frame, result)
            total_counts.update(counts)
            writer.write(annotated)

            frame_idx += 1
            if total_frames:
                progress.progress(min(frame_idx / total_frames, 1.0), text=f"Frame {frame_idx}/{total_frames}")

        cap.release()
        writer.release()
        progress.empty()

        st.success("Done.")
        st.video(out_path)
        st.caption("Totals across all frames (a rider appears in many frames, so these are not unique counts):")
        show_counts(total_counts)

        with open(out_path, "rb") as f:
            st.download_button("Download annotated video", f, file_name="annotated_output.mp4")

# ---------------- WEBCAM TAB ----------------
with tab_webcam:
    st.caption("Live detection from your webcam. Allow camera access when prompted.")

    class HelmetProcessor(VideoProcessorBase):
        def __init__(self):
            self.conf = conf_threshold

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            result = model.predict(img, conf=self.conf, verbose=False)[0]
            annotated, _ = draw_boxes(img, result)
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="helmet-webcam",
        video_processor_factory=HelmetProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
    )
