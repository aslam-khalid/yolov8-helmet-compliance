import streamlit as st
import cv2
import numpy as np
import tempfile
import imageio
from pathlib import Path
from collections import Counter
from ultralytics import YOLO
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="Helmet Compliance Inspector", page_icon="🦺", layout="wide")

MODEL_PATH = "best.pt"  # <-- put your trained weights next to this file, or change path
CLASS_NAMES = {0: "Person", 1: "Helmet", 2: "No helmet"}

# BGR for cv2 drawing — matches the UI accent colors below
CLASS_COLORS_BGR = {0: (128, 114, 107), 1: (77, 163, 63), 2: (45, 70, 225)}
CLASS_HEX = {0: "#6B7280", 1: "#3FA34D", 2: "#E1462D"}

# ---------------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --asphalt: #1E2124;
    --panel: #24272A;
    --panel-border: #35393C;
    --amber: #FFB300;
    --text-primary: #F2EFE9;
    --text-muted: #9BA1A6;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--asphalt);
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text-primary);
}

[data-testid="stSidebar"] {
    background-color: var(--panel);
    border-right: 1px solid var(--panel-border);
}

/* hazard stripe bar — the signature element */
.hazard-strip {
    height: 8px;
    width: 100%;
    background: repeating-linear-gradient(
        45deg,
        var(--amber),
        var(--amber) 10px,
        #1a1a1a 10px,
        #1a1a1a 20px
    );
    border-radius: 2px;
    margin: 0 0 20px 0;
    opacity: 0.9;
}
.hazard-strip.bottom { margin: 20px 0 28px 0; }

.console-title {
    font-family: 'Oswald', sans-serif;
    font-weight: 700;
    font-size: 2.6rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: var(--text-primary);
    margin-bottom: 0.2rem;
    line-height: 1.1;
}
.console-subtitle {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text-muted);
    font-size: 1rem;
    margin-bottom: 0;
}

.sidebar-label {
    font-family: 'Oswald', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.85rem;
    color: var(--amber);
    margin-bottom: 4px;
}

/* readout cards */
.readout-row { display: flex; gap: 12px; margin-top: 8px; margin-bottom: 4px; }
.readout-card {
    flex: 1;
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-left: 4px solid var(--card-accent, var(--amber));
    border-radius: 4px;
    padding: 14px 16px;
}
.readout-label {
    font-family: 'Oswald', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.78rem;
    color: var(--text-muted);
}
.readout-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: var(--text-primary);
}

.status-banner {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.95rem;
    padding: 10px 16px;
    border-radius: 4px;
    margin-bottom: 6px;
    border: 1px solid var(--panel-border);
}
.status-violation { background: rgba(225,70,45,0.12); border-color: #E1462D; color: #FF8A75; }
.status-clear { background: rgba(63,163,77,0.12); border-color: #3FA34D; color: #7CD68B; }

[data-testid="stTabs"] button p {
    font-family: 'Oswald', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.9rem;
}

[data-testid="stFileUploader"] {
    background: var(--panel);
    border: 1px dashed var(--panel-border);
    border-radius: 6px;
    padding: 6px;
}

.stSlider label p {
    font-family: 'Oswald', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.85rem;
    color: var(--text-muted) !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(path):
    return YOLO(path)


@st.cache_resource
def load_vehicle_model():
    # General-purpose COCO model — used only to check "is there a motorcycle/bicycle
    # in frame" so helmet detections outside a road context don't get flagged.
    return YOLO("yolov8n.pt")


VEHICLE_CLASS_IDS = {1, 3}  # COCO: 1=bicycle, 3=motorcycle
VEHICLE_CONF = 0.25


def get_vehicle_boxes(frame, vehicle_model):
    result = vehicle_model.predict(frame, conf=VEHICLE_CONF, classes=list(VEHICLE_CLASS_IDS), verbose=False)[0]
    boxes = []
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
    return boxes


def is_near_vehicle(box, vehicle_boxes):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    for vx1, vy1, vx2, vy2 in vehicle_boxes:
        vw, vh = vx2 - vx1, vy2 - vy1
        # riders' heads sit above the vehicle box, so extend generously upward
        ex1, ex2 = vx1 - 0.6 * vw, vx2 + 0.6 * vw
        ey1, ey2 = vy1 - 2.5 * vh, vy2 + 0.3 * vh
        if ex1 <= cx <= ex2 and ey1 <= cy <= ey2:
            return True
    return False


def draw_boxes(frame, result, vehicle_boxes=None, require_vehicle=False):
    boxes = result.boxes
    counts = Counter()
    if boxes is None:
        return frame, counts
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if require_vehicle and cls_id in (1, 2) and not is_near_vehicle((x1, y1, x2, y2), vehicle_boxes or []):
            continue

        color = CLASS_COLORS_BGR.get(cls_id, (255, 255, 255))
        label = f"{CLASS_NAMES.get(cls_id, cls_id)} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        counts[cls_id] += 1
    return frame, counts


def render_readout(counts):
    no_helmet = counts.get(2, 0)
    if no_helmet > 0:
        st.markdown(
            f'<div class="status-banner status-violation">⚠ {no_helmet} rider(s) without a helmet detected</div>',
            unsafe_allow_html=True,
        )
    elif sum(counts.values()) > 0:
        st.markdown(
            '<div class="status-banner status-clear">✓ No violations detected in this frame</div>',
            unsafe_allow_html=True,
        )

    cards_html = '<div class="readout-row">'
    for cid in sorted(CLASS_NAMES):
        cards_html += f'''
        <div class="readout-card" style="--card-accent:{CLASS_HEX[cid]}">
            <div class="readout-label">{CLASS_NAMES[cid]}</div>
            <div class="readout-value">{counts.get(cid, 0):02d}</div>
        </div>'''
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown('<div class="hazard-strip"></div>', unsafe_allow_html=True)
st.markdown('<div class="console-title">Helmet Compliance Inspector</div>', unsafe_allow_html=True)
st.markdown('<div class="console-subtitle">YOLOv8 road-safety detector — check a photo, review footage, or run a live feed.</div>', unsafe_allow_html=True)
st.markdown('<div class="hazard-strip bottom"></div>', unsafe_allow_html=True)

if not Path(MODEL_PATH).exists():
    st.error(f"Model weights not found at '{MODEL_PATH}'. Update MODEL_PATH at the top of app.py.")
    st.stop()

model = load_model(MODEL_PATH)
vehicle_model = load_vehicle_model()

with st.sidebar:
    st.markdown('<div class="sidebar-label">Inspection Settings</div>', unsafe_allow_html=True)
    conf_threshold = st.slider("Detection sensitivity", 0.05, 0.95, 0.25, 0.05)
    st.caption("Lower catches more, but risks false alarms. Higher is stricter.")
    require_vehicle = st.checkbox("Require a visible motorcycle/bicycle", value=True)
    st.caption("Every person in frame is still counted. This only limits Helmet/No-helmet flags to people near a two-wheeler, so a pedestrian isn't marked as a violation.")

tab_image, tab_video, tab_webcam = st.tabs(["Photo", "Footage", "Live Feed"])

# ---------------- IMAGE TAB ----------------
with tab_image:
    uploaded_img = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"], key="img")
    if uploaded_img:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        result = model.predict(frame, conf=conf_threshold, verbose=False)[0]
        vehicle_boxes = get_vehicle_boxes(frame, vehicle_model) if require_vehicle else []
        annotated, counts = draw_boxes(frame.copy(), result, vehicle_boxes, require_vehicle)

        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        render_readout(counts)

# ---------------- VIDEO TAB ----------------
with tab_video:
    uploaded_vid = st.file_uploader("Upload footage", type=["mp4", "mov", "avi"], key="vid")
    st.caption("Runs two models per frame when vehicle-context filtering is on — expect it to take longer on longer clips.")
    if uploaded_vid:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_vid.read())

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        # H.264 via imageio-ffmpeg — cv2's mp4v codec writes files most browsers can't play back.
        writer = imageio.get_writer(
            out_path, format="FFMPEG", mode="I", fps=fps,
            codec="libx264", pixelformat="yuv420p",
        )

        progress = st.progress(0, text="Reviewing footage...")
        total_counts = Counter()
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = model.predict(frame, conf=conf_threshold, verbose=False)[0]
            vehicle_boxes = get_vehicle_boxes(frame, vehicle_model) if require_vehicle else []
            annotated, counts = draw_boxes(frame, result, vehicle_boxes, require_vehicle)
            total_counts.update(counts)
            writer.append_data(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

            frame_idx += 1
            if total_frames:
                progress.progress(min(frame_idx / total_frames, 1.0), text=f"Frame {frame_idx}/{total_frames}")

        cap.release()
        writer.close()
        progress.empty()

        st.video(out_path)
        st.caption("Totals across all frames — a rider appears in many frames, so these are not unique counts.")
        render_readout(total_counts)

        with open(out_path, "rb") as f:
            st.download_button("Download annotated footage", f, file_name="annotated_output.mp4")

# ---------------- WEBCAM TAB ----------------
with tab_webcam:
    st.caption("Live detection from your camera. Allow access when prompted.")

    class HelmetProcessor(VideoProcessorBase):
        def __init__(self):
            self.conf = conf_threshold
            self.require_vehicle = require_vehicle

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            result = model.predict(img, conf=self.conf, verbose=False)[0]
            vehicle_boxes = get_vehicle_boxes(img, vehicle_model) if self.require_vehicle else []
            annotated, _ = draw_boxes(img, result, vehicle_boxes, self.require_vehicle)
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="helmet-webcam",
        video_processor_factory=HelmetProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
    )
