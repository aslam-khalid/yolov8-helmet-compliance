# 🦺 Helmet Compliance Inspector — YOLOv8

**AIRI Team, PITB — AI Internship**
Individual project by **Muhammad Aslam Khalid**

[![Live Demo](https://img.shields.io/badge/demo-streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://yolov8-helmet-compliance-t9kdjwgf2bdsbjbifpxijj.streamlit.app)
[![Model](https://img.shields.io/badge/model-YOLOv8s-blueviolet)](#model)
[![Python](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](#tech-stack)

---

## Problem

Motorcycle riders not wearing helmets are a major cause of preventable road injuries, and manual enforcement doesn't scale — a traffic officer can watch one intersection, not a whole city. This project builds an automated visual inspector that flags helmet non-compliance from a photo, a video clip, or a live camera feed, giving road-safety teams a scalable first pass at monitoring.

## What it does

- Detects three things in a frame: **Person**, **Helmet**, **No helmet**
- Runs on images, uploaded footage, and live webcam
- Filters detections through a **vehicle-context check** — a rider is only flagged if there's an actual motorcycle/bicycle in frame, so the model doesn't mistake a bare head in an office photo for a violation
- Surfaces results as a live readout (per-class counts + a pass/fail status banner), not just raw bounding boxes

## Live demo

👉 **[Try it here](https://yolov8-helmet-compliance-cjbmvmbwn3clzpvbqournh.streamlit.app/)**

## Dataset

Built by merging three Roboflow Universe sources (helmet/no-helmet imagery) into a single YOLO-format dataset, with every source's class IDs remapped by **name**, not index — an early version of the merge trusted class-index order across datasets, which silently swapped "Person" and "No helmet" labels for ~1,200 boxes. Fixed by matching class names against a canonical scheme before merging.

| Split | Images | Boxes | Person | Helmet | No helmet |
|---|---|---|---|---|---|
| Train | ~1,500 | 4,913 | 2.2% | 73.4% | 24.4% |
| Valid | ~310 | 935 | 1.0% | 78.4% | 20.6% |
| Test | ~95 | 348 | 8.6% | 73.6% | 17.8% |

No-helmet started at ~2% of boxes across the merged dataset — a class the model would have essentially ignored. Brought up to ~18–24% by re-checking a mislabeled "head" class in one source dataset and adding a fourth no-helmet-focused dataset, rather than by duplicating existing images.

## Model

- **Architecture:** YOLOv8s (Ultralytics), fine-tuned from COCO-pretrained weights
- **Classes:** `0: Person`, `1: Helmet`, `2: No helmet`
- **Checkpoint:** `best.pt` (22.5 MB) — trained after the dataset/class-remap fix
- **Context filter:** a second, off-the-shelf YOLOv8n (COCO) checks for `motorcycle`/`bicycle` in frame; helmet-status detections are only kept if they fall near a detected vehicle, cutting false positives on non-traffic imagery without retraining the primary model

## Tech stack

| Layer | Tools |
|---|---|
| Model training | PyTorch, Ultralytics YOLOv8, Google Colab (GPU) |
| Data pipeline | Python, Roboflow-format datasets, custom class-remap merge script |
| App / inference | Streamlit, OpenCV, streamlit-webrtc (live camera) |
| Video encoding | imageio + ffmpeg (H.264, for browser-playable output) |
| Deployment | Streamlit Community Cloud |

## Repo structure

```
yolov8-helmet-compliance/
├── app.py                 # Streamlit app (image / video / webcam inference)
├── requirements.txt
├── best.pt                # trained YOLOv8s weights
├── CV_MODEL.ipynb         # full training notebook (Colab, visible outputs)
├── notebooks/              # supporting training notebooks
└── README.md
```

## Run it locally

```bash
git clone https://github.com/aslam-khalid/yolov8-helmet-compliance.git
cd yolov8-helmet-compliance
pip install -r requirements.txt
streamlit run app.py
```

## Known limitations

- **Person imbalance:** two of the three source datasets never labeled a "Person" class, so person-detection is trained on a narrower slice of the data than helmet/no-helmet — usable, but weaker.
- **No occlusion handling:** heavily obscured or overlapping riders (e.g. three-up on one bike) can under-count.
- **CPU inference:** live webcam detection runs two models per frame (helmet detector + vehicle-context filter) with no GPU in the deployed environment, so there's a visible lag on live feed.

## Credits

Built by Muhammad Aslam Khalid as an individual deliverable for the AIRI/PITB AI internship.
