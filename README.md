# Helmet Detection System — YOLOv8

AIRI Team PITB — AI Internship Task 1
Individual project by MAK

## Problem
Detects `helmet` and `no-helmet` in workplace/road-safety images to support basic
safety monitoring — flagging people not wearing helmets.

## Dataset
- Total images: TBD (target 150–300)
- Classes: helmet, no-helmet
- Sources: Roboflow Universe ("Helmet dataset" 292 imgs, "Helmet Detection using YOLOv8" 430 imgs)
- Split: 70% train / 20% val / 10% test
- Annotation tool: TBD (Roboflow Annotate / CVAT / LabelImg)

## Model
- YOLOv8n, pretrained weights
- Epochs: 30 (min)
- Image size: 640
- Platform: Google Colab

## Results
| Metric | Value |
|---|---|
| Precision | TBD |
| Recall | TBD |
| mAP@0.5 | TBD |
| mAP@0.5:0.95 | TBD |

## Repo structure
```
cv_project/
├── dataset/
├── notebooks/
├── outputs/
├── models/
├── report/
├── README.md
└── requirements.txt
```

## Error analysis
See `report/final_report.pdf` for the full error-analysis table and improvement suggestions.

## Demo
TBD (Streamlit / FastAPI / video inference)
