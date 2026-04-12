# 🕌 Mosque Crowd Intelligence System
### Enterprise-Grade Real-Time AI Crowd Monitoring Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/PyTorch-GPU%20%2F%20CPU-red?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/OpenCV-RealTime-green?style=for-the-badge&logo=opencv" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

---

## 📌 What Is This?

A production-ready AI surveillance system built specifically for **mosques and large venues**.  
It detects, counts, and tracks every worshipper in real time — even in dense prayer rows — using a multi-model fusion pipeline with no missed people.

> Built from scratch as a personal R&D project. No dataset. No labeled data. Just engineering.

---

## 🎥 Demo

> *(![Uploading demo.gif…]()
)*  
> Example: `![Demo](demo.gif)`

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Multi-Model Fusion** | YOLOv8 body + YOLOv8-pose head + MobileNetV3 CNN regressor |
| **Sliced Inference** | SAHI-style 2×2 tiling detects distant/small people |
| **Temporal Head Tracking** | EMA smoothing + velocity prediction — head never lost |
| **Interactive Gate System** | Click+drag to draw doors, counts per-gate entry/exit |
| **Transformer Forecaster** | Predicts crowd in 5s / 30s / 60s / 5min ahead |
| **Online Learning** | CNN fine-tunes itself in background during runtime |
| **6-Thread Async Pipeline** | Video / Detection / Analytics / Learner / Retrainer / UI — zero lag |
| **GPU FP16 Support** | 2× faster on CUDA with half-precision inference |
| **LANCZOS4 Zoom + Enhance** | Face-level clarity at high zoom (CLAHE + unsharp mask) |
| **GUI Launcher** | Tkinter file picker — no config editing needed |
| **REST API** | FastAPI `/api/status` endpoint for dashboard integration |
| **Session Reports** | PDF + TXT with hourly breakdown, peak analysis, gate stats |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   6-Thread Pipeline                     │
├──────────────┬──────────────────┬───────────────────────┤
│ VideoPlayer  │ DetectionWorker  │  AnalyticsThread      │
│ exact-FPS    │ YOLO + Head +    │  Drift / Forecast /   │
│ independent  │ Fusion + Gates   │  Anomaly / CSV        │
├──────────────┴──────────────────┴───────────────────────┤
│  OnlineLearner   │  AutoRetrainer   │  Main Thread (UI)  │
│  CNN fine-tune   │  Nightly retrain │  Display only      │
└──────────────────────────────────────────────────────────┘

Detection Stack:
  Body  →  YOLOv8n/x + ByteTrack + EMA Box Smoothing
  Head  →  YOLOv8n-pose + Velocity-aware Ghost Persistence
  CNN   →  MobileNetV3 128-neuron regressor (HuberLoss)
  Fuse  →  Confidence-weighted body + head + density
```

---

## 🚀 Quick Start

### 1. Install Requirements

```bash
pip install ultralytics torch torchvision opencv-python pillow numpy
pip install fastapi uvicorn   # optional — for REST API
```

### 2. (Optional) Install ffmpeg for audio

```bash
# Windows
winget install ffmpeg

# Linux
sudo apt install ffmpeg
```

### 3. Run

```bash
python crowd_master.py
```

A GUI window will open — browse to your video file and click **Start Analysis**.

---

## ⌨️ Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `P` | Pause / Resume |
| `G` | Gate draw mode (click+drag on video) |
| `TAB` | Select next gate |
| `R` | Reverse selected gate direction |
| `X` | Delete selected gate (or clear all) |
| `I` | Toggle image enhance (CLAHE + sharpen) |
| `D` | Toggle detection on/off |
| `]/[` | Zoom in / out |
| `WASD` / Arrows | Pan |
| `+/-` | Playback speed |
| `V/C` | Volume up / down |
| `. ,` | Seek +5s / -5s |
| `N` | Toggle stats panel |
| `B H Z M` | Toggle boxes / heads / zones / heatmap |
| `F` | Fullscreen |
| `T` | TTA toggle |
| `Ctrl+R` | Force retrain CNN |

---

## ⚙️ Configuration

Edit `Config` class at the top of `crowd_master.py`:

```python
# Change video (or use the GUI)
VIDEO_PATH = r"C:\path\to\your\video.mp4"

# Switch model for more accuracy (slower)
BODY_MODEL = "yolov8x.pt"   # n=fast, s/m/l/x=accurate

# Crowd alert threshold
CROWD_THRESHOLD = 15

# Detection frequency
DETECT_EVERY = 2   # run detection every N frames
```

Or use environment variables (no code editing):
```bash
set CROWD_MASTER_VIDEO=C:\videos\mosque.mp4
set CROWD_MASTER_DEVICE=cuda
```

---

## 📁 Output Files

All saved to `DATA/` folder next to the script:

| File | Content |
|------|---------|
| `crowd_log.csv` | Per-frame detection log |
| `crowd_report.txt` | Session summary with peak analysis |
| `crowd_report.pdf` | Visual PDF report with charts |
| `drift_events.csv` | Scene change log |
| `gates.json` | Saved gate positions (auto-reloaded) |
| `best_crowd_model.pth` | Fine-tuned CNN checkpoint |

---

## 🧰 Tech Stack

```
Python 3.9+     → Core language
YOLOv8          → Body + Pose detection (Ultralytics)
PyTorch         → CNN regressor + Transformer forecaster
OpenCV          → Video capture + display pipeline
ByteTrack       → Multi-object tracker
MobileNetV3     → Lightweight CNN crowd regressor
FastAPI         → REST API endpoint
Tkinter         → GUI video launcher
ffplay (ffmpeg) → Audio playback
```

---

## 📊 System Capabilities

- ✅ Detects people in dense prayer rows
- ✅ Counts partially visible / occluded worshippers via head tracking
- ✅ Per-door entry/exit counting with configurable direction
- ✅ Predicts crowd surge before it happens
- ✅ Detects panic, reverse flow, chaotic motion via optical flow
- ✅ Runs on CPU (30ms/frame with YOLOv8n) or GPU (< 5ms with FP16)
- ✅ Generates full session report on exit

---

## 🎯 Potential Use Cases

- **Ministry of Awqaf** — mosque capacity monitoring
- **Hajj / Umrah management** — pilgrimage crowd safety
- **Event venues** — stadium / arena crowd control
- **Smart city** — public space density analytics

---

## 👤 Author

**Youssef Ayman Kabbary**  
Computer Science Graduate — AI & Robotics  
Pharos University in Alexandria  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/youssef-kabbary-a989742a6)
[![Email](https://img.shields.io/badge/Email-Contact-red?style=flat&logo=gmail)](mailto:Youssefkabbary1152003@gmail.com)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

```
MIT License
Copyright (c) 2025 Youssef Ayman Kabbary
```

---

> *"Built not for a grade. Built because the idea came to mind and I wanted to make it real."*
