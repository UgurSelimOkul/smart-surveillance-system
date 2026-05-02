# 🎥 Smart Surveillance System
### AI-Powered Multi-Camera Surveillance Platform

A real-time, multi-threat intelligent surveillance application built entirely in Python. Designed for school, campus, and institutional security, this system runs **6 simultaneous AI analysis threads** on a single machine — each camera independently detecting a different type of security threat using computer vision and deep learning models.

> 👨‍💻 **Designed & Developed by Uğur Selim Okul**
> Hardware Design (PCB) | Embedded Systems | IoT Integrations | Computer Vision

---

## 🖥️ System Overview

This is not a simple motion-detection tool. Each camera channel is powered by its own dedicated AI thread, running a specialized YOLOv8 model trained for a specific threat scenario. The system is designed to run continuously, recover from errors automatically, and alert operators in real time — all through a custom PyQt5 control interface.

---

## 🧠 AI Threat Detection Modules

| Camera | Module | AI Model | Threat Detected |
|--------|--------|----------|-----------------|
| **CAM 1** | Face Recognition | YOLOv8n-Face + DeepFace (Facenet512) | Unauthorized personnel at entry |
| **CAM 2** | Speed / Motion Analysis | YOLOv8n + ByteTrack | Running / high-speed movement inside premises |
| **CAM 3** | Loitering Detection | YOLOv8n + ByteTrack | Individuals waiting beyond a time threshold |
| **CAM 4** | Crowd Density | YOLOv8n + ByteTrack | Abnormal crowd accumulation |
| **CAM 5** | Suspicious Object | YOLOv8m + ByteTrack | Unattended bags and packages |
| **CAM 6** | Illegal Parking | YOLOv8n + ByteTrack | Vehicles stopped beyond a time threshold |

---

## ⚙️ Architecture

```
main.py  ──────────────────────────────────────────────────────
│  PyQt5 GUI (QMainWindow)                                     │
│  ├── Sidebar: Camera on/off controls                         │
│  ├── 3×2 Camera Grid (live feed, auto-scaling)               │
│  └── Real-time Alarm & Event Log (scrolling, 500 entry cap)  │
│                                                               │
video_processor.py  ────────────────────────────────────────── │
│  ├── YuzTanimaThread      (Camera 1 — Face ID)               │
│  ├── HareketAnalizThread  (Camera 2 — Speed)                 │
│  ├── BeklemeAnalizThread  (Camera 3 — Loitering)             │
│  ├── KalabalikAnalizThread(Camera 4 — Crowd)                 │
│  ├── SupheliPaketThread   (Camera 5 — Object)                │
│  └── HataliParkThread     (Camera 6 — Parking)               │
│                                                               │
startup_check.py  ──────────────────────────────────────────── │
│  Dependency validator — runs before UI loads                  │
│  Checks: Python packages / .pt model files / video files     │
│  Shows detailed popup → UI always opens, broken cams degrade │
└───────────────────────────────────────────────────────────────
```

---

## 🛡️ Key Features

- **6 independent AI threads** — one crash does not affect the others
- **Face database system** — authorized personnel enrolled via `Gorevli/` folder; embeddings cached in `face_database.pkl` for fast startup
- **ByteTrack multi-object tracking** — persistent IDs survive brief occlusions
- **Graceful degradation** — missing models or videos show informative messages on-screen instead of crashing
- **Startup dependency checker** — popup dialog on launch lists every missing file with exact install commands
- **Full-screen camera view** — any camera can be expanded to a dedicated window
- **Anomaly border highlight** — active threat cameras glow red via QSS dynamic property
- **Auto video loop** — demo/test videos restart seamlessly when they end
- **GPU acceleration** — automatically uses CUDA if available, falls back to CPU

---

## 📂 Required Project Structure

```
project/
├── main.py
├── video_processor.py
├── startup_check.py
├── style.qss
├── yolov8n.pt                    ← auto-downloaded on first run
├── yolov8m.pt                    ← auto-downloaded on first run
├── yolov8n-face-lindevs.pt       ← manual download required (see below)
├── Gorevli/                      ← authorized personnel photos
│   ├── PersonName1/
│   │   ├── photo1.jpg
│   │   └── photo2.jpg
│   └── PersonName2/
│       └── photo1.jpg
└── Videolar/
    ├── kosma.mp4
    ├── bekleme.mp4
    ├── topluluk.mp4
    ├── esya.mp4
    └── araba.mp4
```

---

## 🚀 Installation & Setup

### 1 — Clone the repository

```bash
git clone https://github.com/UgurSelimOkul/smart-surveillance-system.git
cd smart-surveillance-system
```

### 2 — Install dependencies

```bash
pip install opencv-python numpy torch torchvision torchaudio ultralytics deepface tensorflow PyQt5
```

> ⚡ For GPU support, install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/) instead of the command above.

### 3 — Download YOLO models

`yolov8n.pt` and `yolov8m.pt` are downloaded automatically on first run. To pre-download them manually:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8m.pt')"
```

### 4 — Download the face detection model (manual)

`yolov8n-face-lindevs.pt` must be downloaded separately and placed in the project root:

➡️ [github.com/lindevs/yolov8-face/releases](https://github.com/lindevs/yolov8-face/releases)

### 5 — Add authorized personnel (optional)

Create a subfolder per person inside `Gorevli/` and add their photos (`.jpg` / `.png`).  
The system builds a face embedding database automatically on the first run.

### 6 — Add test videos (optional)

Place your `.mp4` files inside `Videolar/`. Camera 1 uses a live webcam (index `0`) by default.

### 7 — Run

```bash
python main.py
```

---

## 🖥️ Interface Preview

The control panel features a **320px sidebar** for per-camera enable/disable controls and a **3×2 live feed grid** with expandable full-screen view. A scrolling alarm log at the bottom captures all timestamped events.

| Feature | Detail |
|--------|--------|
| Sidebar | Toggle each camera on/off independently at runtime |
| Grid view | 3 columns × 2 rows, all feeds live simultaneously |
| Full-screen | Click ⤢ on any camera to open a dedicated window |
| Alarm log | Real-time timestamped events, capped at 500 entries |
| Threat indicator | Camera border turns red during active anomaly detection |

---

## ⚠️ Notes

- Camera 1 requires a physical webcam connected at index `0`. If unavailable, it will display "SINYAL BEKLENIYOR..."
- The face database (`face_database.pkl`) is rebuilt automatically when new photos are added to `Gorevli/`
- The startup checker runs every launch and informs you of any missing dependencies — the UI always opens regardless

---

## 👨‍💻 Author

**Uğur Selim Okul**
Hardware Design (PCB) · Embedded Systems · IoT Integrations · Computer Vision

[![GitHub](https://img.shields.io/badge/GitHub-UgurSelimOkul-181717?style=flat&logo=github)](https://github.com/UgurSelimOkul)
