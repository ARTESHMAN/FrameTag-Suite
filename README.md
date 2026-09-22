# FrameTag-Suite

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![GUI](https://img.shields.io/badge/GUI-PySide6-green.svg)](https://pypi.org/project/PySide6/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red.svg)](https://opencv.org/)
[![Tracking](https://img.shields.io/badge/Tracking-ByteTrack%20%7C%20CSRT-orange.svg)](#)

**FrameTag-Suite** is a desktop workstation engineered for high-throughput video annotation, multi-object tracking, and computer vision dataset curation. Developed with Python and PySide6, it pairs responsive manual annotation tools with deep learning detection (YOLO), automated tracking engines (ByteTrack, OpenCV CSRT), keyframe interpolation, and automated dataset quality audits.

---

## Features

* **Interactive Graphics Canvas:** Low-latency 2D bounding box creation with drag-to-draw (Left or Right Click), 8-directional resize handles, boundary clamping, and onion-skin ghosting across adjacent frames.
* **Cadence Hotkey Workflow:** Dedicated keyboard shortcuts (`D` for +1 frame, `F` for +20 frames) and track termination (`E`) for rapid sequential labeling.
* **Hybrid Tracking Engine:**
  * **Automated Batch Tracking:** Integrated ByteTrack algorithm for multi-object association across extended video sequences.
  * **Visual Object Tracking:** Single-object visual tracking via OpenCV CSRT.
  * **Linear Interpolation:** Fast temporal bounding box interpolation between keyframes.
* **AI-Assisted Detection:** In-app YOLO integration for zero-shot object proposals, single-frame auto-tagging, and batch pre-labeling.
* **Dataset Quality Audit (QA):** Built-in diagnostic validator that automatically flags temporal overlaps, ID swaps, zero-area coordinates, and missing frames.
* **Universal Dataset Exporters:** Lossless export pipelines supporting YOLO (`.txt`), COCO (`.json`), Pascal VOC (`.xml`), and MOT Challenge formats.
* **Workspace Persistence:** Project file serialization (`.dlm`), undo/redo command history, and background autosave with crash recovery.

---

## Hotkeys & Shortcuts

| Shortcut | Action | Context |
| :--- | :--- | :--- |
| **Left / Right Click Drag** | Draw new bounding box | Canvas View |
| **D** | Propagate active box(es) forward by **1 frame** and step playhead | Global |
| **F** | Propagate active box(es) forward across **20 frames** and seek | Global |
| **E** | Terminate active track at current frame (purges downstream track history) | Active Track |
| **I** | Interpolate boxes between previous and current keyframe | Active Track |
| **Return / Enter** | Step-track forward 1 frame using CSRT | Active Track |
| **X** | Toggle occluded state on active box | Active Box |
| **O** | Toggle outside / out-of-frame state | Active Box |
| **Delete / Backspace** | Delete active box on current frame | Canvas View |
| **Shift + Delete** | Permanently delete entire track across all frames | Global |
| **Space** | Play / Pause video playback | Global |
| **Right / Left Arrow** | Step forward / backward by 1 frame | Global |
| **Shift + Right / Left** | Jump forward / backward by 10 frames | Global |
| **Alt + Right / Left** | Jump to next / previous keyframe | Global |
| **Ctrl + Z / Ctrl + Y** | Undo / Redo annotation command | Global |
| **Shift + F** | Fit canvas to window | Canvas View |
| **1** | Reset canvas to 1:1 native pixel scale | Canvas View |

---

## Installation & Setup

### Prerequisites
* Python 3.10, 3.11, or 3.12
* A CUDA-capable GPU is recommended for accelerated YOLO inference (CPU execution is fully supported).

### Installation Steps

```bash
# Clone the repository
git clone [https://github.com/ARTESHMAN/FrameTag-Suite.git](https://github.com/ARTESHMAN/FrameTag-Suite.git)
cd FrameTag-Suite

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux / macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### Launch

```bash
python app.py

```

---

## Project Structure

```text
FrameTag-Suite/
├── app.py                     # Application entry point
├── requirements.txt           # Python dependencies
├── assets/
│   └── style.qss              # Workstation dark theme stylesheet
├── config/
│   └── default_config.json    # Default app settings and layout state
├── core/
│   ├── annotation_manager.py  # Central frame-indexed annotation storage
│   ├── annotation_models.py   # Data models (Annotation, Track, Keyframe)
│   ├── byte_track.py          # ByteTrack multi-object tracking implementation
│   ├── cadence.py             # Hotkey propagation and cadence algorithms
│   ├── detector.py            # YOLO model loading, inference, and NMS
│   ├── history_manager.py     # Undo/Redo command stack (Command Pattern)
│   ├── interpolation.py       # Linear coordinate interpolation engine
│   ├── project_manager.py     # Project file serialization (.dlm) and autosave
│   ├── track_manager.py       # Track lifecycle, ID assignment, and track merging
│   ├── tracker.py             # OpenCV visual tracking wrappers (CSRT)
│   └── video_thread.py        # Multithreaded frame decoding and LRU caching
├── gui/
│   ├── canvas.py              # Main interactive graphics view
│   ├── canvas_items.py        # AnnotationBBoxItem and interactive handles
│   ├── class_panel.py         # Category catalog management and palette
│   ├── dialogs.py             # Settings, track split/merge, interval purge
│   ├── export_dialog.py       # Exporter configuration dialog
│   ├── main_window.py         # Shell window coordinating panels and menus
│   ├── model_panel.py         # AI detection and batch tracking interface
│   ├── object_panel.py        # Active track listing and visibility toggles
│   ├── properties_panel.py    # Per-box attribute inspector
│   ├── timeline.py            # Playhead scrubber with keyframe markers
│   ├── track_lanes.py         # Collapsible multi-track timeline visualization
│   └── validation_dialog.py   # Dataset QA diagnostic interface
└── utils/
    ├── coco_io.py             # COCO JSON importer/exporter
    ├── dataset_validator.py   # Structural and spatial anomaly checker
    ├── mot_io.py              # MOT Challenge format parser
    ├── voc_io.py              # Pascal VOC XML generator
    └── yolo_io.py             # Ultralytics / Darknet YOLO format handler

```

---

## Supported Formats

### Media Input

* **Video Formats:** `.mp4`, `.avi`, `.mkv`, `.mov`, `.webm`
* **Image Sequences:** `.png`, `.jpg`, `.jpeg`

### Dataset Exporters

* **YOLO Format:** `<class_id> <x_center> <y_center> <width> <height>` (normalized `[0.0, 1.0]`)
* **COCO JSON:** Instance annotation schema with bounding box sequences
* **Pascal VOC XML:** Coordinates (`xmin`, `ymin`, `xmax`, `ymax`) with category metadata
* **MOT Challenge:** `<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>`

---
