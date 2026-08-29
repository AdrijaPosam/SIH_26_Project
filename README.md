# NSG Tactical AI Command Engine (ICP-AI-Sentinel)
### AI & ML Enabled Video Analysis and Interpretation
**Ministry of Home Affairs (MHA) &bull; National Security Guard (NSG) Incident Command Post (ICP)**

---

## 1. Problem Statement
The **National Security Guard (NSG)** operates a diverse fleet of multi-source surveillance assets:
- **Medium-Range Aerial Surveillance Drones (UAVs)** for high-altitude compound overwatch.
- **Small Two-Wheel Ground Reconnaissance Robots (UGVs)** for indoor room breaches, low-angle scans, and corridor sweeps.
- **Operator Tactical Bodycams / Helmet Cams** for first-person dynamic assault & hostage rescue operations.
- **Fixed Perimeter CCTVs & Checkpoint PTZ Cameras** for base security and access control.

In real-world tactical environments, the immense volume of converging video data overwhelms human operators at the **Incident Command Post (ICP)**. Cognitive overload leads to fatigue and missed fleeting threats (e.g., perimeter breaches, concealed movement, loitering scouts, abandoned IEDs, or crowd panic).

---

## 2. Solution Overview
The **NSG Tactical AI Command Engine** is a high-throughput, low-latency, multi-stream tactical video intelligence platform designed to automate surveillance analysis, object detection, and behavioral interpretation with minimal human intervention.

```
[Drone UAVs / UGV Robots / Bodycams / Perimeter CCTV]
                     │
                     ▼
  ┌──────────────────────────────────────────────────────────┐
  │         NSG TACTICAL INGESTION & PIPELINE ENGINE         │
  │                                                          │
  │  1. Multi-Object Detection & ByteTrack Tracking (YOLO)    │
  │  2. Dynamic Spatial Geofencing (Polygon / Tripwire / VIP)│
  │  3. Kinematic Feature Vector Extraction (v, a, θ, d)     │
  │  4. Machine Learning Behavioral Anomaly (IsolationForest)│
  │  5. Unattended Baggage / IED Abandonment Detector        │
  │  6. Crowd Dynamics & Ambush / Panic Dispersal Evaluator  │
  │  7. Multi-Spectral Thermal / Night-Ops Vision Filters    │
  └──────────────────────────────────────────────────────────┘
                     │
         WebSocket Binary Stream (JPEG + Telemetry JSON)
                     │
                     ▼
  ┌──────────────────────────────────────────────────────────┐
  │         INCIDENT COMMAND POST (ICP) TACTICAL HUD         │
  │                                                          │
  │  • Live Optical/Thermal HUD with Crosshair Telemetry     │
  │  • Interactive Freeform Perimeter Drawing Engine        │
  │  • Real-Time Target Kinematics & ML Risk Matrix          │
  │  • Tactical Audio Siren Synthesizer (Web Audio API)      │
  │  • Automated MHA/NSG Standard SITREP Generator (.md)     │
  └──────────────────────────────────────────────────────────┘
```

---

## 3. Key Technical Capabilities

### A. Real-Time Multi-Object Detection & Persistent Tracking
- Powered by Ultralytics YOLOv8 / YOLOv11 with ByteTrack multi-object tracking.
- Tracks `person`, `vehicle`, `truck`, `backpack`, `suitcase`, `handbag`, and drone assets with persistent track IDs, velocity vectors, and bounding box stabilization.

### B. Dynamic Interactive Geofencing & Perimeter Defense
- **N-Point Freeform Polygon**: Draw custom restricted sectors directly on live video.
- **Virtual Directional Tripwire**: 2-point laser crossing detection with directional vector math.
- **Radial VIP Exclusion Buffer**: Defines high-value asset safety cordons.

### C. Pure Kinematic ML Anomaly Detection (Isolation Forest)
Extracts 7-dimensional feature vectors per tracked target:
1. Normalized instantaneous velocity: $v = \frac{\Delta s}{\Delta t}$
2. Instantaneous acceleration: $a = \frac{\Delta v}{\Delta t}$
3. Angular curvature / directional shift: $\Delta \theta = 1 - \cos(\vec{v}_1, \vec{v}_2)$
4. Cumulative distance traveled: $d_{\text{total}}$
5. Sector dwell time: $t_{\text{dwell}}$
6. Stationary / loitering duration: $t_{\text{loiter}}$
7. Local crowd density: $N$

Evaluates kinematics through a pre-calibrated **Isolation Forest** ML model to detect erratic movements, evasive zigzagging, sudden sprints, or collapses without relying on biased appearance features.

### D. Unattended Baggage & Suspicious IED Detector
- Tracks bags, backpacks, and luggage.
- Automatically calculates distance to the closest detected human owner ($R_{\text{owner}}$).
- If stationary and separated from any owner for $> 5$ seconds, flags a high-priority `UNATTENDED_BAGGAGE_IED` alert with a countdown timer.

### E. Crowd Dynamics & Ambush / Panic Dispersal Detector
- Calculates vector divergence across all targets in view.
- High velocity + high angular variance triggers `CROWD_DISPERSAL_PANIC` (indicative of explosions, gunfire, or tactical ambush).

### F. Multi-Spectral Vision Filters
- **Optical (RGB)**: Full-color standard feed.
- **FLIR Ironbow Thermal IR**: Pseudo-thermal infrared colormap with CLAHE contrast enhancement.
- **FLIR White-Hot IR**: Inverted high-contrast thermal mode.
- **Gen-3 Green Phosphor NVG**: Military night-vision simulation with scanline generator.

### G. Automated MHA/NSG Standard Situation Report (SITREP) Generator
- Compiles tactical debriefs with target threat summaries, breach counters, chronological incident logs, and automated tactical directives for NSG Hit Teams.
- Exportable to Markdown (`.md`) and JSON.

---

## 4. Quick Start Guide

### Prerequisites
- Python 3.10+
- PyTorch (CPU or CUDA)
- OpenCV, FastAPI, Uvicorn, Scikit-Learn, Shapely, Ultralytics

### Installation
```bash
cd nsg_tactical_ai
pip install -r requirements.txt
```

### Launching the System
```bash
python app.py
```
Open your browser and navigate to:
**`http://localhost:8000`**

---

## 5. Directory Structure
```
nsg_tactical_ai/
├── app.py                      # FastAPI Backend & WebSocket Ingestion Hub
├── requirements.txt            # System dependencies
├── generate_sample_dataset.py  # Synthetic video sample generator
├── sample_videos/              # Pre-generated tactical MP4 test feeds
├── engine/
│   ├── __init__.py
│   ├── geofence.py             # Shapely Polygon, Tripwire, Circle Geofencing
│   ├── kinematics.py           # Kinematic extractor & Isolation Forest ML
│   ├── unattended_baggage.py   # IED / Abandoned luggage detector
│   ├── crowd_dynamics.py       # Crowd surge & panic dispersal engine
│   ├── thermal_vision.py       # FLIR & Gen-3 NVG thermal filters
│   ├── feed_simulator.py       # Realistic synthetic Drone, UGV, Bodycam feeds
│   ├── sitrep.py               # MHA/NSG Situation Report Generator
│   └── tactical_engine.py      # Master analytics orchestrator
└── static/
    └── index.html              # Cyber-Tactical Command HUD Interface
```
