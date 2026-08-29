"""
NSG Tactical AI Command Engine - FastAPI Command & Control Hub
Real-time multi-feed processing, WebSockets binary streaming, REST API, SITREP generation.
"""
import os
import cv2
import json
import time
import queue
import asyncio
import threading
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

import torch
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI

from engine.tactical_engine import TacticalAnalyticsEngine
from engine.feed_simulator import SyntheticFeedGenerator
from engine.sitrep import SITREPGenerator
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NSG Surveillance AI Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://sih-26-project-65p6tkqzc-small-potato-stuff.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Directories
SAMPLE_DIR = Path("sample_videos")
ANOMALY_DIR = Path("Anomaly-Videos")
UPLOAD_DIR = Path("uploaded_videos")
UPLOAD_DIR.mkdir(exist_ok=True)
SAMPLE_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Instantiate Master Tactical Engine
engine = TacticalAnalyticsEngine(model_path="yolov8n.pt")
feed_sim = SyntheticFeedGenerator()

def numpy_json_serializer(obj):
    """Serializes numpy primitives safely to JSON."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


class ZoneUpdatePayload(BaseModel):
    shape_type: str = "POLYGON"  # "POLYGON", "TRIPWIRE", "CIRCLE"
    points: List[List[float]]     # Normalized coordinates [[x, y], ...]
    radius: float = 0.1          # For CIRCLE
    name: str = "Tactical Perimeter"


class VisionModePayload(BaseModel):
    mode: str = "RGB"  # "RGB", "FLIR_IRONBOW", "WHITE_HOT", "NIGHT_VISION"


def video_stream_worker(
    source_type: str,
    source_path: Optional[str],
    synthetic_type: str,
    frame_queue: queue.Queue,
    stop_event: threading.Event,
    playback_speed: float = 1.0
):
    """Background worker for frame decoding, AI inference, and queuing."""
    if source_type == "SYNTHETIC":
        frame_idx = 0
        fps = 25.0
        while not stop_event.is_set():
            frame, video_time_sec = feed_sim.generate_frame(synthetic_type)
            frame_idx += 1

            processed_frame, profiles, alerts = engine.process_frame(frame, video_time_sec)
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()

            payload = {
                "status": "STREAMING",
                "video_time": round(video_time_sec, 2),
                "frame_progress": f"{frame_idx}/INF",
                "profiles": profiles,
                "alerts": alerts,
                "feed_name": f"SIM-{synthetic_type}"
            }

            while not stop_event.is_set():
                try:
                    frame_queue.put((payload, frame_bytes), timeout=0.08)
                    break
                except queue.Full:
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass

            time.sleep(1.0 / (fps * playback_speed))

    else:
        if not source_path or not Path(source_path).exists():
            frame_queue.put({"status": "ERROR", "message": f"Source file not found: {source_path}"})
            return

        cap = cv2.VideoCapture(str(source_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while not stop_event.is_set() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                frame_queue.put({"status": "COMPLETED"})
                break

            current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            raw_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            video_time_sec = float((raw_msec / 1000.0) if raw_msec > 0 else (current_frame_idx / fps))

            h, w = frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                frame = cv2.resize(frame, (1280, int(h * scale)), interpolation=cv2.INTER_LINEAR)

            processed_frame, profiles, alerts = engine.process_frame(frame, video_time_sec)
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()

            payload = {
                "status": "STREAMING",
                "video_time": round(video_time_sec, 2),
                "frame_progress": f"{current_frame_idx}/{total_frames}",
                "profiles": profiles,
                "alerts": alerts,
                "feed_name": Path(source_path).name
            }

            while not stop_event.is_set():
                try:
                    frame_queue.put((payload, frame_bytes), timeout=0.08)
                    break
                except queue.Full:
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass

            time.sleep(1.0 / (fps * playback_speed))

        cap.release()


@app.get("/api/feeds")
async def get_available_feeds():
    feeds = {
        "Synthetic Feeds (Tactical Sim)": [
            "DRONE_UAV",
            "ROBOT_UGV",
            "BODYCAM",
            "PERIMETER_CCTV"
        ],
        "Sample Tactical Videos": [],
        "Uploaded Video Archive": [],
        "Anomaly Dataset Videos": {}
    }

    valid_exts = {".mp4", ".avi", ".mkv", ".mov"}

    if SAMPLE_DIR.exists():
        feeds["Sample Tactical Videos"] = [
            f.name for f in sorted(SAMPLE_DIR.iterdir())
            if f.is_file() and f.suffix.lower() in valid_exts
        ]

    if UPLOAD_DIR.exists():
        feeds["Uploaded Video Archive"] = [
            f.name for f in sorted(UPLOAD_DIR.iterdir())
            if f.is_file() and f.suffix.lower() in valid_exts
        ]

    if ANOMALY_DIR.exists():
        for cat_dir in sorted(ANOMALY_DIR.iterdir()):
            if cat_dir.is_dir():
                vids = [
                    f.name for f in sorted(cat_dir.iterdir())
                    if f.is_file() and f.suffix.lower() in valid_exts
                ]
                if vids:
                    feeds["Anomaly Dataset Videos"][cat_dir.name] = vids

    return feeds


@app.post("/api/zone")
async def update_custom_geofence(payload: ZoneUpdatePayload):
    engine.set_custom_zone(payload.shape_type, payload.points, payload.radius, payload.name)
    return {"status": "SUCCESS", "message": f"{payload.shape_type} zone boundary applied."}


@app.get("/api/zones")
async def get_active_zones():
    return engine.geofence_mgr.get_zones_summary()


@app.delete("/api/zones")
async def clear_all_zones():
    engine.clear_zones()
    return {"status": "SUCCESS", "message": "All geofence perimeters cleared."}


@app.post("/api/vision_mode")
async def set_vision_mode(payload: VisionModePayload):
    engine.set_vision_mode(payload.mode)
    return {"status": "SUCCESS", "mode": payload.mode}


@app.get("/api/sitrep")
async def get_tactical_sitrep():
    sitrep = SITREPGenerator.generate(
        operation_name="OPERATION SAGAR DEFENSE - NSG COMMAND",
        icp_node="ICP-DELTA-01 (HQ)",
        sector="SECTOR 4 - PERIMETER WEST",
        active_profiles=engine.latest_profiles,
        incident_log=engine.incident_history,
        active_zones=engine.geofence_mgr.get_zones_summary()
    )
    return sitrep


@app.post("/api/upload")
async def upload_video_feed(file: UploadFile = File(...)):
    dest = UPLOAD_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)
    return {"status": "SUCCESS", "filename": file.filename, "message": f"Saved {file.filename} for tactical analysis."}


@app.get("/api/system_status")
async def get_system_status():
    return {
        "engine": "NSG Tactical AI Command Engine v2.5",
        "device": str(engine.device).upper(),
        "cuda_available": torch.cuda.is_available(),
        "active_vision_mode": engine.thermal_mgr.active_mode,
        "active_zones_count": len(engine.geofence_mgr.zones),
        "tracked_targets_count": len(engine.latest_profiles),
        "total_incidents_recorded": len(engine.incident_history)
    }


@app.websocket("/ws/tactical_stream")
async def tactical_stream_websocket(
    websocket: WebSocket,
    source_type: str = Query("SYNTHETIC"),
    category: str = Query(""),
    filename: str = Query("DRONE_UAV"),
    speed: float = Query(1.0)
):
    await websocket.accept()
    engine.reset_state()

    source_path = None
    synthetic_type = "DRONE_UAV"

    if source_type == "SYNTHETIC":
        synthetic_type = filename.upper()
    elif source_type == "SAMPLE":
        source_path = str(SAMPLE_DIR / filename)
    elif source_type == "UPLOAD":
        source_path = str(UPLOAD_DIR / filename)
    elif source_type == "ANOMALY":
        source_path = str(ANOMALY_DIR / category / filename)

    frame_queue = queue.Queue(maxsize=20)
    stop_event = threading.Event()

    worker = threading.Thread(
        target=video_stream_worker,
        args=(source_type, source_path, synthetic_type, frame_queue, stop_event, speed),
        daemon=True
    )
    worker.start()

    try:
        while True:
            try:
                data = await asyncio.to_thread(frame_queue.get, timeout=1.0)
            except Exception:
                if not worker.is_alive():
                    break
                continue

            if isinstance(data, dict) and data.get("status") in {"COMPLETED", "ERROR"}:
                await websocket.send_text(json.dumps(data, default=numpy_json_serializer))
                break

            payload, frame_bytes = data
            await websocket.send_text(json.dumps(payload, default=numpy_json_serializer))
            await websocket.send_bytes(frame_bytes)

    except WebSocketDisconnect:
        pass
    finally:
        stop_event.set()
        worker.join(timeout=1.0)


@app.get("/")
async def get_command_dashboard():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    print("[NSG COMMAND] Launching Tactical AI Command Engine on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
