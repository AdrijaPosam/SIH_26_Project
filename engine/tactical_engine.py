"""
NSG Tactical AI Command Engine - Master Analytics Orchestrator
Integrates YOLO + ByteTrack, Geofencing, Kinematic ML, Unattended IED Detector, Crowd Dynamics, and Thermal Vision.
"""
import cv2
import time
import torch
import numpy as np
from typing import List, Dict, Tuple
from shapely.geometry import Point
from ultralytics import YOLO

from .geofence import GeofenceManager, TacticalZone
from .kinematics import KinematicFeatureExtractor, TargetKinematicProfile
from .unattended_baggage import UnattendedBaggageDetector
from .crowd_dynamics import CrowdDynamicsEngine
from .thermal_vision import ThermalVisionEngine

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TacticalAnalyticsEngine:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.device = DEVICE
        print(f"[NSG ENGINE] Initializing YOLO model ({model_path}) on {self.device.upper()}...")
        self.detector = YOLO(model_path)

        # Core Subsystems
        self.geofence_mgr = GeofenceManager()
        self.kinematics_mgr = KinematicFeatureExtractor()
        self.baggage_mgr = UnattendedBaggageDetector(proximity_threshold=0.16, stationary_time_sec=5.0)
        self.crowd_mgr = CrowdDynamicsEngine()
        self.thermal_mgr = ThermalVisionEngine()

        self.incident_history: List[Dict] = []
        self.latest_profiles: List[Dict] = []

    def reset_state(self):
        self.kinematics_mgr.reset_state()
        self.baggage_mgr.reset_state()
        self.crowd_mgr.reset_state()
        self.incident_history.clear()
        self.latest_profiles.clear()

    def set_vision_mode(self, mode: str):
        self.thermal_mgr.set_mode(mode)

    def set_custom_zone(self, shape_type: str, points: List[List[float]], radius: float = 0.1, name: str = "Tactical Perimeter"):
        self.geofence_mgr.set_single_zone(shape_type, points, radius, name)

    def clear_zones(self):
        self.geofence_mgr.clear_all()

    def process_frame(self, frame: np.ndarray, video_time_sec: float) -> Tuple[np.ndarray, List[Dict], List[Dict]]:
        h, w = frame.shape[:2]
        profiles = []
        alerts = []
        time_label = time.strftime("%M:%S", time.gmtime(video_time_sec))

        # 1. Optical/Thermal pre-filter if desired
        display_frame = frame.copy()

        # 2. YOLO Object Detection & Multi-Object Tracking (ByteTrack)
        results = self.detector.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            imgsz=480,
            device=self.device
        )

        detected_bags = []
        detected_persons = []
        person_vectors = []

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            cls_ids = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist() if results[0].boxes.conf is not None else [1.0] * len(track_ids)
            total_detected = len(track_ids)

            feature_batch = []
            meta_batch = []

            for box, track_id, cls_id, conf in zip(boxes, track_ids, cls_ids, confidences):
                cls_name = self.detector.names.get(cls_id, f"obj_{cls_id}")
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                cx_norm, cy_norm = cx / w, cy / h
                curr_pt = Point(cx_norm, cy_norm)

                # Kinematic Feature Extraction
                speed, accel, dir_change, dist, dwell_time, stat_dur, prev_pt = self.kinematics_mgr.extract(
                    track_id, (cx_norm, cy_norm), video_time_sec
                )

                # Geofence Perimeter Breach Check
                breached_zones = self.geofence_mgr.check_all_breaches(curr_pt, prev_pt)
                in_zone = 1 if len(breached_zones) > 0 else 0

                feature_batch.append([speed, accel, dir_change, dist, dwell_time, stat_dur, total_detected])
                meta_batch.append((track_id, cls_name, box, in_zone, dwell_time, stat_dur, total_detected))

                # Track categorization for specialized detectors
                if cls_name.lower() in {"backpack", "suitcase", "handbag", "bag"}:
                    detected_bags.append((track_id, cls_name, (x1, y1, x2, y2), (cx_norm, cy_norm)))
                elif cls_name.lower() == "person":
                    detected_persons.append((track_id, (cx_norm, cy_norm)))
                    vx = (cx_norm - prev_pt.x) if prev_pt else 0.0
                    vy = (cy_norm - prev_pt.y) if prev_pt else 0.0
                    person_vectors.append((track_id, (vx, vy), (cx_norm, cy_norm)))

                # Geofence Breach Alert Generation
                for z in breached_zones:
                    alerts.append({
                        "type": z.alert_type,
                        "track_id": int(track_id),
                        "timestamp": time_label,
                        "severity": "CRITICAL",
                        "message": f"[{time_label}] CRITICAL: {z.alert_type} in {z.name} by Target #{track_id} ({cls_name})"
                    })

            # Kinematic ML Anomaly Analysis
            if feature_batch:
                ml_profiles, ml_alerts = self.kinematics_mgr.analyze_batch(feature_batch, meta_batch, time_label)
                profiles.extend(ml_profiles)
                alerts.extend(ml_alerts)

        # 3. Unattended Baggage / IED Threat Detection
        bag_alerts = self.baggage_mgr.update(detected_bags, detected_persons, video_time_sec, time_label)
        alerts.extend(bag_alerts)

        # 4. Crowd Dynamics Analysis (Surge / Panic Ambush)
        crowd_alerts = self.crowd_mgr.analyze(person_vectors, time_label)
        alerts.extend(crowd_alerts)

        # 5. Render Geofence Polygons & Visual Overlays
        display_frame = self.geofence_mgr.render_zones_on_frame(display_frame)

        # 6. Render Target Bounding Boxes & HUD telemetry
        for p in profiles:
            t_id = p["track_id"]
            risk_level = p["risk_level"]
            risk_score = p["risk_score"]
            in_zone = p["in_restricted_zone"]

            # Find matching box
            for meta in meta_batch:
                if meta[0] == t_id:
                    box = meta[2]
                    x1, y1, x2, y2 = map(int, box)

                    # Tactical Color Mapping
                    if in_zone or risk_level == "CRITICAL":
                        color = (0, 0, 255)  # Red
                    elif risk_level == "HIGH":
                        color = (0, 140, 255)  # Orange / Amber
                    elif risk_level == "MEDIUM":
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (0, 255, 0)  # Green

                    # Draw tactical corner brackets
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # HUD Tag
                    tag = f"#{t_id} {p['cls_name']} | {'BREACH' if in_zone else f'{risk_level} ({risk_score:.0f}%)'}"
                    cv2.putText(display_frame, tag, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 2)
                    break

        # 7. Apply Multi-Spectral Thermal / Night-Vision filter if enabled
        display_frame = self.thermal_mgr.process(display_frame)

        # Keep state
        self.latest_profiles = profiles
        for a in alerts:
            self.incident_history.append(a)
            if len(self.incident_history) > 100:
                self.incident_history.pop(0)

        return display_frame, profiles, alerts
