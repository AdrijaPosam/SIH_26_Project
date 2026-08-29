"""
NSG Tactical AI Command Engine - Synthetic Tactical Feed Simulator
Generates high-fidelity simulated feeds for Drone UAV, Ground UGV, Bodycam, and Perimeter CCTVs.
"""
import cv2
import time
import math
import numpy as np
from typing import Tuple


class SyntheticFeedGenerator:
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.frame_idx = 0

    def generate_frame(self, feed_type: str = "DRONE_UAV") -> Tuple[np.ndarray, float]:
        self.frame_idx += 1
        t = self.frame_idx * 0.04  # ~25 FPS
        w, h = self.width, self.height

        # Base Tactical Background
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        if feed_type == "DRONE_UAV":
            # Tactical Aerial Drone Recon Perspective (Top-Down compound)
            frame[:] = (35, 45, 30)  # Dark terrain olive

            # Road / Runway grid
            cv2.rectangle(frame, (0, int(h * 0.45)), (w, int(h * 0.65)), (55, 60, 55), -1)
            cv2.line(frame, (0, int(h * 0.55)), (w, int(h * 0.55)), (180, 180, 180), 2)

            # Building compound
            cv2.rectangle(frame, (int(w * 0.15), int(h * 0.15)), (int(w * 0.45), int(h * 0.40)), (50, 50, 60), -1)
            cv2.rectangle(frame, (int(w * 0.15), int(h * 0.15)), (int(w * 0.45), int(h * 0.40)), (90, 90, 100), 2)
            cv2.putText(frame, "HANGAR ALPHA", (int(w * 0.17), int(h * 0.28)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            # Moving Vehicle (Patrol truck)
            veh_x = int((w * 0.1) + (w * 0.8) * (0.5 + 0.5 * math.sin(t * 0.4)))
            veh_y = int(h * 0.55)
            cv2.rectangle(frame, (veh_x - 35, veh_y - 20), (veh_x + 35, veh_y + 20), (40, 70, 110), -1)
            cv2.rectangle(frame, (veh_x - 15, veh_y - 12), (veh_x + 15, veh_y + 12), (60, 90, 130), -1)

            # Moving Pedestrians / Suspects
            p1_x = int(w * 0.35 + 120 * math.sin(t * 0.6))
            p1_y = int(h * 0.30 + 60 * math.cos(t * 0.6))
            cv2.circle(frame, (p1_x, p1_y), 10, (220, 220, 220), -1)
            cv2.circle(frame, (p1_x, p1_y), 4, (120, 120, 120), -1)

            # High-speed sprinting suspect
            p2_x = int(w * 0.60 + 200 * math.sin(t * 1.5))
            p2_y = int(h * 0.70 + 80 * math.cos(t * 1.5))
            cv2.circle(frame, (p2_x, p2_y), 10, (0, 0, 220), -1)

            # Unattended Backpack
            bag_x, bag_y = int(w * 0.50), int(h * 0.35)
            cv2.rectangle(frame, (bag_x - 6, bag_y - 6), (bag_x + 6, bag_y + 6), (0, 165, 255), -1)

            # Drone HUD Reticle
            cx, cy = w // 2, h // 2
            cv2.circle(frame, (cx, cy), 80, (0, 255, 120), 1)
            cv2.line(frame, (cx - 100, cy), (cx + 100, cy), (0, 255, 120), 1)
            cv2.line(frame, (cx, cy - 100), (cx, cy + 100), (0, 255, 120), 1)
            cv2.putText(frame, f"ALT: 120m | AZIMUTH: {int((t * 20) % 360)} | UAV-ALPHA", (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1)

        elif feed_type == "ROBOT_UGV":
            # Ground Recon Robot Perspective (Corridor indoor)
            frame[:] = (20, 20, 25)
            # Perspective corridor lines
            cv2.line(frame, (0, 0), (int(w * 0.35), int(h * 0.5)), (80, 80, 80), 2)
            cv2.line(frame, (w, 0), (int(w * 0.65), int(h * 0.5)), (80, 80, 80), 2)
            cv2.line(frame, (0, h), (int(w * 0.35), int(h * 0.5)), (100, 100, 100), 2)
            cv2.line(frame, (w, h), (int(w * 0.65), int(h * 0.5)), (100, 100, 100), 2)

            # Target ahead
            target_scale = 0.8 + 0.3 * math.sin(t * 0.5)
            tx, ty = int(w * 0.5 + 40 * math.sin(t * 0.8)), int(h * 0.55)
            th, tw = int(120 * target_scale), int(40 * target_scale)
            cv2.rectangle(frame, (tx - tw // 2, ty - th), (tx + tw // 2, ty), (180, 150, 120), -1)
            cv2.circle(frame, (tx, ty - th - 12), int(14 * target_scale), (200, 180, 150), -1)

            # UGV Radar scanline
            scan_y = int(h * 0.5 + (h * 0.45) * ((t * 0.8) % 1.0))
            cv2.line(frame, (int(w * 0.2), scan_y), (int(w * 0.8), scan_y), (0, 255, 255), 1)
            cv2.putText(frame, f"UGV-BRAVO | LIDAR PROX: {2.4 + math.sin(t):.1f}m | TILT: -4.2", (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        elif feed_type == "BODYCAM":
            # Dynamic Helmet/Bodycam view with tactical sway
            sway_x = int(15 * math.sin(t * 2.0))
            sway_y = int(10 * math.cos(t * 4.0))

            frame[:] = (30, 25, 25)
            # Wall corner
            cv2.line(frame, (int(w * 0.4 + sway_x), 0), (int(w * 0.4 + sway_x), h), (70, 70, 70), 3)

            # Suspect moving past doorway
            suspect_x = int(w * 0.7 + sway_x + 80 * math.sin(t * 1.2))
            cv2.rectangle(frame, (suspect_x - 30, int(h * 0.3 + sway_y)), (suspect_x + 30, int(h * 0.85 + sway_y)), (160, 120, 100), -1)
            cv2.circle(frame, (suspect_x, int(h * 0.24 + sway_y)), 22, (200, 170, 140), -1)

            # Tactical Helmet HUD Reticle
            cv2.drawMarker(frame, (w // 2 + sway_x, h // 2 + sway_y), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
            cv2.putText(frame, "NSG HIT-TEAM-1 | BODYCAM-ALPHA | LIVE ENCRYPTED", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 150), 1)

        else:
            # Fixed Perimeter CCTV HQ
            frame[:] = (25, 30, 35)
            # Checkpoint barrier
            cv2.line(frame, (int(w * 0.1), int(h * 0.6)), (int(w * 0.9), int(h * 0.6)), (120, 120, 120), 4)

            # Walking pedestrians
            px = int(w * 0.2 + (w * 0.6) * ((t * 0.2) % 1.0))
            cv2.rectangle(frame, (px - 20, int(h * 0.35)), (px + 20, int(h * 0.75)), (190, 190, 190), -1)
            cv2.circle(frame, (px, int(h * 0.30)), 16, (210, 180, 150), -1)
            cv2.putText(frame, "CCTV-PERIMETER-GATE-02 | 1080P 60HZ", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame, t
