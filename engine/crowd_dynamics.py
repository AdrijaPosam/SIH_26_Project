"""
NSG Tactical AI Command Engine - Crowd Dynamics & Tactical Panic / Ambush Detector
Monitors crowd density surges and sudden divergent velocity dispersal (Panic / Blast / Ambush).
"""
import numpy as np
from typing import List, Dict, Tuple
from collections import deque


class CrowdDynamicsEngine:
    def __init__(self, history_len: int = 15):
        self.density_history = deque(maxlen=history_len)
        self.velocity_history = deque(maxlen=history_len)

    def reset_state(self):
        self.density_history.clear()
        self.velocity_history.clear()

    def analyze(
        self,
        person_vectors: List[Tuple[int, Tuple[float, float], Tuple[float, float]]],  # (track_id, (vx, vy), (cx, cy))
        video_time_label: str
    ) -> List[Dict]:
        alerts = []
        count = len(person_vectors)
        self.density_history.append(count)

        if count < 3:
            return alerts

        velocities = np.array([v for _, (v), _ in person_vectors])
        speeds = np.linalg.norm(velocities, axis=1)
        avg_speed = float(np.mean(speeds))

        # Vector divergence (Variance of angles)
        angles = np.arctan2(velocities[:, 1], velocities[:, 0])
        angle_variance = float(np.var(angles))

        # Sudden high speed + high angular dispersal => Panic / Dispersal
        if avg_speed > 0.035 and angle_variance > 1.8:
            alerts.append({
                "type": "CROWD_DISPERSAL_PANIC",
                "track_id": 0,
                "timestamp": video_time_label,
                "severity": "HIGH",
                "message": f"[{video_time_label}] TACTICAL AMBUSH / PANIC ALERT: Sudden crowd dispersal detected (Speed: {avg_speed:.3f}, Angular Div: {angle_variance:.2f})"
            })

        # Sudden density surge
        if len(self.density_history) >= 8:
            prev_avg_density = np.mean(list(self.density_history)[:4])
            curr_avg_density = np.mean(list(self.density_history)[-4:])
            if curr_avg_density >= prev_avg_density * 2.5 and curr_avg_density >= 6:
                alerts.append({
                    "type": "CROWD_SURGE",
                    "track_id": 0,
                    "timestamp": video_time_label,
                    "severity": "MEDIUM",
                    "message": f"[{video_time_label}] SECTOR ADVISORY: Rapid crowd density surge detected (Count: {int(curr_avg_density)})"
                })

        return alerts
