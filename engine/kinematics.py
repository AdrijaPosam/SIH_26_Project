"""
NSG Tactical AI Command Engine - Kinematics & Behavioral Machine Learning
Extracts kinematic vectors (Speed, Accel, Curvature, Dwell, Loiter) and applies Isolation Forest ML.
"""
import time
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from shapely.geometry import Point
from sklearn.ensemble import IsolationForest


@dataclass
class TargetKinematicProfile:
    track_id: int
    cls_name: str
    speed_norm: float
    accel_norm: float
    direction_change: float
    distance_traveled: float
    dwell_time: float
    in_restricted_zone: int
    stationary_duration: float
    crowd_density: int
    anomaly_score: float
    risk_score: float
    risk_level: str


class KinematicFeatureExtractor:
    def __init__(self):
        self.history = defaultdict(lambda: deque(maxlen=45))
        self.entry_times = {}
        self.last_stationary_checkpoint = {}
        self.stationary_duration = defaultdict(float)

        # Scikit-Learn Isolation Forest for Kinematic Anomaly Modeling
        self.iso_forest = IsolationForest(
            n_estimators=60,
            contamination=0.06,
            random_state=42,
            n_jobs=-1
        )
        self._warmup_ml_model()

    def reset_state(self):
        self.history.clear()
        self.entry_times.clear()
        self.last_stationary_checkpoint.clear()
        self.stationary_duration.clear()

    def _warmup_ml_model(self):
        """Train ML baseline on calibrated nominal movement patterns."""
        np.random.seed(42)
        n_samples = 400
        nominal_speeds = np.random.normal(0.012, 0.004, (n_samples, 1))
        nominal_accel = np.random.normal(0.0015, 0.0008, (n_samples, 1))
        nominal_dir = np.random.uniform(0.0, 0.25, (n_samples, 1))
        nominal_dist = np.random.uniform(0.05, 0.40, (n_samples, 1))
        nominal_dwell = np.random.uniform(2.0, 25.0, (n_samples, 1))
        nominal_stat = np.random.uniform(0.0, 3.0, (n_samples, 1))
        nominal_crowd = np.random.randint(1, 6, (n_samples, 1))

        nominal_dataset = np.hstack([
            nominal_speeds, nominal_accel, nominal_dir, nominal_dist,
            nominal_dwell, nominal_stat, nominal_crowd
        ])
        nominal_dataset = np.clip(nominal_dataset, 0, None)
        self.iso_forest.fit(nominal_dataset)

    def extract(self, track_id: int, curr_norm_pt: Tuple[float, float], video_time_sec: float):
        history = self.history[track_id]
        history.append((*curr_norm_pt, float(video_time_sec)))

        if track_id not in self.entry_times:
            self.entry_times[track_id] = float(video_time_sec)
            self.last_stationary_checkpoint[track_id] = (curr_norm_pt, float(video_time_sec))

        dwell_time = max(float(video_time_sec) - self.entry_times[track_id], 0.0)

        if len(history) < 2:
            return 0.0, 0.0, 0.0, 0.0, dwell_time, 0.0, None

        x_curr, y_curr, t_curr = history[-1]
        x_prev, y_prev, t_prev = history[-2]
        dt = max(t_curr - t_prev, 1e-4)
        instant_speed = float(np.hypot(x_curr - x_prev, y_curr - y_prev) / dt)

        instant_accel = 0.0
        if len(history) >= 3:
            x_p2, y_p2, t_p2 = history[-3]
            dt_prev = max(t_prev - t_p2, 1e-4)
            prev_speed = np.hypot(x_prev - x_p2, y_prev - y_p2) / dt_prev
            instant_accel = float(abs(instant_speed - prev_speed) / dt)

        dir_change = 0.0
        if len(history) >= 3:
            v1 = np.array([x_prev - history[-3][0], y_prev - history[-3][1]])
            v2 = np.array([x_curr - x_prev, y_curr - y_prev])
            norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm_v1 > 1e-4 and norm_v2 > 1e-4:
                cosine = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
                dir_change = float(1.0 - cosine)

        x_orig, y_orig, _ = history[0]
        dist_traveled = float(np.hypot(x_curr - x_orig, y_curr - y_orig))

        chk_pt, chk_time = self.last_stationary_checkpoint[track_id]
        if np.hypot(x_curr - chk_pt[0], y_curr - chk_pt[1]) > 0.035:
            self.last_stationary_checkpoint[track_id] = (curr_norm_pt, float(video_time_sec))
            self.stationary_duration[track_id] = 0.0
        else:
            self.stationary_duration[track_id] = float(video_time_sec - chk_time)

        prev_point = Point(x_prev, y_prev)
        return instant_speed, instant_accel, dir_change, dist_traveled, dwell_time, self.stationary_duration[track_id], prev_point

    def analyze_batch(
        self,
        features: List[List[float]],
        meta_list: List[Tuple],
        video_time_label: str
    ) -> Tuple[List[Dict], List[Dict]]:
        profiles = []
        alerts = []

        if not features:
            return profiles, alerts

        raw_scores = self.iso_forest.score_samples(np.array(features))
        anomaly_probs = np.clip(1.0 - ((raw_scores + 0.5) / 1.0), 0.0, 1.0)

        for meta, f_vec, anom_score in zip(meta_list, features, anomaly_probs):
            t_id, c_name, box, in_zone, dwell_time, stat_dur, total_detected = meta
            anom_score = float(anom_score)

            # Tactical Behavioral Risk Calculation
            w_ml = 0.60 * anom_score
            w_loiter = 0.25 * min(stat_dur / 8.0, 1.0)
            w_dwell = 0.15 * min(dwell_time / 30.0, 1.0)
            behavior_risk = float(np.clip((w_ml + w_loiter + w_dwell) * 100, 0, 100))

            if in_zone:
                risk_level = "CRITICAL"
                risk_score = 100.0
            elif behavior_risk >= 75.0:
                risk_level = "HIGH"
                risk_score = behavior_risk
            elif behavior_risk >= 40.0:
                risk_level = "MEDIUM"
                risk_score = behavior_risk
            else:
                risk_level = "LOW"
                risk_score = behavior_risk

            profiles.append(asdict(TargetKinematicProfile(
                track_id=int(t_id),
                cls_name=str(c_name),
                speed_norm=round(float(f_vec[0]), 4),
                accel_norm=round(float(f_vec[1]), 4),
                direction_change=round(float(f_vec[2]), 2),
                distance_traveled=round(float(f_vec[3]), 3),
                dwell_time=round(float(dwell_time), 1),
                in_restricted_zone=int(in_zone),
                stationary_duration=round(float(stat_dur), 1),
                crowd_density=int(total_detected),
                anomaly_score=round(anom_score, 2),
                risk_score=round(risk_score, 1),
                risk_level=str(risk_level)
            )))

            if not in_zone and anom_score > 0.65:
                alerts.append({
                    "type": "ML_BEHAVIOR_ANOMALY",
                    "track_id": int(t_id),
                    "timestamp": video_time_label,
                    "severity": "HIGH",
                    "message": f"[{video_time_label}] Tactical Anomaly: Erratic Kinematics on Target #{t_id} ({c_name}, Score: {anom_score:.2f})"
                })
            elif not in_zone and stat_dur > 5.0:
                alerts.append({
                    "type": "LOITERING",
                    "track_id": int(t_id),
                    "timestamp": video_time_label,
                    "severity": "MEDIUM",
                    "message": f"[{video_time_label}] Loitering Alert: Target #{t_id} stationary for {int(stat_dur)}s"
                })

        return profiles, alerts
