"""
NSG Tactical AI Command Engine - Unattended Object & Suspicious Baggage / IED Detector
Monitors stationary backpacks, luggage, boxes separated from human owners.
"""
import time
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class UnattendedBaggageDetector:
    def __init__(self, proximity_threshold: float = 0.18, stationary_time_sec: float = 6.0):
        self.proximity_threshold = proximity_threshold  # Normalized screen distance
        self.stationary_time_sec = stationary_time_sec
        self.bag_first_seen = {}
        self.bag_positions = {}
        self.bag_abandoned_duration = defaultdict(float)

    def reset_state(self):
        self.bag_first_seen.clear()
        self.bag_positions.clear()
        self.bag_abandoned_duration.clear()

    def update(
        self,
        detected_bags: List[Tuple[int, str, Tuple[float, float, float, float], Tuple[float, float]]],
        detected_persons: List[Tuple[int, Tuple[float, float]]],
        video_time_sec: float,
        video_time_label: str
    ) -> List[Dict]:
        alerts = []
        target_classes = {"backpack", "handbag", "suitcase", "bag", "box"}

        current_bag_ids = set()

        for track_id, cls_name, box, (cx_norm, cy_norm) in detected_bags:
            if cls_name.lower() not in target_classes:
                continue

            current_bag_ids.add(track_id)

            if track_id not in self.bag_first_seen:
                self.bag_first_seen[track_id] = video_time_sec
                self.bag_positions[track_id] = (cx_norm, cy_norm)

            # Check if bag has moved
            prev_pos = self.bag_positions[track_id]
            dist_moved = np.hypot(cx_norm - prev_pos[0], cy_norm - prev_pos[1])

            if dist_moved > 0.05:
                # Bag was picked up or moved
                self.bag_positions[track_id] = (cx_norm, cy_norm)
                self.bag_abandoned_duration[track_id] = 0.0
                continue

            # Calculate distance to nearest person
            min_dist_to_person = 999.0
            for p_id, (p_cx, p_cy) in detected_persons:
                d = np.hypot(cx_norm - p_cx, cy_norm - p_cy)
                if d < min_dist_to_person:
                    min_dist_to_person = d

            if min_dist_to_person > self.proximity_threshold:
                self.bag_abandoned_duration[track_id] += 0.2  # increment abandoned timer
            else:
                self.bag_abandoned_duration[track_id] = max(0.0, self.bag_abandoned_duration[track_id] - 0.5)

            if self.bag_abandoned_duration[track_id] >= self.stationary_time_sec:
                alerts.append({
                    "type": "UNATTENDED_BAGGAGE_IED",
                    "track_id": int(track_id),
                    "timestamp": video_time_label,
                    "severity": "CRITICAL",
                    "message": f"[{video_time_label}] PRIORITY THREAT: Unattended {cls_name.upper()} (Item #{track_id}) abandoned for {int(self.bag_abandoned_duration[track_id])}s (Min Owner Dist: {min_dist_to_person:.2f})"
                })

        # Cleanup disappeared bags
        for old_id in list(self.bag_first_seen.keys()):
            if old_id not in current_bag_ids:
                del self.bag_first_seen[old_id]
                if old_id in self.bag_positions:
                    del self.bag_positions[old_id]
                if old_id in self.bag_abandoned_duration:
                    del self.bag_abandoned_duration[old_id]

        return alerts
