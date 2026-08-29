"""
NSG Tactical AI Command Engine - Geofencing & Spatial Boundary Defense
Supports:
1. N-Point Freeform Polygon Geofences (Sector Restrictions)
2. Virtual Directional Tripwires (Perimeter Crossing)
3. Radial VIP / High-Value Asset Exclusion Zones
"""
import threading
from typing import List, Dict, Tuple, Optional
from shapely.geometry import Point, Polygon, LineString
import numpy as np
import cv2


class TacticalZone:
    def __init__(
        self,
        zone_id: str,
        name: str,
        shape_type: str,
        points: List[List[float]],
        radius: float = 0.1,
        color: Tuple[int, int, int] = (0, 0, 255),
        alert_type: str = "ZONE_BREACH"
    ):
        self.zone_id = zone_id
        self.name = name
        self.shape_type = shape_type.upper()  # 'POLYGON', 'TRIPWIRE', 'CIRCLE'
        self.points = points  # Normalized coords [[x, y], ...]
        self.radius = radius
        self.color = color
        self.alert_type = alert_type
        
        self.geometry = None
        self._build_geometry()

    def _build_geometry(self):
        if self.shape_type == "POLYGON" and len(self.points) >= 3:
            self.geometry = Polygon([(p[0], p[1]) for p in self.points])
        elif self.shape_type == "TRIPWIRE" and len(self.points) >= 2:
            self.geometry = LineString([(self.points[0][0], self.points[0][1]), (self.points[1][0], self.points[1][1])])
        elif self.shape_type == "CIRCLE" and len(self.points) >= 1:
            self.geometry = Point(self.points[0][0], self.points[0][1])

    def check_breach(self, curr_pt: Point, prev_pt: Optional[Point] = None) -> bool:
        if self.geometry is None:
            return False
            
        if self.shape_type == "POLYGON":
            return bool(self.geometry.contains(curr_pt))
        elif self.shape_type == "TRIPWIRE":
            if prev_pt is not None:
                movement_line = LineString([prev_pt, curr_pt])
                return bool(movement_line.intersects(self.geometry))
            return False
        elif self.shape_type == "CIRCLE":
            return bool(self.geometry.distance(curr_pt) <= self.radius)
        return False

    def to_dict(self) -> Dict:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "shape_type": self.shape_type,
            "points": self.points,
            "radius": self.radius,
            "alert_type": self.alert_type
        }


class GeofenceManager:
    def __init__(self):
        self.zones: Dict[str, TacticalZone] = {}
        self.lock = threading.Lock()
        
        # Default starting tactical defense sector
        self.add_zone(TacticalZone(
            zone_id="PRIMARY",
            name="Alpha Restricted Sector",
            shape_type="POLYGON",
            points=[[0.15, 0.15], [0.65, 0.15], [0.65, 0.70], [0.15, 0.70]],
            color=(0, 0, 255),
            alert_type="ZONE_BREACH"
        ))

    def add_zone(self, zone: TacticalZone):
        with self.lock:
            self.zones[zone.zone_id] = zone

    def set_single_zone(self, shape_type: str, points: List[List[float]], radius: float = 0.1, name: str = "Tactical Perimeter"):
        with self.lock:
            self.zones.clear()
            alert_name = "ZONE_BREACH"
            if shape_type == "TRIPWIRE":
                alert_name = "TRIPWIRE_CROSSING"
            elif shape_type == "CIRCLE":
                alert_name = "VIP_CORDON_BREACH"

            self.zones["PRIMARY"] = TacticalZone(
                zone_id="PRIMARY",
                name=name,
                shape_type=shape_type,
                points=points,
                radius=radius,
                color=(0, 0, 255),
                alert_type=alert_name
            )

    def clear_all(self):
        with self.lock:
            self.zones.clear()

    def check_all_breaches(self, curr_pt: Point, prev_pt: Optional[Point] = None) -> List[TacticalZone]:
        breached = []
        with self.lock:
            for zone in self.zones.values():
                if zone.check_breach(curr_pt, prev_pt):
                    breached.append(zone)
        return breached

    def render_zones_on_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        with self.lock:
            for zone in self.zones.values():
                if zone.shape_type == "POLYGON" and len(zone.points) >= 3:
                    pts_px = np.array([[int(p[0] * w), int(p[1] * h)] for p in zone.points], np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [pts_px], color=(0, 0, 160))
                    cv2.polylines(frame, [pts_px], isClosed=True, color=(0, 0, 255), thickness=2)
                    label_x = int(pts_px[0][0][0])
                    label_y = max(int(pts_px[0][0][1]) - 8, 20)
                    cv2.putText(frame, f"[RESTRICTED SECTOR: {zone.name}]", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    
                elif zone.shape_type == "TRIPWIRE" and len(zone.points) >= 2:
                    p1 = (int(zone.points[0][0] * w), int(zone.points[0][1] * h))
                    p2 = (int(zone.points[1][0] * w), int(zone.points[1][1] * h))
                    cv2.line(frame, p1, p2, (0, 165, 255), 3)
                    cv2.putText(frame, f"[TRIPWIRE: {zone.name}]", (p1[0], max(p1[1] - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2)
                    
                elif zone.shape_type == "CIRCLE" and len(zone.points) >= 1:
                    center = (int(zone.points[0][0] * w), int(zone.points[0][1] * h))
                    radius_px = int(zone.radius * w)
                    cv2.circle(overlay, center, radius_px, (0, 80, 220), -1)
                    cv2.circle(frame, center, radius_px, (0, 165, 255), 2)
                    cv2.putText(frame, f"[RADIAL CORDON: {zone.name}]", (center[0] - 60, max(center[1] - radius_px - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2)

        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
        return frame

    def get_zones_summary(self) -> List[Dict]:
        with self.lock:
            return [z.to_dict() for z in self.zones.values()]
