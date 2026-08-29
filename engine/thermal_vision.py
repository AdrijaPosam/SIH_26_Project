"""
NSG Tactical AI Command Engine - Multi-Spectral Thermal & Night-Ops Vision Filter
Modes: RGB, FLIR Ironbow, White-Hot IR, Gen-3 Green Phosphor Night Vision.
"""
import cv2
import numpy as np


class ThermalVisionEngine:
    def __init__(self):
        self.active_mode = "RGB"  # "RGB", "FLIR_IRONBOW", "WHITE_HOT", "NIGHT_VISION"
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def set_mode(self, mode: str):
        mode = mode.upper()
        if mode in {"RGB", "FLIR_IRONBOW", "WHITE_HOT", "NIGHT_VISION"}:
            self.active_mode = mode

    def process(self, frame: np.ndarray) -> np.ndarray:
        if self.active_mode == "RGB":
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced_gray = self.clahe.apply(gray)

        if self.active_mode == "FLIR_IRONBOW":
            # Pseudo-thermal infrared colormap (Inferno / Ironbow)
            thermal = cv2.applyColorMap(enhanced_gray, cv2.COLORMAP_INFERNO)
            # Add thermal HUD watermark
            cv2.putText(thermal, "[FLIR THERMAL IR ACTIVE]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            return thermal

        elif self.active_mode == "WHITE_HOT":
            # FLIR White-Hot mode
            white_hot = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
            cv2.putText(white_hot, "[FLIR WHITE-HOT IR ACTIVE]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            return white_hot

        elif self.active_mode == "NIGHT_VISION":
            # Gen-3 Military Green Phosphor Night-Vision simulation
            nvg = np.zeros_like(frame)
            nvg[:, :, 1] = np.clip(enhanced_gray.astype(np.int32) * 1.35, 0, 255).astype(np.uint8)  # Boost Green
            nvg[:, :, 0] = (enhanced_gray * 0.15).astype(np.uint8)  # Subtle Blue
            nvg[:, :, 2] = (enhanced_gray * 0.15).astype(np.uint8)  # Subtle Red

            # Subtle scanlines
            h, w = frame.shape[:2]
            scanlines = np.zeros((h, w), dtype=np.uint8)
            scanlines[::4, :] = 25
            nvg = cv2.subtract(nvg, cv2.cvtColor(scanlines, cv2.COLOR_GRAY2BGR))

            cv2.putText(nvg, "[GEN-3 NVG GREEN PHOSPHOR]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            return nvg

        return frame
