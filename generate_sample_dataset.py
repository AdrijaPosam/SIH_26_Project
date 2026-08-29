"""
Generates synthetic sample tactical surveillance MP4 videos for local testing.
Creates Drone UAV, UGV Robot, Bodycam, and Perimeter CCTV clips.
"""
import cv2
import math
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("sample_videos")
OUTPUT_DIR.mkdir(exist_ok=True)

def create_sample_clips():
    width, height = 1280, 720
    fps = 25.0
    duration_sec = 12
    total_frames = int(fps * duration_sec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    clips = {
        "Drone_UAV_Recon.mp4": "DRONE",
        "UGV_Robot_Corridor.mp4": "UGV",
        "Bodycam_Operator_Lead.mp4": "BODYCAM",
        "Perimeter_Gate_CCTV.mp4": "CCTV"
    }

    for filename, clip_type in clips.items():
        out_path = OUTPUT_DIR / filename
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        print(f"[DATASET GENERATOR] Generating {filename} ({total_frames} frames)...")

        for f in range(total_frames):
            t = f / fps
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            if clip_type == "DRONE":
                frame[:] = (35, 45, 30)
                # Runway / road
                cv2.rectangle(frame, (0, int(height * 0.45)), (width, int(height * 0.65)), (55, 60, 55), -1)
                # Moving target 1 (Walking)
                p1_x = int(width * 0.35 + 140 * math.sin(t * 0.8))
                p1_y = int(height * 0.35 + 80 * math.cos(t * 0.8))
                cv2.circle(frame, (p1_x, p1_y), 14, (220, 220, 220), -1)
                # Moving target 2 (Fast vehicle)
                vx = int((width * 0.1) + (width * 0.8) * ((t * 0.2) % 1.0))
                cv2.rectangle(frame, (vx - 35, int(height * 0.55) - 18), (vx + 35, int(height * 0.55) + 18), (50, 80, 130), -1)
                # Stationary bag
                cv2.rectangle(frame, (int(width * 0.5) - 8, int(height * 0.35) - 8), (int(width * 0.5) + 8, int(height * 0.35) + 8), (0, 165, 255), -1)

            elif clip_type == "UGV":
                frame[:] = (20, 20, 25)
                # Perspective lines
                cv2.line(frame, (0, 0), (int(width * 0.35), int(height * 0.5)), (80, 80, 80), 2)
                cv2.line(frame, (width, 0), (int(width * 0.65), int(height * 0.5)), (80, 80, 80), 2)
                cv2.line(frame, (0, height), (int(width * 0.35), int(height * 0.5)), (100, 100, 100), 2)
                cv2.line(frame, (width, height), (int(width * 0.65), int(height * 0.5)), (100, 100, 100), 2)
                # Approaching target
                scale = 0.5 + 0.5 * ((t * 0.15) % 1.0)
                tx, ty = int(width * 0.5), int(height * 0.55)
                tw, th = int(50 * scale), int(140 * scale)
                cv2.rectangle(frame, (tx - tw // 2, ty - th), (tx + tw // 2, ty), (180, 150, 120), -1)
                cv2.circle(frame, (tx, ty - th - 12), int(16 * scale), (200, 180, 150), -1)

            elif clip_type == "BODYCAM":
                sway = int(12 * math.sin(t * 3.0))
                frame[:] = (30, 25, 25)
                cv2.line(frame, (int(width * 0.4 + sway), 0), (int(width * 0.4 + sway), height), (70, 70, 70), 3)
                sx = int(width * 0.7 + sway + 100 * math.sin(t * 1.5))
                cv2.rectangle(frame, (sx - 30, int(height * 0.3)), (sx + 30, int(height * 0.85)), (160, 120, 100), -1)
                cv2.circle(frame, (sx, int(height * 0.24)), 22, (200, 170, 140), -1)

            else:
                frame[:] = (25, 30, 35)
                # Gate barrier
                cv2.line(frame, (int(width * 0.1), int(height * 0.6)), (int(width * 0.9), int(height * 0.6)), (120, 120, 120), 4)
                px = int(width * 0.2 + (width * 0.6) * ((t * 0.25) % 1.0))
                cv2.rectangle(frame, (px - 20, int(height * 0.35)), (px + 20, int(height * 0.75)), (190, 190, 190), -1)
                cv2.circle(frame, (px, int(height * 0.30)), 16, (210, 180, 150), -1)

            out.write(frame)

        out.release()
        print(f"[DATASET GENERATOR] Saved {out_path}")

if __name__ == "__main__":
    create_sample_clips()
