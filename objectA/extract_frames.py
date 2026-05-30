"""
Step 1: Extract frames from video for COLMAP SfM
Extracts evenly spaced frames from the input video
"""
import cv2
import os
import numpy as np
from pathlib import Path

VIDEO_PATH = "/apdcephfs_hzlf/share_1227201/easyi/objectA/物体A视频.mov"
OUTPUT_DIR = "/apdcephfs_hzlf/share_1227201/easyi/objectA/images"
TARGET_FRAMES = 120  # Good number for object-level 3DGS
MAX_RESOLUTION = 1080  # Resize longer side to this

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = total_frames / fps

print(f"Video: {total_frames} frames, {fps:.1f} fps, {w}x{h}, {duration:.1f}s")

# Select evenly spaced frame indices
frame_indices = np.linspace(0, total_frames - 1, TARGET_FRAMES, dtype=int)
print(f"Extracting {len(frame_indices)} frames...")

# Compute scale for resizing
scale = MAX_RESOLUTION / max(w, h)
new_w = int(w * scale)
new_h = int(h * scale)
# Make even
new_w = new_w - (new_w % 2)
new_h = new_h - (new_h % 2)

saved = 0
frame_idx = 0
set_indices = set(frame_indices.tolist())

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx in set_indices:
        # Resize
        resized = cv2.resize(frame, (new_w, new_h))
        out_path = os.path.join(OUTPUT_DIR, f"frame_{saved:04d}.jpg")
        cv2.imwrite(out_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
        if saved % 20 == 0:
            print(f"  Saved {saved}/{TARGET_FRAMES} frames")

    frame_idx += 1

cap.release()
print(f"\nExtracted {saved} frames to {OUTPUT_DIR}")
print(f"Image size: {new_w}x{new_h}")
