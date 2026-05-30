"""
Step 5b: TripoSR - Fast 3D reconstruction from single image
Used as reconstruction step after Zero123++ generates multi-views
OR as a standalone fast single-image to 3D (better quality than Zero123 alone)
"""
import os
import sys
import subprocess
import torch
import time
import json

os.environ["HF_HOME"] = "/apdcephfs_hzlf/share_1227201/easyi/model_cache"
os.environ["TORCH_HOME"] = "/apdcephfs_hzlf/share_1227201/easyi/model_cache"

BASE_DIR = "/apdcephfs_hzlf/share_1227201/easyi/objectC"
INPUT_IMG = os.path.join(BASE_DIR, "object_C_256_rgba.png")
OUTPUT_DIR = os.path.join(BASE_DIR, "triposr_output")
TRIPOSR_DIR = "/apdcephfs_hzlf/share_1227201/easyi/TripoSR"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# TripoSR with --no-remove-bg doesn't create the image subdir, pre-create it
os.makedirs(os.path.join(OUTPUT_DIR, "0"), exist_ok=True)

print("="*60)
print("TripoSR - Single Image 3D Reconstruction")
print("="*60)

# Clone TripoSR if not present
if not os.path.exists(TRIPOSR_DIR):
    print("Cloning TripoSR...")
    subprocess.run(["git", "clone", "--depth=1",
                   "https://github.com/VAST-AI-Research/TripoSR.git", TRIPOSR_DIR])

# Install TripoSR dependencies
print("\nInstalling TripoSR dependencies...")
os.chdir(TRIPOSR_DIR)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."],
               capture_output=True)

# Additional deps
subprocess.run([sys.executable, "-m", "pip", "install",
               "rembg[gpu]", "trimesh[easy]", "xatlas", "open3d"],
               capture_output=True)
print("Dependencies installed")

# Run TripoSR
print("\nRunning TripoSR reconstruction...")
t_start = time.time()

cmd = [
    sys.executable, os.path.join(TRIPOSR_DIR, "run.py"),
    INPUT_IMG,
    "--output-dir", OUTPUT_DIR,
    "--device", "cuda",
    "--chunk-size", "8192",
    "--mc-resolution", "256",  # Marching cubes resolution
    "--no-remove-bg",  # Already have RGBA with transparent bg
    "--foreground-ratio", "0.85",
    "--model-save-format", "obj",
]

print(f"Command: {' '.join(cmd[:5])} ...")
result = subprocess.run(cmd, capture_output=False)
triposr_time = time.time() - t_start

if result.returncode == 0:
    print(f"\nTripoSR complete in {triposr_time:.1f}s")
    timing = {"triposr_time_seconds": triposr_time, "triposr_time_minutes": triposr_time/60}
    with open(os.path.join(OUTPUT_DIR, "timing.json"), "w") as f:
        json.dump(timing, f, indent=2)

    # List output files
    print(f"\nOutput files in {OUTPUT_DIR}:")
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            fpath = os.path.join(root, f)
            print(f"  {fpath} ({os.path.getsize(fpath)/1024:.1f} KB)")
else:
    print(f"\nTripoSR failed with code {result.returncode}")
    sys.exit(1)
