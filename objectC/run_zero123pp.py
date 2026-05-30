"""
Step 5: Zero123++ - Single image to 3D generation for Object C
Uses Zero123++ (or Zero123) to generate multi-view images, then TripoSR/instant3d for mesh

Pipeline:
1. Run Zero123++ to generate 6 novel views from single image
2. Run reconstruction (SuGaR/instant-ngp/TripoSR) to get 3D mesh
"""
import os
import sys
import subprocess
import torch
import time

os.environ["HF_HOME"] = "/apdcephfs_hzlf/share_1227201/easyi/model_cache"
os.environ["TORCH_HOME"] = "/apdcephfs_hzlf/share_1227201/easyi/model_cache"

BASE_DIR = "/apdcephfs_hzlf/share_1227201/easyi/objectC"
INPUT_IMG = os.path.join(BASE_DIR, "object_C_256_rgba.png")
OUTPUT_DIR = os.path.join(BASE_DIR, "zero123_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("Zero123++ Single Image to 3D")
print("="*60)

# Try to use Zero123++ (the improved version)
# Install dependencies
print("\nInstalling Zero123++ dependencies...")
deps = [
    "diffusers==0.24.0",
    "transformers",
    "accelerate",
    "safetensors",
    "invisible_watermark",
    "kornia",
    "einops",
    "trimesh",
    "open3d",
    "huggingface_hub",
]
subprocess.run([sys.executable, "-m", "pip", "install"] + deps, capture_output=True)
print("Dependencies installed")

# Method 1: Try Zero123++ via diffusers
print("\n[Method] Using Zero123++ via diffusers pipeline...")

try:
    import torch
    from diffusers import DiffusionPipeline
    from PIL import Image
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Zero123++ pipeline
    print("Loading Zero123++ model (this may take a while for first download)...")
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16,
        cache_dir="/apdcephfs_hzlf/share_1227201/easyi/model_cache",
    )
    pipeline.to(device)
    print("Zero123++ model loaded!")

    # Load input image
    cond_img = Image.open(INPUT_IMG).convert("RGB")
    print(f"Input image: {cond_img.size}")

    # Generate 6 views
    print("\nGenerating 6 novel views...")
    t_start = time.time()
    result = pipeline(cond_img, num_inference_steps=75)
    gen_time = time.time() - t_start
    print(f"Generation complete in {gen_time:.1f}s")

    # Save the 6-view output image
    result_img = result.images[0]
    result_img.save(os.path.join(OUTPUT_DIR, "zero123pp_6views.png"))

    # Split into individual views (Zero123++ outputs a 3x2 grid)
    w, h = result_img.size
    views = []
    for row in range(2):
        for col in range(3):
            x = col * (w // 3)
            y = row * (h // 2)
            view = result_img.crop((x, y, x + w // 3, y + h // 2))
            view.save(os.path.join(OUTPUT_DIR, f"view_{row*3+col:02d}.png"))
            views.append(view)
    print(f"Saved 6 individual views to {OUTPUT_DIR}")

    print(f"\nZero123++ generation complete!")
    print(f"Generation time: {gen_time:.1f}s")

    # Save timing
    import json
    with open(os.path.join(OUTPUT_DIR, "generation_time.json"), "w") as f:
        json.dump({"zero123pp_time_seconds": gen_time}, f)

except Exception as e:
    print(f"Zero123++ failed: {e}")
    print("Trying alternative: Zero123 via threestudio...")
    # Fallback will be handled in the next script
    sys.exit(1)
