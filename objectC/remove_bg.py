"""
Step 4: Remove background from Object C image and prepare for Zero123
"""
import os
import sys
from PIL import Image
import numpy as np

BASE_DIR = "/apdcephfs_hzlf/share_1227201/easyi/objectC"
INPUT_IMG = os.path.join(BASE_DIR, "物体C图片.jpg")
OUTPUT_RGBA = os.path.join(BASE_DIR, "object_C_rgba.png")
OUTPUT_RGB_WHITE = os.path.join(BASE_DIR, "object_C_white_bg.png")

print("Installing rembg for background removal...")
os.system(f"{sys.executable} -m pip install rembg[gpu] onnxruntime-gpu 2>&1 | tail -5")

# Now try rembg
try:
    from rembg import remove
    print("rembg loaded successfully (GPU mode)")
    GPU_AVAILABLE = True
except ImportError:
    print("GPU rembg not available, trying CPU version...")
    os.system(f"{sys.executable} -m pip install rembg onnxruntime 2>&1 | tail -3")
    from rembg import remove
    GPU_AVAILABLE = False

print(f"\n{'='*60}")
print("Background Removal for Object C")
print(f"{'='*60}")

# Load and check input image
img = Image.open(INPUT_IMG)
print(f"Input image: {img.size}, mode={img.mode}")

# Remove background using rembg (u2net model by default)
print("\nRemoving background...")
with open(INPUT_IMG, "rb") as f:
    input_bytes = f.read()

output_bytes = remove(input_bytes)

# Save RGBA output
with open(OUTPUT_RGBA, "wb") as f:
    f.write(output_bytes)

print(f"Saved RGBA image: {OUTPUT_RGBA}")

# Also create white background version for Zero123
rgba = Image.open(OUTPUT_RGBA).convert("RGBA")
white_bg = Image.new("RGB", rgba.size, (255, 255, 255))
white_bg.paste(rgba, mask=rgba.split()[3])
white_bg.save(OUTPUT_RGB_WHITE)
print(f"Saved white-bg image: {OUTPUT_RGB_WHITE}")

# Center and crop the object
rgba_arr = np.array(rgba)
alpha = rgba_arr[:, :, 3]
rows = np.any(alpha > 10, axis=1)
cols = np.any(alpha > 10, axis=0)
rmin, rmax = np.where(rows)[0][[0, -1]]
cmin, cmax = np.where(cols)[0][[0, -1]]

# Add 10% padding
h, w = rgba_arr.shape[:2]
pad_r = int((rmax - rmin) * 0.1)
pad_c = int((cmax - cmin) * 0.1)
rmin = max(0, rmin - pad_r)
rmax = min(h, rmax + pad_r)
cmin = max(0, cmin - pad_c)
cmax = min(w, cmax + pad_c)

# Crop and save as square
cropped = rgba.crop((cmin, rmin, cmax, rmax))
# Make square
max_dim = max(cropped.size)
square = Image.new("RGBA", (max_dim, max_dim), (255, 255, 255, 0))
offset = ((max_dim - cropped.width) // 2, (max_dim - cropped.height) // 2)
square.paste(cropped, offset)
# Resize to 256x256 for Zero123
square_256 = square.resize((256, 256), Image.LANCZOS)
square_256.save(os.path.join(BASE_DIR, "object_C_256_rgba.png"))

# Also save white background version
white_square = Image.new("RGB", (256, 256), (255, 255, 255))
white_square.paste(square_256, mask=square_256.split()[3])
white_square.save(os.path.join(BASE_DIR, "object_C_256_white.png"))

print(f"\nCropped + resized to 256x256:")
print(f"  RGBA: {BASE_DIR}/object_C_256_rgba.png")
print(f"  White BG: {BASE_DIR}/object_C_256_white.png")
print("\nBackground removal complete!")
