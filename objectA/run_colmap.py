"""
Step 2: COLMAP SfM pipeline using pycolmap
Runs feature extraction, matching, and incremental mapping
"""
import pycolmap
import os
import shutil
from pathlib import Path
import time

BASE_DIR = "/apdcephfs_hzlf/share_1227201/easyi/objectA"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
DB_PATH = os.path.join(BASE_DIR, "database.db")
SPARSE_DIR = os.path.join(BASE_DIR, "sparse")

os.makedirs(SPARSE_DIR, exist_ok=True)

# Remove existing database if present
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

print("="*60)
print("COLMAP SfM Pipeline")
print("="*60)

# Step 1: Feature Extraction
print("\n[1/3] Extracting SIFT features...")
t0 = time.time()

sift_options = pycolmap.SiftExtractionOptions()
sift_options.max_num_features = 8192
sift_options.max_image_size = 1600

pycolmap.extract_features(
    database_path=DB_PATH,
    image_path=IMAGE_DIR,
    camera_mode=pycolmap.CameraMode.AUTO,
    camera_model="OPENCV",
    sift_options=sift_options,
)
print(f"  Feature extraction done in {time.time()-t0:.1f}s")

# Step 2: Feature Matching (sequential for video frames)
print("\n[2/3] Matching features (sequential)...")
t1 = time.time()

# Sequential matching is ideal for ordered video frames
seq_options = pycolmap.SequentialMatchingOptions()
seq_options.overlap = 15  # Match with 15 neighboring frames
seq_options.loop_detection = False  # Disable loop closure (requires vocab tree file)

try:
    pycolmap.match_sequential(
        database_path=DB_PATH,
        sift_options=pycolmap.SiftMatchingOptions(),
        matching_options=seq_options,
    )
    print(f"  Sequential matching done in {time.time()-t1:.1f}s")
except Exception as e:
    print(f"  Sequential matching error: {e}, falling back to exhaustive...")
    t1 = time.time()
    pycolmap.match_exhaustive(
        database_path=DB_PATH,
    )
    print(f"  Exhaustive matching done in {time.time()-t1:.1f}s")

# Step 3: Incremental Mapping
print("\n[3/3] Running incremental mapping (SfM)...")
t2 = time.time()

mapper_options = pycolmap.IncrementalPipelineOptions()
mapper_options.min_num_matches = 15
mapper_options.num_threads = -1  # Use all threads

reconstructions = pycolmap.incremental_mapping(
    database_path=DB_PATH,
    image_path=IMAGE_DIR,
    output_path=SPARSE_DIR,
    options=mapper_options,
)
print(f"  Incremental mapping done in {time.time()-t2:.1f}s")

print("\n" + "="*60)
print(f"Reconstruction Results:")
for idx, rec in reconstructions.items():
    num_images = len(rec.images)
    num_points = len(rec.points3D)
    print(f"  Model {idx}: {num_images} images, {num_points} 3D points")

# Find the best reconstruction (most images registered)
if reconstructions:
    best_key = max(reconstructions, key=lambda k: len(reconstructions[k].images))
    best_rec = reconstructions[best_key]
    print(f"\nBest model: {best_key} with {len(best_rec.images)} images")

    # Save summary
    summary = {
        "num_images": len(best_rec.images),
        "num_points3D": len(best_rec.points3D),
        "mean_reprojection_error": best_rec.compute_mean_reprojection_error(),
        "total_time": time.time() - t0,
    }
    import json
    with open(os.path.join(BASE_DIR, "colmap_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {BASE_DIR}/colmap_summary.json")
    print(f"Total COLMAP time: {summary['total_time']:.1f}s")
    print(f"Mean reprojection error: {summary['mean_reprojection_error']:.4f} pixels")
else:
    print("ERROR: No reconstruction produced!")
