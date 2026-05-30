#!/bin/bash
# Master pipeline script for Object A (3DGS)
# Runs all steps sequentially: frame extraction -> COLMAP -> 3DGS training
# Usage: nohup bash run_objectA_pipeline.sh > /apdcephfs_hzlf/share_1227201/easyi/logs/objectA.log 2>&1 &

LOG_DIR="/apdcephfs_hzlf/share_1227201/easyi/logs"
BASE_DIR="/apdcephfs_hzlf/share_1227201/easyi/objectA"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Object A Pipeline - $(date)"
echo "========================================"

# Step 1: Extract frames
echo ""
echo "[STEP 1/3] Extracting video frames..."
python3 "${BASE_DIR}/extract_frames.py" 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: Frame extraction failed!"
    exit 1
fi

# Step 2: COLMAP SfM
echo ""
echo "[STEP 2/3] Running COLMAP SfM..."
python3 "${BASE_DIR}/run_colmap.py" 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: COLMAP SfM failed!"
    exit 1
fi

# Step 3: 3DGS Training
echo ""
echo "[STEP 3/3] Training 3D Gaussian Splatting..."
python3 "${BASE_DIR}/train_3dgs.py" 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: 3DGS training failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "Object A Pipeline Complete! - $(date)"
echo "========================================"
