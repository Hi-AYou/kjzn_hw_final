#!/bin/bash
# Master pipeline script for Object C (Zero123 / TripoSR)
# Runs: background removal -> Zero123++ -> TripoSR 3D reconstruction
# Usage: nohup bash run_objectC_pipeline.sh > /apdcephfs_hzlf/share_1227201/easyi/logs/objectC.log 2>&1 &

LOG_DIR="/apdcephfs_hzlf/share_1227201/easyi/logs"
BASE_DIR="/apdcephfs_hzlf/share_1227201/easyi/objectC"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Object C Pipeline - $(date)"
echo "========================================"

# Step 1: Background removal
echo ""
echo "[STEP 1/3] Removing background from Object C image..."
/usr/bin/python3.9 -u "${BASE_DIR}/remove_bg.py" 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: Background removal failed!"
    exit 1
fi

# Step 2: Zero123++ multi-view generation
echo ""
echo "[STEP 2/3] Running Zero123++ novel view generation..."
/usr/bin/python3.9 -u "${BASE_DIR}/run_zero123pp.py" 2>&1
ZERO123_STATUS=$?

if [ $ZERO123_STATUS -ne 0 ]; then
    echo "Zero123++ failed or unavailable, skipping to TripoSR..."
fi

# Step 3: TripoSR reconstruction
echo ""
echo "[STEP 3/3] Running TripoSR 3D reconstruction..."
/usr/bin/python3.9 -u "${BASE_DIR}/run_triposr.py" 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: TripoSR failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "Object C Pipeline Complete! - $(date)"
echo "========================================"
