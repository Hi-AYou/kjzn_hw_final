#!/bin/bash
# Sequential pipeline: Object B v5 → Object A v3 render comparison
# Usage: nohup bash /apdcephfs_hzlf/share_1227201/easyi/run_pipeline.sh > /apdcephfs_hzlf/share_1227201/easyi/logs/pipeline.log 2>&1 &

set -e
LOG_DIR="/apdcephfs_hzlf/share_1227201/easyi/logs"
mkdir -p "$LOG_DIR"

export HF_HOME="/apdcephfs_hzlf/share_1227201/easyi/model_cache"
export TORCH_HOME="/apdcephfs_hzlf/share_1227201/easyi/model_cache"
export WANDB_API_KEY="${WANDB_API_KEY}"  # set this in your environment before running
# Proxy for any runtime downloads (HuggingFace model cache hits should skip this)
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128

echo "========================================================"
echo " Pipeline start: $(date)"
echo "========================================================"

# ── Step 1: Object B v5 (DreamFusion + Perp-Neg, ~40 min) ──────────────────
echo ""
echo "[STEP 1] Object B v5 — Corgi with Perp-Neg (start: $(date))"
python3.9 /apdcephfs_hzlf/share_1227201/easyi/objectB/run_objectB_v5.py \
    2>&1 | tee "$LOG_DIR/objectB_v5.log"
echo "[STEP 1] Done: $(date)"

# ── Step 2: Object A v3 render comparison ─────────────────────────────────
echo ""
echo "[STEP 2] Object A v3 — Render comparison images (start: $(date))"
python3.9 /apdcephfs_hzlf/share_1227201/easyi/objectA/render_gs_v3.py \
    2>&1 | tee "$LOG_DIR/objectA_render_v3.log"
echo "[STEP 2] Done: $(date)"

echo ""
echo "========================================================"
echo " Pipeline complete: $(date)"
echo " Object B renders : /apdcephfs_hzlf/share_1227201/easyi/objectB/v5_render_*.png"
echo " Object A renders : /apdcephfs_hzlf/share_1227201/easyi/objectA/gs_renders_v3/"
echo "========================================================"
