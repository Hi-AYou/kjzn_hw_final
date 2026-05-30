"""
Object B v5: Corgi — Janus-problem fix via Perp-Neg + view-dependent prompting
Key upgrades over v4:
  1. use_perp_neg=True  → perpendicular-negative guidance (the strongest Janus fix)
  2. view_dependent_prompt_front=True → threestudio prepends "front/side/back view of"
  3. Richer, anatomy-accurate prompt with explicit back/side descriptions
  4. Tweaked guidance_scale (50) and max_step_percent to reduce over-saturation
  5. 15 000 steps for better convergence
"""
import os, sys, subprocess, time, json, shutil, glob

os.environ["HF_HOME"]   = "/apdcephfs_hzlf/share_1227201/easyi/model_cache"
os.environ["TORCH_HOME"] = "/apdcephfs_hzlf/share_1227201/easyi/model_cache"
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", "")  # set via environment variable
# Disable wandb online sync to avoid blocking on network issues in this environment
os.environ["WANDB_MODE"] = "offline"
# Use cached HuggingFace models only (no network calls)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

THREESTUDIO_DIR = "/apdcephfs_hzlf/share_1227201/easyi/threestudio"
BASE_DIR        = "/apdcephfs_hzlf/share_1227201/easyi/objectB"
SD_MODEL        = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# ── Prompt design ────────────────────────────────────────────────────────────
# The view_dependent_prompt_front feature will automatically prepend
# "front view of", "side view of", "back view of", "overhead view of"
# to the prompt depending on the current camera elevation/azimuth, so the
# base prompt should describe the FULL animal without directional words.
PROMPT = (
    "a Pembroke Welsh Corgi dog, sitting, orange white fur, "
    "very short legs, long body, white chest, "
    "large upright ears, one face, one nose, one mouth, "
    "brown eyes, stubby tail, "
    "toy figurine, white background, studio lighting, 3D render"
)

# Negative prompt: explicitly list every Janus artifact + anatomical errors
NEGATIVE_PROMPT = (
    "two faces, multiple faces, face on back, face on side, "
    "extra heads, three ears, four ears, extra ears, "
    "extra legs, missing legs, extra tails, extra noses, "
    "blurry, low quality, bad anatomy, ugly, deformed, mutated, "
    "extra limbs, floating limbs"
)

print("=" * 70)
print("Object B v5: Corgi — Perp-Neg + view-dependent prompting")
print("=" * 70)
print(f"Prompt  : {PROMPT}")
print(f"Negative: {NEGATIVE_PROMPT}")

t_start = time.time()

cmd = [
    sys.executable, os.path.join(THREESTUDIO_DIR, "launch.py"),
    "--config", os.path.join(THREESTUDIO_DIR, "configs/dreamfusion-sd.yaml"),
    "--train",
    "--gpu", "0",
    # ── core prompts ─────────────────────────────────────────────────────────
    f"system.prompt_processor.prompt={PROMPT}",
    f"system.prompt_processor.negative_prompt={NEGATIVE_PROMPT}",
    # ── KEY Janus fixes ───────────────────────────────────────────────────────
    # 1) view-dependent prompting: automatically adds "front/back/side view of"
    "system.prompt_processor.view_dependent_prompt_front=true",
    # 2) perp_neg: perpendicular-negative guidance (strongest Janus suppression)
    "system.prompt_processor.use_perp_neg=true",
    # ── model ─────────────────────────────────────────────────────────────────
    f"system.prompt_processor.pretrained_model_name_or_path={SD_MODEL}",
    f"system.guidance.pretrained_model_name_or_path={SD_MODEL}",
    # ── training config ───────────────────────────────────────────────────────
    "trainer.max_steps=15000",
    f"exp_root_dir={BASE_DIR}/outputs_v5",
    "tag=corgi_v5",
    "data.width=128",
    "data.height=128",
    # Lower guidance than v4 (was 60) to reduce over-saturation/artifacts
    "system.guidance.guidance_scale=50.0",
    "system.guidance.min_step_percent=0.02",
    "system.guidance.max_step_percent=0.85",
    # Camera sampling: wider elevation range to see top/bottom more
    "data.elevation_range=[-20, 50]",
    # Orient loss weight: keeps the object "upright" and front-facing
    "system.loss.lambda_orient=[0,10.,1000.,3000]",
    # WandB logging — disabled online to avoid network blocking; logs saved offline
    "system.loggers.wandb.enable=false",
]

print(f"\nLaunching training (15 000 steps, perp-neg enabled)...")
result = subprocess.run(cmd, cwd=THREESTUDIO_DIR)
train_time = time.time() - t_start
status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
print(f"\n[Training] {status} in {train_time/60:.1f} min")

# ── Copy final renders ────────────────────────────────────────────────────────
exp_dirs = sorted(glob.glob(os.path.join(BASE_DIR, "outputs_v5/*/*/")))
if exp_dirs:
    latest = exp_dirs[-1]
    print(f"Experiment dir: {latest}")

    # Last 6 PNG renders
    renders = sorted(glob.glob(os.path.join(latest, "save/*.png")))
    for i, f in enumerate(renders[-6:]):
        dst = os.path.join(BASE_DIR, f"v5_render_{i:02d}.png")
        shutil.copy(f, dst)
        print(f"  Copied {os.path.basename(f)} → {os.path.basename(dst)}")

    # Latest test video
    videos = sorted(glob.glob(os.path.join(latest, "save/*test*.mp4")))
    if not videos:
        videos = sorted(glob.glob(os.path.join(latest, "save/*.mp4")))
    for f in videos[-1:]:
        dst = os.path.join(BASE_DIR, "v5_" + os.path.basename(f))
        shutil.copy(f, dst)
        print(f"  Copied {os.path.basename(f)} → {os.path.basename(dst)}")
else:
    print("WARNING: no experiment directory found!")

# ── Save timing / metadata ────────────────────────────────────────────────────
timing = {
    "method": "DreamFusion-SD v5 (corgi, perp-neg + view-dependent)",
    "prompt": PROMPT,
    "negative_prompt": NEGATIVE_PROMPT,
    "steps": 15000,
    "resolution": "128x128",
    "guidance_scale": 50.0,
    "use_perp_neg": True,
    "view_dependent_prompt_front": True,
    "time_minutes": train_time / 60,
    "status": status,
}
timing_path = os.path.join(BASE_DIR, "timing_v5.json")
with open(timing_path, "w") as f:
    json.dump(timing, f, indent=2, ensure_ascii=False)

print(f"\n{'='*70}")
print(f"Done! Time: {train_time/60:.1f} min | Status: {status}")
print(f"Renders: {BASE_DIR}/v5_render_*.png")
print(f"Timing : {timing_path}")
