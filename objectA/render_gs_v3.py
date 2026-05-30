"""
Render comparison images for Object A v3 (undistorted, 50K iteration model)
- Uses point_cloud_clean.ply (594K Gaussians) from gs_output_v3
- Uses images_undistorted/ as GT (pinhole camera, no distortion)
- Outputs: comp_XX_*.png (GT left | Render right) + render_metrics_v3.json
"""
import os, sys, json, math, time, struct
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

os.environ["HF_HOME"]    = "/apdcephfs_hzlf/share_1227201/easyi/model_cache"
os.environ["TORCH_HOME"] = "/apdcephfs_hzlf/share_1227201/easyi/model_cache"

BASE_DIR    = "/apdcephfs_hzlf/share_1227201/easyi/objectA"
PLY_PATH    = os.path.join(BASE_DIR, "gs_output_v3/point_cloud/iteration_50000/point_cloud_clean.ply")
# Fallback to full ply if clean doesn't exist
if not os.path.exists(PLY_PATH):
    PLY_PATH = os.path.join(BASE_DIR, "gs_output_v3/point_cloud/iteration_50000/point_cloud.ply")
SPARSE_DIR  = os.path.join(BASE_DIR, "sparse/0")
IMAGES_DIR  = os.path.join(BASE_DIR, "images_undistorted")
PINHOLE_K   = os.path.join(IMAGES_DIR, "pinhole_K.json")
OUT_DIR     = os.path.join(BASE_DIR, "gs_renders_v3")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"PLY    : {PLY_PATH}")
print(f"Images : {IMAGES_DIR}")
print(f"Output : {OUT_DIR}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")

# ── GSplat ───────────────────────────────────────────────────────────────────
from gsplat import rasterization
from plyfile import PlyData

# ── Load PLY ────────────────────────────────────────────────────────────────
print(f"\nLoading PLY ...")
ply = PlyData.read(PLY_PATH)
v   = ply["vertex"]
N   = len(v)
print(f"  {N:,} Gaussians loaded")

xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

# SH coefficients
f_dc  = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
f_rest_names = sorted([n for n in v.data.dtype.names if n.startswith("f_rest_")],
                      key=lambda s: int(s.split("_")[-1]))
if f_rest_names:
    f_rest = np.stack([v[n] for n in f_rest_names], axis=1).astype(np.float32)
else:
    f_rest = np.zeros((N, 45), dtype=np.float32)

K_sh = 16   # (sh_degree=3+1)^2
sh   = np.zeros((N, K_sh, 3), dtype=np.float32)
sh[:, 0, :] = f_dc
# Save ordering: sh_np[:,1:,:].reshape(N,-1) with C order → band-major interleaved
# f_rest[n, i] = sh[n, 1 + i//3, i%3]  i.e. [b1c0, b1c1, b1c2, b2c0, b2c1, b2c2, ...]
# Reshape [N,45] → [N,15,3] directly gives correct [band, channel] layout
sh[:, 1:, :] = f_rest.reshape(N, 15, 3)

opacities   = v["opacity"].astype(np.float32)
scale_names = sorted([n for n in v.data.dtype.names if n.startswith("scale_")])
scales      = np.stack([v[n] for n in scale_names], axis=1).astype(np.float32)
rot_names   = sorted([n for n in v.data.dtype.names if n.startswith("rot_")])
quats       = np.stack([v[n] for n in rot_names], axis=1).astype(np.float32)

means_t    = torch.from_numpy(xyz).to(device)
quats_t    = torch.from_numpy(quats).to(device)
scales_t   = torch.from_numpy(scales).to(device)
opacities_t= torch.from_numpy(opacities).to(device)
sh_t       = torch.from_numpy(sh).to(device)

# ── Camera parameters (pinhole after undistortion) ────────────────────────────
pk = json.load(open(PINHOLE_K))
FX, FY, CX, CY = pk["fx"], pk["fy"], pk["cx"], pk["cy"]
W_cam, H_cam   = pk["W"], pk["H"]
print(f"\nPinhole K: fx={FX:.1f}, fy={FY:.1f}, cx={CX:.1f}, cy={CY:.1f}  ({W_cam}x{H_cam})")

K_mat = torch.tensor([[FX, 0, CX],
                      [0, FY, CY],
                      [0,  0,  1]], dtype=torch.float32, device=device)

# ── COLMAP binary readers ──────────────────────────────────────────────────────
def _read_cameras(path):
    cameras = {}
    with open(path, "rb") as f:
        n = struct.unpack("Q", f.read(8))[0]
        for _ in range(n):
            cid   = struct.unpack("I", f.read(4))[0]
            model = struct.unpack("I", f.read(4))[0]
            W_    = struct.unpack("Q", f.read(8))[0]
            H_    = struct.unpack("Q", f.read(8))[0]
            npar  = {0:3,1:4,2:5,3:6,4:8,5:8,6:9}.get(model, 4)
            par   = struct.unpack(f"{npar}d", f.read(npar*8))
            cameras[cid] = dict(model=model, W=W_, H=H_, params=par)
    return cameras

def _read_images(path):
    imgs = {}
    with open(path, "rb") as f:
        n = struct.unpack("Q", f.read(8))[0]
        for _ in range(n):
            iid   = struct.unpack("I", f.read(4))[0]
            qvec  = struct.unpack("4d", f.read(32))
            tvec  = struct.unpack("3d", f.read(24))
            cid   = struct.unpack("I", f.read(4))[0]
            name  = b""
            while True:
                c = f.read(1)
                if c == b"\x00": break
                name += c
            n2d = struct.unpack("Q", f.read(8))[0]
            struct.unpack(f"{2*n2d}d", f.read(16*n2d))
            struct.unpack(f"{n2d}q",   f.read(8*n2d))
            imgs[iid] = dict(qvec=np.array(qvec), tvec=np.array(tvec),
                             cid=cid, name=name.decode())
    return imgs

def qvec2R(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]
    ], dtype=np.float32)

print("\nLoading COLMAP sparse reconstruction ...")
imgs_data = _read_images(os.path.join(SPARSE_DIR, "images.bin"))
print(f"  {len(imgs_data)} images registered")

# Match COLMAP image names to undistorted frames
undist_files = sorted([f for f in os.listdir(IMAGES_DIR)
                       if f.endswith(('.jpg', '.png'))])
print(f"  {len(undist_files)} undistorted frames found")

# Build views: sort by name, match by index
class View: pass
views = []
sorted_img_items = sorted(imgs_data.items(), key=lambda x: x[1]["name"])
for idx, (iid, idata) in enumerate(sorted_img_items):
    v = View()
    v.W, v.H   = W_cam, H_cam
    v.K        = K_mat

    R    = qvec2R(idata["qvec"])
    t    = idata["tvec"].astype(np.float32)
    vm   = np.eye(4, dtype=np.float32)
    vm[:3, :3] = R
    vm[:3,  3] = t
    v.viewmat = torch.from_numpy(vm).to(device)
    v.colmap_name = idata["name"]

    # Map to undistorted frame by index
    if idx < len(undist_files):
        v.img_path = os.path.join(IMAGES_DIR, undist_files[idx])
        v.frame_name = undist_files[idx]
    else:
        v.img_path = None
        v.frame_name = f"frame_{idx:04d}"
    views.append(v)

print(f"  {len(views)} views ready")

# ── Render helper ─────────────────────────────────────────────────────────────
def render_view(view):
    with torch.no_grad():
        renders, alphas, info = rasterization(
            means      = means_t,
            quats      = F.normalize(quats_t, dim=-1),
            scales     = torch.exp(scales_t),
            opacities  = torch.sigmoid(opacities_t),
            colors     = sh_t,
            viewmats   = view.viewmat.unsqueeze(0),
            Ks         = view.K.unsqueeze(0),
            width      = view.W,
            height     = view.H,
            sh_degree  = 3,
            near_plane = 0.01,
            far_plane  = 1000.0,
            backgrounds= torch.zeros(1, 3, device=device),   # black bg matches training
            render_mode= "RGB",
            absgrad    = False,
            packed     = False,
        )
    img = renders[0].clamp(0, 1).cpu().numpy()   # [H, W, 3]
    return (img * 255).astype(np.uint8)

# ── Render every 10th view (12 comparison images) ─────────────────────────────
RENDER_EVERY = 10
render_views = views[::RENDER_EVERY]
print(f"\nRendering {len(render_views)} comparison frames (every {RENDER_EVERY}th) ...")

psnr_list  = []
min_psnr   = float("inf")
max_psnr   = float("-inf")
best_idx   = 0
t_render   = time.time()

for i, vw in enumerate(render_views):
    render = render_view(vw)

    # Ground truth
    if vw.img_path and os.path.exists(vw.img_path):
        gt = np.array(Image.open(vw.img_path).convert("RGB")
                      .resize((vw.W, vw.H), Image.LANCZOS))
        mse  = np.mean((render.astype(np.float32) - gt.astype(np.float32))**2)
        psnr = 20*math.log10(255.0) - 10*math.log10(mse) if mse > 0 else 100.0
        psnr_list.append(psnr)
        if psnr < min_psnr: min_psnr = psnr
        if psnr > max_psnr: max_psnr = psnr; best_idx = i

        # Side-by-side: GT (left) | Render (right), with PSNR label
        comp = np.concatenate([gt, render], axis=1)
        comp_img = Image.fromarray(comp)
        draw = ImageDraw.Draw(comp_img)
        label = f"GT (left)  |  3DGS v3 (right)   PSNR={psnr:.2f} dB"
        draw.text((10, 10), label, fill=(255, 50, 50))
        out_name = f"comp_{i:02d}_{os.path.splitext(vw.frame_name)[0]}.png"
        comp_img.save(os.path.join(OUT_DIR, out_name))
        print(f"  [{i:02d}] {vw.frame_name}: PSNR = {psnr:.2f} dB  → {out_name}")
    else:
        # No GT — save render only
        out_name = f"render_{i:02d}_{vw.frame_name}.png"
        Image.fromarray(render).save(os.path.join(OUT_DIR, out_name))
        print(f"  [{i:02d}] {vw.frame_name}: (no GT)")

render_elapsed = time.time() - t_render
mean_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
print(f"\nMean PSNR : {mean_psnr:.2f} dB")
print(f"Min  PSNR : {min_psnr:.2f} dB")
print(f"Max  PSNR : {max_psnr:.2f} dB")
print(f"Render time: {render_elapsed:.1f}s")

# ── Save metrics ───────────────────────────────────────────────────────────────
metrics = {
    "model": "3DGS v3 (50K iters, undistorted, clean.ply)",
    "num_gaussians": N,
    "mean_psnr_db": mean_psnr,
    "min_psnr_db": min_psnr,
    "max_psnr_db": max_psnr,
    "num_views": len(psnr_list),
    "psnr_list": [float(p) for p in psnr_list],
    "render_time_s": render_elapsed,
}
metrics_path = os.path.join(OUT_DIR, "render_metrics_v3.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nMetrics saved → {metrics_path}")
print(f"All renders  → {OUT_DIR}")
print("\nDone!")
