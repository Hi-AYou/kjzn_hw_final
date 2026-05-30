"""
3DGS v3: High-quality training
- Full resolution (606x1080)
- 50K iterations  
- Aggressive floater pruning
- Opacity reset
- Lower SH degree start
"""
import os, sys, json, time, math, struct, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from gsplat import rasterization, DefaultStrategy
from gsplat.strategy.ops import remove as gs_remove

device = torch.device("cuda")
print(f"gsplat | PyTorch {torch.__version__} | GPU mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

BASE_DIR   = "/apdcephfs_hzlf/share_1227201/easyi/objectA"
IMAGE_DIR  = os.path.join(BASE_DIR, "images_undistorted")
SPARSE_DIR = os.path.join(BASE_DIR, "sparse/0")
OUTPUT_DIR = os.path.join(BASE_DIR, "gs_output_v3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

pinhole_K = json.load(open(os.path.join(IMAGE_DIR, "pinhole_K.json")))

CFG = {
    "iterations":       50000,      # more iterations for convergence
    "lr_means":         8e-5,       # stable
    "lr_scales":        3e-3,
    "lr_quats":         8e-4,
    "lr_opacities":     3e-2,
    "lr_sh_dc":         2.5e-3,
    "lr_sh_rest":       2.5e-3 / 20,
    "sh_degree":        3,
    "lambda_dssim":     0.2,
    "white_background": False,
    "log_interval":     500,
    "save_iters":       [15000, 30000, 50000],
}
NUM_SH = (CFG["sh_degree"] + 1) ** 2

# ── COLMAP readers (same as before) ──
def _read_cameras(path):
    cameras = {}
    with open(path, "rb") as f:
        n = struct.unpack("Q", f.read(8))[0]
        for _ in range(n):
            cid=struct.unpack("I",f.read(4))[0]; model=struct.unpack("I",f.read(4))[0]
            W=struct.unpack("Q",f.read(8))[0]; H=struct.unpack("Q",f.read(8))[0]
            npar={0:3,1:4,2:5,3:6,4:8,5:8,6:9}.get(model,4)
            par=struct.unpack(f"{npar}d",f.read(npar*8))
            cameras[cid]=dict(model=model,W=W,H=H,params=par)
    return cameras
def _read_images(path):
    imgs = {}
    with open(path, "rb") as f:
        n = struct.unpack("Q", f.read(8))[0]
        for _ in range(n):
            iid=struct.unpack("I",f.read(4))[0]; qvec=struct.unpack("4d",f.read(32))
            tvec=struct.unpack("3d",f.read(24)); cid=struct.unpack("I",f.read(4))[0]
            name=b""
            while True:
                c=f.read(1)
                if c==b"\x00": break
                name+=c
            n2d=struct.unpack("Q",f.read(8))[0]
            _=struct.unpack(f"{2*n2d}d",f.read(16*n2d)); _=struct.unpack(f"{n2d}q",f.read(8*n2d))
            imgs[iid]=dict(qvec=np.array(qvec),tvec=np.array(tvec),cid=cid,name=name.decode())
    return imgs
def _read_points3d(path):
    pts = {}
    with open(path, "rb") as f:
        n = struct.unpack("Q", f.read(8))[0]
        for _ in range(n):
            pid=struct.unpack("Q",f.read(8))[0]; xyz=struct.unpack("3d",f.read(24))
            rgb=struct.unpack("3B",f.read(3)); err=struct.unpack("d",f.read(8))[0]
            tlen=struct.unpack("Q",f.read(8))[0]; _=struct.unpack(f"{2*tlen}I",f.read(8*tlen))
            pts[pid]=dict(xyz=np.array(xyz),rgb=np.array(rgb))
    return pts
def qvec2R(qvec):
    qw,qx,qy,qz=qvec
    return np.array([[1-2*(qy**2+qz**2),2*(qx*qy-qw*qz),2*(qx*qz+qw*qy)],
                     [2*(qx*qy+qw*qz),1-2*(qx**2+qz**2),2*(qy*qz-qw*qx)],
                     [2*(qx*qz-qw*qy),2*(qy*qz+qw*qx),1-2*(qx**2+qy**2)]])
def simple_ssim(x,y,window_size=11):
    C1,C2=0.01**2,0.03**2; pad=window_size//2
    mu_x=F.avg_pool2d(x.unsqueeze(0),window_size,1,pad)[0]
    mu_y=F.avg_pool2d(y.unsqueeze(0),window_size,1,pad)[0]
    sigma_x2=F.avg_pool2d((x**2).unsqueeze(0),window_size,1,pad)[0]-mu_x**2
    sigma_y2=F.avg_pool2d((y**2).unsqueeze(0),window_size,1,pad)[0]-mu_y**2
    sigma_xy=F.avg_pool2d((x*y).unsqueeze(0),window_size,1,pad)[0]-mu_x*mu_y
    ssim_map=((2*mu_x*mu_y+C1)*(2*sigma_xy+C2))/((mu_x**2+mu_y**2+C1)*(sigma_x2+sigma_y2+C2))
    return 1.0-ssim_map.mean()

# ── Load data ──
print("\nLoading COLMAP data ...")
cams_data=_read_cameras(os.path.join(SPARSE_DIR,"cameras.bin"))
imgs_data=_read_images(os.path.join(SPARSE_DIR,"images.bin"))
pts3d_data=_read_points3d(os.path.join(SPARSE_DIR,"points3D.bin"))
print(f"  {len(imgs_data)} images | {len(pts3d_data)} 3-D points")

fx,fy=pinhole_K['fx'],pinhole_K['fy']
cx,cy=pinhole_K['cx'],pinhole_K['cy']
W,H=pinhole_K['W'],pinhole_K['H']

class View: pass
views=[]
for iid,idata in imgs_data.items():
    v=View(); v.W,v.H=W,H; v.fx,v.fy=fx,fy; v.cx,v.cy=cx,cy
    R=qvec2R(idata["qvec"]).astype(np.float32); t=idata["tvec"].astype(np.float32)
    vm=np.eye(4,dtype=np.float32); vm[:3,:3]=R; vm[:3,3]=t
    v.viewmat=torch.from_numpy(vm).to(device)
    K=np.zeros((3,3),dtype=np.float32); K[0,0]=fx; K[1,1]=fy; K[0,2]=cx; K[1,2]=cy; K[2,2]=1
    v.K=torch.from_numpy(K).to(device)
    img_path=os.path.join(IMAGE_DIR,idata["name"])
    if not os.path.exists(img_path):
        candidates=sorted([c for c in os.listdir(IMAGE_DIR) if c.endswith(('.jpg','.png'))])
        idx=list(imgs_data.keys()).index(iid)
        if idx<len(candidates): img_path=os.path.join(IMAGE_DIR,candidates[idx])
    if not os.path.exists(img_path): continue
    v.img_path=img_path; views.append(v)
print(f"  {len(views)} views loaded")

print("  Loading images ...")
for v in tqdm(views):
    img=np.array(Image.open(v.img_path).convert("RGB"),dtype=np.float32)/255.
    v.img=torch.from_numpy(img).permute(2,0,1).to(device)

# ── Init Gaussians ──
print("\nInitializing Gaussians ...")
xyz=np.stack([p["xyz"] for p in pts3d_data.values()]).astype(np.float32)
rgb=np.stack([p["rgb"] for p in pts3d_data.values()]).astype(np.float32)/255.
N=len(xyz)
sub=min(N,4096); si=np.random.choice(N,sub,replace=False)
sm=torch.from_numpy(xyz[si]).to(device); D=torch.cdist(sm,sm); D.fill_diagonal_(1e9)
avg_d=D.topk(4,dim=1,largest=False).values.mean().item()

SH_C0=0.28209479177387814
means_param=nn.Parameter(torch.from_numpy(xyz).to(device))
sh_dc=(torch.from_numpy(rgb).to(device)-0.5)/SH_C0
sh_rest=torch.zeros(N,NUM_SH-1,3,device=device)
sh_param=nn.Parameter(torch.cat([sh_dc.unsqueeze(1),sh_rest],dim=1))
log_scales_param=nn.Parameter(torch.full((N,3),math.log(avg_d),device=device))
quats_param=nn.Parameter(torch.cat([torch.ones(N,1),torch.zeros(N,3)],dim=1).to(device))
logits_opa=nn.Parameter(torch.full((N,),math.log(0.1/0.9),device=device))

params={"means":means_param,"sh":sh_param,"scales":log_scales_param,"quats":quats_param,"opacities":logits_opa}
print(f"  {N} Gaussians | avg_scale={avg_d:.5f}")

xyz_extent=torch.from_numpy(xyz).max(0).values-torch.from_numpy(xyz).min(0).values
scene_scale=float(xyz_extent.max().item())/2.0

# Aggressive strategy to reduce floaters
strategy=DefaultStrategy(
    grow_grad2d=0.0002,
    grow_scale3d=0.05,       # tighter threshold for splitting
    prune_scale3d=0.3,       # prune large Gaussians more aggressively  
    refine_start_iter=500,
    refine_every=100,
    refine_stop_iter=25000,  # continue refining longer
    reset_every=3000,        # OPACITY RESET every 3000 steps → kills floaters
    verbose=False,
)
strategy_state=strategy.initialize_state(scene_scale=scene_scale)

optimizers={
    "means":torch.optim.Adam([params["means"]],lr=CFG["lr_means"],eps=1e-15),
    "sh":torch.optim.Adam([params["sh"]],lr=CFG["lr_sh_dc"],eps=1e-15),
    "scales":torch.optim.Adam([params["scales"]],lr=CFG["lr_scales"],eps=1e-15),
    "quats":torch.optim.Adam([params["quats"]],lr=CFG["lr_quats"],eps=1e-15),
    "opacities":torch.optim.Adam([params["opacities"]],lr=CFG["lr_opacities"],eps=1e-15),
}

# ── Training ──
print(f"\n{'='*60}\n3DGS v3 (high quality) | {CFG['iterations']} iters | {W}x{H}\n{'='*60}")
t0=time.time(); losses=[]; psnrs=[]

for step in range(CFG["iterations"]):
    v=views[step%len(views)]
    sh_degree_active=min(CFG["sh_degree"], step//1500)  # slower SH activation
    
    renders,alphas,info=rasterization(
        means=params["means"],quats=F.normalize(params["quats"],dim=-1),
        scales=torch.exp(params["scales"]),opacities=torch.sigmoid(params["opacities"]),
        colors=params["sh"],viewmats=v.viewmat.unsqueeze(0),Ks=v.K.unsqueeze(0),
        width=v.W,height=v.H,sh_degree=sh_degree_active,
        near_plane=0.01,far_plane=1000.,
        backgrounds=torch.zeros(1,3,device=device),
        render_mode="RGB",absgrad=strategy.absgrad,packed=False,
    )
    pred=renders[0].permute(2,0,1)
    gt=v.img
    if gt.shape[1:]!=pred.shape[1:]:
        gt=F.interpolate(gt.unsqueeze(0),pred.shape[1:],mode="bilinear",align_corners=False)[0]

    l1=F.l1_loss(pred,gt); dssim=simple_ssim(pred,gt)
    loss=(1.0-CFG["lambda_dssim"])*l1+CFG["lambda_dssim"]*dssim

    for opt in optimizers.values(): opt.zero_grad()
    strategy.step_pre_backward(params=params,optimizers=optimizers,state=strategy_state,step=step,info=info)
    loss.backward()
    for opt in optimizers.values(): opt.step()
    strategy.step_post_backward(params=params,optimizers=optimizers,state=strategy_state,step=step,info=info,packed=False)

    with torch.no_grad():
        mse=F.mse_loss(pred,gt).item()
        psnr=-10*math.log10(mse) if mse>0 else 100
    losses.append(loss.item()); psnrs.append(psnr)

    if step%CFG["log_interval"]==0 or step==CFG["iterations"]-1:
        elapsed=(time.time()-t0)/60
        print(f"  [{step:>5}/{CFG['iterations']}] loss={loss.item():.4f} psnr={psnr:.2f}dB N={len(params['means'])} t={elapsed:.1f}min")

    if (step+1) in CFG["save_iters"]:
        save_dir=os.path.join(OUTPUT_DIR,f"point_cloud/iteration_{step+1}")
        os.makedirs(save_dir,exist_ok=True)
        N_cur=len(params["means"])
        xyz_np=params["means"].detach().cpu().numpy()
        sh_np=params["sh"].detach().cpu().numpy()
        ls_np=params["scales"].detach().cpu().numpy()
        qt_np=F.normalize(params["quats"],dim=-1).detach().cpu().numpy()
        opa_np=params["opacities"].detach().cpu().numpy()
        n_rest=NUM_SH-1
        dtypes=[('x','f4'),('y','f4'),('z','f4'),('nx','f4'),('ny','f4'),('nz','f4'),
                ('f_dc_0','f4'),('f_dc_1','f4'),('f_dc_2','f4'),
                *[(f'f_rest_{i}','f4') for i in range(n_rest*3)],
                ('opacity','f4'),('scale_0','f4'),('scale_1','f4'),('scale_2','f4'),
                ('rot_0','f4'),('rot_1','f4'),('rot_2','f4'),('rot_3','f4')]
        vdata=np.zeros(N_cur,dtype=dtypes)
        vdata['x']=xyz_np[:,0]; vdata['y']=xyz_np[:,1]; vdata['z']=xyz_np[:,2]
        vdata['nx']=vdata['ny']=vdata['nz']=0
        vdata['f_dc_0']=sh_np[:,0,0]; vdata['f_dc_1']=sh_np[:,0,1]; vdata['f_dc_2']=sh_np[:,0,2]
        sh_rest_np=sh_np[:,1:,:].reshape(N_cur,-1)
        for i in range(n_rest*3): vdata[f'f_rest_{i}']=sh_rest_np[:,i]
        vdata['opacity']=opa_np
        vdata['scale_0']=ls_np[:,0]; vdata['scale_1']=ls_np[:,1]; vdata['scale_2']=ls_np[:,2]
        vdata['rot_0']=qt_np[:,0]; vdata['rot_1']=qt_np[:,1]; vdata['rot_2']=qt_np[:,2]; vdata['rot_3']=qt_np[:,3]
        PlyData([PlyElement.describe(vdata,'vertex')]).write(os.path.join(save_dir,"point_cloud.ply"))
        print(f"     Saved {N_cur} Gaussians -> {save_dir}")

        # Post-hoc pruning: remove very low opacity Gaussians from saved PLY
        # (keeps training model intact but produces cleaner PLY for viewing)
        opa_sigmoid = 1.0 / (1.0 + np.exp(-opa_np))
        clean_mask = opa_sigmoid > 0.05  # remove Gaussians with opacity < 5%
        if clean_mask.sum() < N_cur:
            clean_vdata = vdata[clean_mask]
            clean_path = os.path.join(save_dir, "point_cloud_clean.ply")
            PlyData([PlyElement.describe(clean_vdata,'vertex')]).write(clean_path)
            print(f"     Clean PLY: {clean_mask.sum()} Gaussians (removed {N_cur-clean_mask.sum()} low-opacity)")

# ── Eval ──
total_min=(time.time()-t0)/60
print(f"\nTraining complete! {total_min:.1f} min | N={len(params['means'])} | PSNR={psnrs[-1]:.2f} dB")

print("\nFull evaluation...")
eval_psnrs=[]
with torch.no_grad():
    for v in views:
        renders,_,_=rasterization(means=params["means"],quats=F.normalize(params["quats"],dim=-1),
            scales=torch.exp(params["scales"]),opacities=torch.sigmoid(params["opacities"]),
            colors=params["sh"],viewmats=v.viewmat.unsqueeze(0),Ks=v.K.unsqueeze(0),
            width=v.W,height=v.H,sh_degree=3,near_plane=0.01,far_plane=1000.,
            backgrounds=torch.zeros(1,3,device=device),render_mode="RGB",packed=False)
        pred=renders[0].permute(2,0,1); gt=v.img
        if gt.shape[1:]!=pred.shape[1:]:
            gt=F.interpolate(gt.unsqueeze(0),pred.shape[1:],mode="bilinear",align_corners=False)[0]
        mse=F.mse_loss(pred,gt).item()
        eval_psnrs.append(-10*math.log10(mse) if mse>0 else 100)
mean_psnr=np.mean(eval_psnrs)
print(f"  Mean PSNR: {mean_psnr:.2f} dB | Min: {min(eval_psnrs):.2f} | Max: {max(eval_psnrs):.2f}")

metrics=dict(method="3DGS v3 (high quality, undistorted)",total_time_minutes=total_min,
    final_gaussians=len(params["means"]),final_psnr_db=psnrs[-1],mean_eval_psnr_db=float(mean_psnr),final_loss=losses[-1])
with open(os.path.join(OUTPUT_DIR,"metrics.json"),"w") as f: json.dump(metrics,f,indent=2)

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4))
ax1.plot(losses[::20]); ax1.set_title("Loss"); ax1.set_xlabel("Iter (x20)")
ax2.plot(psnrs[::20]); ax2.set_title("PSNR (dB)"); ax2.set_xlabel("Iter (x20)")
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,"training_curves.png"),dpi=100)
print(f"\nAll saved to {OUTPUT_DIR}")
