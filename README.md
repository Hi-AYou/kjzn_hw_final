# HW3: 多源3D资产生成与真实场景融合

深度学习与空间智能课程 第三次作业代码仓库。

## 📊 主要结果

| 物体 | 方法 | 关键指标 |
|------|------|----------|
| A | 3D Gaussian Splatting (gsplat 1.5.3, 50K iter) | PSNR **33.00 dB**，594K高斯球，训练 **13.5 min** |
| B | DreamFusion + Perp-Neg (SD 1.5, 15K step) | Janus问题有效消除，训练 **58 min** |
| C | Zero123++ + TripoSR | 推理总耗时仅 **32秒** |

## 🔗 外部链接

- **模型权重 & 大文件**：https://drive.google.com/drive/folders/1MwAw3u8L82WEmT9QWiWHFWnTHp8FljmB?usp=sharing
- **WandB 训练曲线**：
  - 物体A (3DGS)：https://wandb.ai/cyi15772-fudan-university/hw3-3dgs/runs/ss9fw0i9
  - 物体B (DreamFusion v5)：https://wandb.ai/cyi15772-fudan-university/hw3-3dgs/runs/3ciqzy0y

---

## 📁 仓库结构

```
├── objectA/                     # 物体A：真实多视角重建（3DGS）
│   ├── extract_frames.py        # 从视频提取帧
│   ├── run_colmap.py            # COLMAP SfM 位姿估计
│   ├── retrain_3dgs_v3.py       # ★ 最终3DGS训练脚本（50K步）
│   ├── render_gs_v3.py          # ★ 渲染对比图生成脚本
│   ├── run_objectA_pipeline.sh  # 一键运行全流程
│   ├── sparse/0/                # COLMAP 稀疏重建结果
│   ├── colmap_summary.json      # COLMAP 统计（重投影误差等）
│   ├── gs_renders_v3/           # ★ GT vs 渲染对比图（12张）
│   └── gs_output_v3/            # 训练曲线 & 指标（PLY 权重见 Google Drive）
│
├── objectB/                     # 物体B：文本到3D（DreamFusion + Perp-Neg）
│   ├── run_objectB_v5.py        # ★ 最终训练脚本（v5, Perp-Neg, 15K步）
│   ├── timing_v5.json           # 训练耗时记录
│   ├── v5_render_*.png          # ★ 最终渲染图（6个视角）
│   └── v5_it15000-test.mp4      # ★ 360°旋转视频
│
├── objectC/                     # 物体C：单图到3D（Zero123++ + TripoSR）
│   ├── remove_bg.py             # rembg 背景去除
│   ├── run_zero123pp.py         # Zero123++ 多视角生成
│   ├── run_triposr.py           # TripoSR 3D 重建
│   ├── run_objectC_pipeline.sh  # 一键运行全流程
│   ├── object_C_256_rgba.png    # 处理后的输入图（去背景，256×256）
│   ├── object_C_256_white.png   # 白底版本
│   ├── zero123_output/          # Zero123++ 生成的6个新视角
│   └── triposr_output/          # TripoSR 重建结果（mesh.obj 见 Google Drive）
│
├── report/                      # LaTeX 实验报告
│   ├── hw3_report.tex           # ★ 主报告源文件
│   └── neurips_2026.sty         # NeurIPS 2026 模板
│
├── wandb_figures/               # WandB 导出的训练曲线图（7张）
├── pip_key_packages.txt         # 关键 Python 包版本列表
├── run_pipeline.sh              # 串联流水线（objectB训练 → objectA渲染）
└── .gitignore
```

---

## ⚙️ 环境配置（Requirements）

### 系统要求

- Python 3.9
- CUDA 12.1（已在 NVIDIA H20 上验证）
- GCC 7+，CMake 3.20+

### 安装依赖

```bash
# 1. 安装 PyTorch（CUDA 12.1）
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. 核心依赖
pip install "numpy<2" gsplat==1.5.3 plyfile==1.1.3 pycolmap==0.6.1

# 3. 扩散模型
pip install diffusers==0.26.0 transformers==4.37.2 accelerate

# 4. 训练框架
pip install lightning==2.6.0 omegaconf jaxtyping einops trimesh

# 5. 工具包
pip install rembg[gpu] wandb==0.16.0 tensorboard \
    opencv-python-headless xatlas typeguard envlight

# 6. 编译型依赖（需联网）
pip install git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# 7. 物体B 依赖：克隆 threestudio
git clone https://github.com/threestudio-project/threestudio.git

# 8. 物体C 依赖：克隆 TripoSR
git clone https://github.com/VAST-AI-Research/TripoSR.git
```

### 环境变量

```bash
export HF_HOME=/path/to/model_cache      # HuggingFace 模型缓存目录
export TORCH_HOME=/path/to/model_cache
export WANDB_API_KEY=<your_wandb_api_key>
```

---

## 📂 数据准备

### 物体A（真实多视角重建）

用手机以**环绕方式**拍摄目标物体约1分钟，将视频文件命名为 `物体A视频.mov` 放入 `objectA/` 目录：

```
objectA/
└── 物体A视频.mov   ← 放入手机拍摄的环绕视频
```

脚本会自动从视频中均匀提取120帧作为训练输入。

### 物体B（文本到3D）

无需任何数据集。修改 `objectB/run_objectB_v5.py` 中的 `PROMPT` 变量为目标物体的文本描述即可。

### 物体C（单图到3D）

准备一张目标物体的清晰照片，命名为 `object_C_original.jpg` 放入 `objectC/` 目录：

```
objectC/
└── object_C_original.jpg   ← 放入物体照片（建议简洁背景，分辨率 ≥ 512×512）
```

---

## 🏋️ 训练（Train）

### 物体A：3DGS 多视角重建

```bash
cd objectA

# Step 1：从视频提取120帧
python extract_frames.py

# Step 2：COLMAP SfM 位姿估计（约4分钟）
python run_colmap.py

# Step 3：3DGS 训练（50K步，H20 约13.5分钟）
python retrain_3dgs_v3.py
```

一键运行：

```bash
nohup bash objectA/run_objectA_pipeline.sh > logs/objectA.log 2>&1 &
```

### 物体B：DreamFusion 文本到3D

```bash
# 训练（15K步，H20 约58分钟）
nohup python objectB/run_objectB_v5.py > logs/objectB_v5.log 2>&1 &
```

### 物体C：Zero123++ + TripoSR

```bash
# 一键推理（背景去除 + Zero123++ + TripoSR，约32秒）
bash objectC/run_objectC_pipeline.sh
```

---

## 🧪 测试/推理（Test / Inference）

### 物体A：渲染对比图（评估重建质量）

```bash
# 从训练好的高斯模型渲染12个视角，输出 GT vs 渲染对比图
python objectA/render_gs_v3.py

# 结果保存至：objectA/gs_renders_v3/comp_*.png
# 指标记录于：objectA/gs_renders_v3/render_metrics_v3.json
```

### 物体B：查看渲染结果 / 导出网格

```bash
# 训练完成后渲染图自动保存：
# objectB/v5_render_*.png      - 6个视角静态图
# objectB/v5_it15000-test.mp4  - 360°旋转视频

# 从 checkpoint 导出 .obj 网格：
cd threestudio
python launch.py \
    --config ../objectB/outputs_v5/dreamfusion-sd/corgi_v5@*/configs/parsed.yaml \
    --export --gpu 0 \
    resume=../objectB/outputs_v5/dreamfusion-sd/corgi_v5@*/ckpts/last.ckpt \
    system.exporter_type=mesh-exporter \
    system.exporter.fmt=obj \
    exp_root_dir=../objectB/export tag=mesh_export
```

### 物体C：查看重建结果

```bash
# Zero123++ 生成的6视角
ls objectC/zero123_output/view_*.png

# TripoSR 重建的网格文件（从 Google Drive 下载）：
# objectC/triposr_output/0/mesh.obj  (8.5 MB)
```

---

## 📈 实验指标汇总

| 物体 | 指标 | 数值 |
|------|------|------|
| A | 均值 PSNR（120视角） | **33.00 dB** |
| A | 高斯球数量（最终） | 594,197 |
| A | 训练时间（H20） | 13.5 min |
| A | COLMAP 重投影误差 | 0.54 px |
| B | 训练步数 | 15,000 |
| B | 训练时间（H20） | 58.0 min |
| B | Janus 问题修复 | Perp-Neg + view-dependent |
| C | Zero123++ 推理时间 | 19.7 s |
| C | TripoSR 推理时间 | 12.0 s |
| C | 总推理时间 | ~32 s |
