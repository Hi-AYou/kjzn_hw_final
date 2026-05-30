# HW3: 多源3D资产生成与真实场景融合

深度学习与空间智能课程 第三次作业代码仓库。

## 📁 仓库结构

```
├── objectA/                  # 物体A：真实多视角重建（3DGS）
│   ├── extract_frames.py     # 从视频提取帧
│   ├── run_colmap.py         # COLMAP SfM位姿估计
│   ├── retrain_3dgs_v3.py    # ★ 最终3DGS训练脚本（50K步）
│   ├── render_gs_v3.py       # ★ 渲染对比图生成
│   ├── run_objectA_pipeline.sh  # 一键运行全流程
│   ├── sparse/               # COLMAP稀疏重建结果
│   ├── colmap_summary.json   # COLMAP统计
│   ├── gs_renders_v3/        # ★ GT vs 渲染对比图（12张）
│   └── gs_output_v3/         # 训练曲线 & 指标（权重见GDrive）
│
├── objectB/                  # 物体B：文本到3D（DreamFusion）
│   ├── run_objectB_v5.py     # ★ 最终训练脚本（v5, Perp-Neg）
│   ├── timing_v5.json        # 训练耗时记录
│   └── v5_render_*.png       # ★ 最终渲染图（6张）
│
├── objectC/                  # 物体C：单图到3D（Zero123++ + TripoSR）
│   ├── remove_bg.py          # rembg背景去除
│   ├── run_zero123pp.py      # Zero123++多视角生成
│   ├── run_triposr.py        # TripoSR 3D重建
│   ├── run_objectC_pipeline.sh  # 一键运行全流程
│   ├── object_C_256_rgba.png # 处理后的输入图
│   ├── object_C_256_white.png
│   ├── zero123_output/       # 6视角生成结果
│   └── triposr_output/       # ★ 重建网格 mesh.obj
│
├── report/                   # LaTeX报告
│   ├── hw3_report.tex        # ★ 主报告
│   ├── neurips_2026.sty      # NeurIPS模板
│   └── checklist.tex
│
├── PROJECT_STATUS.md         # 实验记录与复现说明
├── pip_key_packages.txt      # 关键Python包版本
└── run_pipeline.sh           # 串联 objectB→objectA 渲染的流水线
```

## 🚀 复现说明

### 环境配置

```bash
# 基础环境（需要 CUDA 12.1 + Python 3.9）
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2" gsplat==1.5.3 plyfile pycolmap
pip install diffusers==0.26.0 transformers==4.37.2 accelerate
pip install lightning==2.6.0 omegaconf jaxtyping einops trimesh
pip install rembg[gpu] wandb tensorboard opencv-python-headless
pip install git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# 环境变量
export HF_HOME=/path/to/model_cache
export WANDB_API_KEY=<your_wandb_api_key>
```

### 物体A：3DGS多视角重建

```bash
cd objectA
# 1. 提取帧
python extract_frames.py
# 2. COLMAP SfM
python run_colmap.py
# 3. 3DGS训练（50K步，约13.5分钟）
python retrain_3dgs_v3.py
# 4. 渲染对比图
python render_gs_v3.py
```

### 物体B：DreamFusion文本到3D

需要先克隆 [threestudio](https://github.com/threestudio-project/threestudio) 到 `../threestudio/`：
```bash
cd objectB
python run_objectB_v5.py   # Perp-Neg + view-dependent, 15K步
```

### 物体C：单图到3D

需要先克隆 [TripoSR](https://github.com/VAST-AI-Research/TripoSR) 到 `../TripoSR/`：
```bash
cd objectC
bash run_objectC_pipeline.sh
```

## 📊 主要结果

| 物体 | 方法 | 关键指标 |
|------|------|----------|
| A | 3DGS (gsplat 1.5.3, 50K iter) | PSNR **33.00 dB**，594K高斯，13.5 min |
| B | DreamFusion + Perp-Neg (SD1.5, 15K step) | Janus问题消除，58 min |
| C | Zero123++ + TripoSR | 推理总时长 **32秒** |

## 🔗 相关链接

- 模型权重 & 大文件：见 Google Drive（链接由作者提供）
- WandB 训练曲线：
  - 物体A：https://wandb.ai/cyi15772-fudan-university/hw3-3dgs/runs/ss9fw0i9
  - 物体B：https://wandb.ai/cyi15772-fudan-university/hw3-3dgs/runs/3ciqzy0y
