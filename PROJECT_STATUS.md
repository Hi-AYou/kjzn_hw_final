# HW3 项目状态记录

> 深度学习与空间智能 - 第三次作业
> 最后更新：2026-05-24 18:15

---

## 一、项目概述

### 分工
- **我的部分**：物体A（真实多视角3DGS重建）、物体C（单图到3D）、质量评估、实验报告
- **队友部分**：物体B（文本到3D）由我代跑完成、背景场景重建、场景融合渲染、题目二 LeRobot

### 最终选择
| 物体 | 方法 | 输入 | 版本 |
|------|------|------|------|
| A | 3D Gaussian Splatting (gsplat 1.5.3) | 手机视频→120帧 | **v3** (50K iters, undistorted) |
| B | DreamFusion (threestudio + SD 1.5) | 文本prompt: 柯基犬 | **v4** (10K steps, negative prompt) |
| C | Zero123++ → TripoSR | 九尾狐手办照片 | 最新版 |

---

## 二、物体A：3DGS 真实多视角重建

### 最终结果 (v3)
- **PSNR**: 33.00 dB（全120视角均值），Min 29.11, Max 36.94
- **训练时间**: 13.5 分钟
- **高斯数量**: 594,197
- **训练步数**: 50,000

### 关键文件
```
objectA/
├── images/                          ← 原始视频帧 (120张, 606×1080)
├── images_undistorted/              ← ★ 去畸变后的帧 + pinhole_K.json
├── sparse/0/                        ← COLMAP 重建 (cameras.bin, images.bin, points3D.bin)
├── gs_output_v3/                    ← ★ 最终模型
│   ├── point_cloud/iteration_50000/
│   │   ├── point_cloud.ply          ← 完整模型 (141MB, 594K高斯)
│   │   └── point_cloud_clean.ply    ← ★ 清洁版 (95MB, 去除低opacity floaters)
│   ├── training_curves.png          ← matplotlib训练曲线
│   └── metrics.json                 ← 指标汇总
├── gs_renders_v2/                   ← 渲染对比图 (基于v2模型, 32 dB)
│   ├── comp_00~11_*.png             ← GT vs Render 对比 (12张)
│   └── render_metrics.json
├── train_3dgs_gsplat.py             ← 最初训练脚本 (v1, 有问题)
├── retrain_3dgs.py                  ← v2 脚本
└── retrain_3dgs_v3.py               ← ★ v3 最终脚本
```

### 技术要点
1. **去畸变是关键**：COLMAP检测到OPENCV相机模型 (k1=-0.097, k2=0.081)，不去畸变会导致训练不稳定(PSNR仅6.55dB)
2. **Opacity Reset**：每3000步重置opacity，有效消除floaters（"五彩斑斓的光"）
3. **查看PLY时用 `point_cloud_clean.ply`**，不要用 `point_cloud.ply`
4. **渲染对比图在 `gs_renders_v2/`**（基于v2模型32dB，v3的渲染图还未单独生成但质量更好）

### 遇到的问题及解决
| 问题 | 原因 | 解决 |
|------|------|------|
| v1训练PSNR仅6.55dB | 镜头畸变未校正 | 用OpenCV undistort所有图像 |
| 渲染全黑/纯色 | SH系数加载顺序+gsplat版本问题 | 用训练脚本相同的加载方式 |
| PLY打开有彩色光 | 低opacity floater高斯 | Opacity Reset + 生成clean PLY |
| 轨道新视角全黑 | 黑色背景+视角超出训练覆盖 | 3DGS固有限制，用训练视角对比图展示 |

---

## 三、物体B：DreamFusion 文本到3D

### 最终结果 (v4)
- **Prompt**: "A cute corgi dog, sitting, orange and white fur, two pointed ears, four short legs, fluffy round body, cute face, toy figurine, studio lighting, white background, 3D render"
- **Negative Prompt**: "three ears, extra ears, extra legs, deformed, mutated, ugly, blurry, bad anatomy, extra limbs, multiple heads, multiple faces"
- **训练时间**: 30.3 分钟
- **步数**: 10,000
- **分辨率**: 128×128
- **Guidance Scale**: 60.0
- **模型**: Stable Diffusion 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`)

### 关键文件
```
objectB/
├── v4_render_00~05.png              ← ★ 最终渲染图 (柯基犬)
├── v4_it10000-test.mp4              ← ★ 360°旋转视频
├── timing_v4.json                   ← 计时信息
├── outputs_v4/dreamfusion-sd/corgi_v4@.../
│   └── save/                        ← 所有中间渲染 + 视频
├── run_objectB_v4.py                ← ★ 最终运行脚本
└── (v1, v2, v3 旧版本也在，可清理)
```

### 技术要点
1. **SD 2.1 无法访问**（HuggingFace 401），改用 SD 1.5
2. **Janus Problem**：v3有三只耳朵，通过 negative prompt + 降低 guidance scale 在v4中改善
3. **DreamFusion 不输出 .obj mesh**：输出是隐式NeRF体积，需额外Marching Cubes才能导出mesh
4. **文件名过长问题**：需加 `tag=xxx` 参数避免threestudio用完整prompt作目录名

### 版本迭代
| 版本 | Prompt | 问题 |
|------|--------|------|
| v1 | display platform (原始) | SD 2.1不可用, 文件名太长, prompt太抽象→模糊blob |
| v2 | wooden display shelf | 有几何结构但很模糊 |
| v3 | corgi puppy | 三只耳朵(Janus问题), 脚畸形 |
| **v4** | corgi + negative prompt | ★ 两耳、五官清晰、柯基特征明确 |

---

## 四、物体C：单图到3D（Zero123++ + TripoSR）

### 最终结果
- **输入**: 九尾狐手办照片 (`物体C图片.jpg`, 4096×3072)
- **Zero123++ 时间**: 19.7秒 (75步diffusion, H20 GPU)
- **TripoSR 时间**: 12.0秒 (Marching Cubes 256分辨率)
- **总时间**: ~32秒

### 关键文件
```
objectC/
├── object_C_original.jpg            ← 九尾狐原图 (4096×3072)
├── object_C_256_rgba.png            ← ★ 去背景RGBA 256×256 (输入给模型)
├── object_C_256_white.png           ← 白底版
├── zero123_output/
│   ├── zero123pp_6views.png         ← ★ 6视角合成图
│   └── view_00~05.png              ← 单独视角
├── triposr_output/
│   └── 0/mesh.obj                   ← ★ 3D网格 (8.6MB)
├── remove_bg.py                     ← 背景去除脚本
├── run_zero123pp.py                 ← Zero123++脚本
└── run_triposr.py                   ← TripoSR脚本
```

### 技术要点
1. **rembg** 去背景，CPU fallback（CUDA onnxruntime有libcublas.so.11兼容问题）
2. **Zero123++** 使用 `sudo-ai/zero123plus-v1.2`，输入必须是256×256
3. **TripoSR** 用 `--no-remove-bg`（输入已去背景）+ `--model-save-format obj`
4. 需要预创建 `os.makedirs(output_dir/0)` 否则TripoSR报错

---

## 五、WandB 可视化

### 账号信息
- **用户**: cyi15772 (cyi15772-fudan-university)
- **项目**: hw3-3dgs

### 有效的 WandB Runs
| Run | 内容 | 链接 |
|-----|------|------|
| ObjectA-3DGS-v3 | Loss + PSNR + 验证集曲线 + 高斯数量 | https://wandb.ai/cyi15772-fudan-university/hw3-3dgs/runs/ss9fw0i9 |
| ObjectB-DreamFusion-Corgi-v2 | SDS Loss + 正则化Loss + 梯度范数 | https://wandb.ai/cyi15772-fudan-university/hw3-3dgs/runs/2mkoqxaa |

### 导出图表方法
1. 打开 run 链接
2. 在 Charts 面板找到目标图表
3. 鼠标悬停 → 右上角 **⋯** → **"Download plot as PNG"**
4. 放入报告

### 需要导出的图表
- **物体A**: `train/loss`, `train/psnr_db`, `val/mean_psnr_db`, `train/num_gaussians`
- **物体B**: `train/loss_sds`, `train/loss_total`, `train/grad_norm`

### 注意
- 可以删除的旧run: `ooi4jxpg`(物体A旧版), `8p1ejaop`(物体B有bug的版本)
- WandB登录方式：设置环境变量 `WANDB_API_KEY=<your_wandb_api_key>`

---

## 六、LaTeX 报告

### 当前状态
- 报告框架已写好：`report/hw3_report.tex`
- 使用 NeurIPS 2026 模板：`report/neurips_2026.sty`
- **需要更新**：报告中的数据仍是旧版（v1/v2的数据），需要用v3/v4最终数据更新
- Overleaf zip 在 `report/hw3_overleaf.zip`（需重新打包）

### 报告中需要更新的数据
| 项目 | 旧值 | 新值 |
|------|------|------|
| 物体A PSNR | 6.55 dB / 32.24 dB | **33.00 dB** |
| 物体A 训练时间 | 15.3 min / 8 min | **13.5 min** |
| 物体A 高斯数 | 461K / 380K | **594,197** |
| 物体A 迭代数 | 30K | **50,000** |
| 物体B prompt | display platform | **corgi dog** |
| 物体B 时间 | 21.7 min | **30.3 min** |
| 物体C 输入 | 存钱罐 | **九尾狐手办** |
| 物体C Zero123++时间 | 10.6s | **19.7s** |
| 物体C TripoSR时间 | 11.0s | **12.0s** |

### 报告需要包含的图表（从WandB导出）
1. 物体A: Loss曲线, PSNR曲线, 验证集指标
2. 物体B: SDS Loss曲线, 总Loss曲线
3. 物体A: GT vs Render 对比图 (从 `gs_renders_v2/comp_*.png`)
4. 物体B: 最终渲染图 (从 `objectB/v4_render_*.png`)
5. 物体C: 输入图 + 6视角 + mesh截图

---

## 七、环境恢复（新机器）

### 关键Python包
```bash
# 见 /apdcephfs_hzlf/share_1227201/easyi/pip_key_packages.txt
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2" gsplat==1.5.3 plyfile pycolmap
pip install diffusers==0.26.0 transformers==4.37.2 accelerate
pip install lightning==2.6.0 omegaconf jaxtyping einops trimesh
pip install rembg[gpu] xatlas controlnet-aux wandb tensorboard
pip install git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install /apdcephfs_hzlf/share_1227201/easyi/nvdiffrast/ --no-build-isolation
```

### 环境变量
```bash
export HF_HOME=/apdcephfs_hzlf/share_1227201/easyi/model_cache
export TORCH_HOME=/apdcephfs_hzlf/share_1227201/easyi/model_cache
export WANDB_API_KEY=<your_wandb_api_key>
```

### 特殊处理
1. **igl**: 无cp39-linux wheel，需创建stub模块到 `/usr/local/lib/python3.9/site-packages/igl/__init__.py`
2. **nvdiffrast**: pip安装后需手动copy Python包到site-packages + 创建dist-info
3. **wandb**: 0.16.0版本可用，更新版本build失败

---

## 八、待完成工作

1. ✅ ~~物体A v3 训练~~ → 完成 (33.00 dB)
2. ✅ ~~物体B v4 训练~~ → 完成 (柯基犬)
3. ✅ ~~物体C 重做~~ → 完成 (九尾狐)
4. ✅ ~~WandB 上传~~ → 完成
5. ⬜ **更新LaTeX报告**（用最终数据+图表）
6. ⬜ **重新打包Overleaf zip**
7. ⬜ **物体A v3 渲染对比图**（当前对比图用的v2模型，v3的还没渲染）
8. ⬜ 清理旧版本文件（v1/v2/v3 objectB旧结果等）
9. ⬜ GitHub push（由用户手动完成）
10. ⬜ Google Drive 上传（由用户手动完成）
