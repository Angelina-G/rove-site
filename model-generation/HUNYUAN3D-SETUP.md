# Hunyuan3D-2mv 多视图生成：本机部署与实测记录

本文件记录 2026-11-05 在本机（联想拯救者 Y9000X，RTX 4070 Laptop 8 GB）把 Tencent
Hunyuan3D-2mv 多视图形体模型跑通的全过程、实测数据和复现命令。模型只做形体（shape），
不含纹理、骨架或形态键。

## 结论

- **跑通了**：四视图（正/左/右/背）真实抠图输入，octree 320、turbo 权重 5 步推理，
  产出 `output/jobs/hunyuan3d-2mv-v2/hunyuan3d-2mv.glb`：132,176 顶点 / 264,356 面，
  闭合（watertight），单连通块。
- **显存够用**：`torch.cuda.max_memory_allocated` 峰值 4.89 GB，reserved 5.79 GB，
  卡总显存 8 GB；mmgp profile 4 的低显存 offload 生效。
- **耗时**：模型加载 20.99 s，mmgp offload 7.89 s，生成 71.69 s，导出 3.1 s。
- **已修复**：第一版（`hunyuan3d-2mv-turntable/`，octree 256）双腿不完整，根因是抠图
  腿/鞋补回只保留最大连通块（丢了一条腿）+ left/right 两张侧视图其实来自同一侧；
  现已修正为保留所有腿部件，并用 9e01（左）/ 52a9（右）作为左右视图。
- **已知问题**：照片里人物手里拿的布料/道具仍被抠进了蒙版（人体解析把它判成上衣/裙子），
  所以生成模型的双手位置带有布料团块；这是输入蒙版问题，不是模型本身的问题。

## 目录与文件

| 项目 | 路径 |
|---|---|
| 代码（社区低显存 fork） | `rove-site/model-generation/source/Hunyuan3D-2GP` |
| Python 环境 | `rove-site/model-generation/envs/hunyuan3d`（Python 3.10.16 + torch 2.5.1+cu124） |
| 权重 | `rove-site/model-generation/weights/hunyuan3d-2mv/hunyuan3d-dit-v2-mv-turbo/` |
| 生成脚本 | `rove-site/model-generation/multiview_worker.py` |
| 产物 | `rove-site/model-generation/output/jobs/hunyuan3d-2mv-turntable/` |

权重来源为 ModelScope 镜像仓库 `Tencent-Hunyuan/Hunyuan3D-2mv`：
- `hunyuan3d-dit-v2-mv-turbo/model.fp16.safetensors`，4,930,777,530 字节（约 4.6 GiB），
  sha256 `172d7a989b99f66af760da0080d8c6f6350c72609536cf0c55b87a4afa574e45`
- `hunyuan3d-dit-v2-mv-turbo/config.yaml`
- 同仓库还有非 turbo 的 `hunyuan3d-dit-v2-mv`；本次为降低 8 GB 显存下的耗时选了 turbo（5 步）。

> 许可：Tencent Hunyuan3D 使用腾讯社区许可，含地域和使用条款；公开分发或商业化前必须复核
> `source/Hunyuan3D-2GP/LICENSE` 与上游 `LICENSE`，不要默认它等同于 MIT。

## 安装步骤（已执行，供复现）

```powershell
# 1) 环境（已存在则跳过）
uv venv 'rove-site\model-generation\envs\hunyuan3d' --python 3.10.16

# 2) 基础 torch（cu124）
uv pip install --python 'rove-site\model-generation\envs\hunyuan3d\Scripts\python.exe' `
  torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 3) 其余依赖（国内镜像；uv cache 指到工程内，避免用户目录权限问题）
$env:UV_CACHE_DIR = (Resolve-Path 'rove-site\model-generation').Path + '\envs\hunyuan3d-uv-cache'
uv pip install --python 'rove-site\model-generation\envs\hunyuan3d\Scripts\python.exe' `
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple `
  diffusers einops opencv-python omegaconf trimesh pymeshlab pygltflib xatlas accelerate `
  mmgp==3.2.7 scikit-image safetensors transformers==4.49.0 tqdm timm pyyaml
```

要点：
- **不要装 `diso`**：它是自定义 CUDA 扩展，Windows 上需要编译；`surface_extractors.py`
  里有 `skimage.measure.marching_cubes` 兜底，直接走兜底即可。
- **HTTPS 下载用 Python**：本机 `curl.exe` / .NET `Invoke-WebRequest` 会因 schannel
  凭据问题失败，`huggingface_hub` / `urllib` 正常。权重从 ModelScope 下载更快。
- pymeshlab 2025 版在本机调用 `save_current_mesh` 会抛 `PyMeshLabException`，
  已把 `FloaterRemover` 后处理换成 trimesh 自带实现（见 `multiview_worker.py` 的 cleanup 段）。

## 运行命令

```powershell
cd 'D:\Desktop\01_项目与产品\Rove如晤_跨时空交互\rove-site\model-generation'
$env:HY3DGEN_MODELS = (Resolve-Path '.\weights').Path
$env:HF_HOME = (Resolve-Path '.\cache\huggingface').Path

# 快速冒烟：只喂正视图，octree 128，验证环境/显存
.\envs\hunyuan3d\Scripts\python.exe -u multiview_worker.py --preset smoke

# 正式四视图：octree 256，turbo 5 步，低显存 offload
.\envs\hunyuan3d\Scripts\python.exe -u multiview_worker.py --preset full --octree 256 --steps 5 --profile 4
```

脚本默认读取 `test-images/processed-v2/` 的抠图（存在 `mask-manifest.json` 时优先），
视角映射写在 `VIEW_FILES`：front=9ca2f019、left=52a9f704、right=fc555d3e、back=01139015。

## 产物

```
output/jobs/hunyuan3d-2mv-turntable/
├── inputs/{front,left,right,back}.png   # 真正送进模型的 512×512 视图
├── hunyuan3d-2mv.glb                    # 2.98 MB，82,616 顶点 / 165,256 面
├── generation-report.json               # 参数、耗时、显存、网格校验、每阶段记录
├── stage-log.txt                        # ROVE_STAGE:... 逐阶段日志
├── hunyuan3d-2mv-preview.png            # 本机正交渲染的四视图预览
└── hunyuan3d-2mv-view1..4.png           # 单视角渲染图
```

## 下一步

1. **修蒙版里的手持道具**：`mask_v2_pipeline.py` 的 `IMAGE_CONFIG[*].prop_rois` 支持人工多边形，
   可先对手部布料做精确 ROI，再重跑 `mask_v2_pipeline.py fuse qc` 与
   `multiview_worker.py`。
2. 显存还有余量，可试 octree 320 或非 turbo 权重做质量对照。
3. 形体验收通过后再接 Hunyuan3D-Paint 纹理；8 GB 显存下纹理阶段需要另行评估。
4. 绑定（骨架/手指/形态键）仍未做，生成模型目前不能直接驱动作捕。
