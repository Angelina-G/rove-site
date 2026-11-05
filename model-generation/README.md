# Rove 角色生成工作台：模型侧目录说明

这里是“照片 → 人物抠图 → 多视图几何 → GLB”的全部本机资产。网页工作台在
`../rove-model-studio.html`，本地服务在 `../model-api`（默认 <http://localhost:8790/>）。

## 目录

| 路径 | 内容 |
|---|---|
| `test-images/` | 10 张真人转身照片原图（未覆盖）、`face-detail/` 面部近景 |
| `test-images/processed-approved/` | 第一版抠图（双模型融合）与质检报告 |
| `test-images/processed-v2/` | 第二版抠图（多模型 + 人体解析先验，补回双脚），含 `mask-manifest.json` 与逐张放大自查 |
| `source/TripoSR/` | TripoSR 单视图重建代码（含 Windows CPU marching-cubes 兜底） |
| `source/Hunyuan3D-2GP/` | Hunyuan3D-2mv 社区低显存 fork（mmgp offload） |
| `envs/triposr/` | TripoSR 与 rembg 的隔离 Python 环境 |
| `envs/hunyuan3d/` | Hunyuan3D 的隔离 Python 环境（torch 2.5.1+cu124） |
| `weights/` | TripoSR `model.ckpt` 与 Hunyuan3D-2mv turbo 权重 |
| `output/jobs/` | 每个生成任务的输入/中间产物/GLB/报告 |
| `cache/` | rembg、HuggingFace、uv 等缓存 |

## 当前可交付的结果

| 任务 | 模型 | 结果 |
|---|---|---|
| `output/jobs/hunyuan3d-2mv-v2/` | Hunyuan3D-2mv（四视图，turbo fp16，octree 320） | 132,176 顶点 / 264,356 面，闭合；生成 71.7 s，显存峰值 4.89 GB（当前主结果） |
| `output/jobs/hunyuan3d-2mv-turntable/` | Hunyuan3D-2mv（第一版，octree 256） | 82,616 顶点 / 165,256 面；左右视图误用同一侧、左腿被截断，仅作对比 |
| `output/jobs/visual-hull-turntable/` | visual hull（四视图轮廓雕刻，CPU） | 22,596 顶点，闭合；用于检查“是否缺半、比例是否合理” |
| `output/jobs/turntable-fused-triposr-256/` | TripoSR（单视图基线） | 19,067 顶点 / 38,158 面，闭合 |
| `output/jobs/iteration-u2net-triposr-256/` | TripoSR（更早对照） | 21,263 顶点 / 42,514 面 |

每个任务目录里都有：`generation-report.json`（参数/耗时/显存/网格校验）、
`*-preview.png`（本机正交渲染四视图）、`*-view1..4.png`、以及对应输入。

### 2026-11-05 修复的两个问题

- **左腿只剩一半**：根因有两个。① `mask_v2_pipeline.py` 的腿/鞋补回用了“最大连通块”，
  而两条腿在解析图里是分开的，导致稳定丢掉一条腿；② 多视图的 `left/right` 误用了同一侧
  （52a9 与 fc55 都是右侧），左侧缺少真实约束。现已改为保留所有腿部件并用 9e01（左）、
  52a9（右）作为左右视图，v2 模型双腿完整。
- **左右视图方向确认**：人脸朝向判断——52a9/fc55 鼻子朝画面右（右侧视图），9e01 鼻子朝画面左
  （左侧视图），9ca2/f33c 正面，0113/458d/6970 背面。

## 常用命令

```powershell
cd 'D:\Desktop\01_项目与产品\Rove如晤_跨时空交互\rove-site\model-generation'

# 1) 重新生成 v2 抠图（masks 阶段有缓存，可只跑 fuse + qc）
.\envs\triposr\Scripts\python.exe -u mask_v2_pipeline.py fuse
.\envs\triposr\Scripts\python.exe -u mask_v2_pipeline.py qc

# 2) 四视图 Hunyuan3D-2mv 形体生成（详见 HUNYUAN3D-SETUP.md）
$env:HY3DGEN_MODELS = (Resolve-Path '.\weights').Path
.\envs\hunyuan3d\Scripts\python.exe -u multiview_worker.py --preset full --octree 256 --steps 5 --profile 4

# 3) 给任意 GLB 生成四视图预览
.\envs\triposr\Scripts\python.exe render_glb.py <mesh.glb> <输出目录> --label "说明"

# 4) 纯几何多视图雕刻对照
.\envs\triposr\Scripts\python.exe visual_hull.py --front ... --back ... --left ... --right ... --out-dir ...
```

## 已知缺陷（重要）

- **手持道具未去除**：照片里人物拿着深灰布料袋和格纹布，`segformer-b2-clothes` 把它们
  判成“上衣/裙子/背景”，解析先验无法剔除，因此会进入蒙版并把布料团块带进 3D 模型（模型
  双手附近的团块就是这个原因）。已试过矩形 `prop_rois`（会切掉裙子留下缺口）和
  肤色锚定的自动删除（侧视图仍会切到裙子），均已回退。可靠解法：① 让真人空手重拍一组；
  ② 在页面上加蒙版笔刷，由人手动擦除；③ 接 SAM/手部-道具分割模型。
- **没有纹理**：当前只有形体；Hunyuan3D-Paint/纹理阶段未接。
- **没有绑定**：GLB 无骨架、手指骨骼、面部形态键，不能直接驱动作捕。
- TripoSR 仍只消费一张主视图，它的侧/背面是猜测，不代表真实拍摄信息。
