# Rove 角色生成工作台：本地服务

Node.js 20+，原生 `http`，无第三方依赖。页面文件是 `../rove-model-studio.html`，
服务只监听本机回环地址（默认 `127.0.0.1:8790`）。

## 启动

```powershell
cd 'D:\Desktop\01_项目与产品\Rove如晤_跨时空交互\rove-site\model-api'
npm start
```

打开 <http://localhost:8790/>。端口由 `.env` 的 `PORT` 决定；8787/8788 曾被其他服务占用，
当前默认 8790。原来的实时动捕页仍是 `../rove-studio.html`，本服务不修改它。

## 页面能力

- **任务与阶段**：左侧列出 `model-generation/output/jobs/*` 的真实任务；右侧按阶段展示
  拍摄视角、抠图、模型输入、几何生成、验收，每个产物都能点开大图或下载。
  所有数字来自磁盘上的报告文件，缺失就显示“缺失”，不做占位数据。
- **抠图质检**：读取 `model-generation/test-images/processed-*`，默认优先 `processed-v2`；
  显示质检总览图、每张原图/透明抠图对照、前景占比、边缘泄漏比、角度建议。
- **新建生成**：上传一张照片 → `POST /api/local-models` → 本机 TripoSR（单视图基线）。
  原图写入临时目录，任务结束删除；中间产物保留在任务目录。
- **能力与边界**：明确区分已实现（分割、TripoSR 单视图、Hunyuan3D-2mv 四视图、visual hull）
  与未实现（纹理、骨架、手指骨骼、形态键、自动绑定）。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/config` | 服务配置、端口、后端、产物目录 |
| GET | `/api/jobs` | 任务列表 + 模型阵容 + 质检版本概览 |
| GET | `/api/jobs/:id` | 单个任务的分阶段详情、产物列表与网格分析 |
| GET | `/api/jobs/:id/files/<rel>` | 任务产物（支持 `?download=1`，防目录穿越） |
| GET | `/api/qc` | 质检版本概览；`?version=processed-v2` 取单版本逐张条目 |
| GET | `/api/qc/files/:version/<rel>` | 质检产物文件 |
| GET | `/api/qc/source/<name>` | `test-images` 根目录原图 |
| POST | `/api/local-models` | 提交 `{ image: dataURL }`，异步跑本机 TripoSR，返回任务号 |
| POST | `/api/models` | Meshy 云端任务（需 `MESHY_API_KEY`，未配置返回 503） |

## 生成后端

- **本机 TripoSR**（默认，`MODEL_PROVIDER=local-triposr`）：环境、源码、权重都在
  `model-generation/` 下；`worker.py` 先做 rembg 人物分割与中灰底构图再调 TripoSR。
- **Hunyuan3D-2mv 四视图**：独立脚本 `model-generation/multiview_worker.py` + 环境
  `model-generation/envs/hunyuan3d`，命令与实测见 `model-generation/HUNYUAN3D-SETUP.md`。
  该任务写入 `output/jobs/hunyuan3d-2mv-turntable/`，本页面会自动把它列出来。
- **Meshy 云端**：可选，需要付费计划的 API Key，密钥只由本机后端读取。

## 产物目录

```
model-generation/output/jobs/<任务号>/
  subject-cutout.png / model-input.png      # TripoSR 中间产物
  0/mesh.glb                                # TripoSR 网格
  inputs/{front,left,right,back}.png        # Hunyuan 四视图输入
  hunyuan3d-2mv.glb                         # Hunyuan 网格
  generation-report.json                    # 参数/耗时/显存/校验
  *-preview.png, *-view1..4.png             # 本机正交渲染预览
```

清理任务目录会同时删掉该任务的所有产物；页面不提供跨任务批量删除。

## 已知限制

- 任务状态以磁盘产物为准；服务重启不影响已生成文件。
- 抠图目前仍会把人物手持的布料/道具判进人体（人体解析将其识别为上衣），
  这会带入几何；需要在 `mask_v2_pipeline.py` 的 `IMAGE_CONFIG[*].prop_rois` 补人工 ROI。
- 生成模型没有骨架、手指骨骼、面部形态键，不能直接驱动作捕。
