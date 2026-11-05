# Rove 蒙版修正学习闭环

## 已实现

1. 前端每次确认保存 `processed-v3` 后，服务端在 `output/feedback/<uuid>.json` 留下审计记录。
2. `build_training_dataset.py` 将确认记录整理为可追溯 manifest，并明确标记缺少源蒙版或修正版蒙版的样本。
3. `learning/train_mask_adapter.py` 读取 manifest 做训练就绪检查，默认只生成训练计划，不虚假修改模型权重。

## 运行

```powershell
python model-generation/build_training_dataset.py `
  --feedback-dir model-generation/feedback `
  --output model-generation/output/learning/dataset-manifest.json

python model-generation/learning/train_mask_adapter.py `
  --manifest model-generation/output/learning/dataset-manifest.json `
  --output model-generation/output/learning/training-plan.json `
  --min-samples 500
```

达到样本量后，下一步才接入具体分割骨干（例如 BiRefNet / SAM 适配器）和固定的训练/验证划分。当前脚本刻意不声称已经更新模型参数。
