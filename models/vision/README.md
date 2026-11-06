# 动捕运行文件

身体、手部和面部模型文件已放在本目录，网页会优先从本机读取：

| 文件名 | MediaPipe 官方模型地址 |
| --- | --- |
| `pose_landmarker_lite.task` | <https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task> |
| `hand_landmarker.task` | <https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task> |
| `face_landmarker.task` | <https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task> |

MediaPipe JavaScript/WASM 运行库和 Three.js 也已放在网页 `node_modules` 目录。完整复制或部署网页时，需要一并包含 `models/vision` 与 `node_modules`，才能在不访问外部 CDN 的情况下运行。

模型文件的备用来源：[MediaPipe 官方模型](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)和[手部/面部 GitHub 镜像](https://github.com/sanderdesnaijer/mediapipe-model-mirrors/releases/tag/v1)。
