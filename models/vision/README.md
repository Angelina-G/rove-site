# 动捕模型离线文件

追踪页面会优先检查当前目录中的模型文件，再访问在线地址。若 `storage.googleapis.com` 无法访问，可在网络可用的设备上下载后放入此目录：

| 文件名 | MediaPipe 官方模型地址 |
| --- | --- |
| `pose_landmarker_lite.task` | <https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task> |
| `hand_landmarker.task` | <https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task> |
| `face_landmarker.task` | <https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task> |

手部和面部模型也会尝试 [GitHub 上的备用镜像](https://github.com/sanderdesnaijer/mediapipe-model-mirrors/releases/tag/v1)。身体模型使用 Google 官方 Lite 文件。

放好文件后刷新工作台，再开启摄像头。页面会依次尝试 jsDelivr、Fastly jsDelivr、npmmirror 和 unpkg 加载 JavaScript/WASM 运行库；如果这些地址也都不可访问，仍需先把运行库与 WASM 文件放进站点后才能完全离线使用。
