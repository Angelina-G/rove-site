# Rove 默认角色模型

## 当前默认角色：`rove-avatar-cc0.glb`

- 文件：`rove-avatar-cc0.glb`（约 36.8 MB）
- 来源资产：TalkingHead 示例角色 `mpfb.glb`，[met4citizen/TalkingHead](https://github.com/met4citizen/TalkingHead)
- 制作方式：Blender + [MPFB](https://static.makehumancommunity.org/mpfb.html)（MakeHuman 生态），作者在 TalkingHead 的资产许可说明中标注为 **CC0**。
- 许可：CC0 1.0（公共领域贡献）。可复制、修改、商用；本页保留来源与致谢。
- 结构（网页读取到的 GLB 实际数据）：
  - 67 个骨架关节，含 Mixamo 风格命名（`Hips`、`Spine`、`LeftArm`、`LeftHandIndex1` 等）
  - 30 个手指骨（双手各 15 个指节）
  - 66 个唯一面部 morph target，覆盖 ARKit 命名（`eyeBlinkLeft/Right`、`jawOpen`、`mouthSmileLeft/Right`、`browInnerUp`、`eyeWideLeft/Right` 等）及 14 个 viseme 口型
  - 8 个网格：身体、休闲服、眼睛、眉毛、睫毛、马尾、牙齿、舌头
  - 7 张内嵌贴图；无内置动画片段
- 网页用途：默认实时角色。MediaPipe Pose 驱动身体骨骼，Hand 的 21 点驱动手指骨，Face 的 blendshape 系数驱动面部 morph target；渲染循环每帧应用表情，因此“表情自检”按钮无需摄像头即可验证表情变形。
- 限制：单目摄像头姿态估计在遮挡、侧身、快速手部动作时仍会失真；模型手指为 3 节指骨（MediaPipe 输出 4 点，映射到 3 节）；viseme 可用于后续口型，当前网页未接入语音驱动。

## 已移除的候选角色（授权不适合本项目）

| 候选 | 来源 | 许可 | 结论 |
| --- | --- | --- | --- |
| `brunette.glb` | Ready Player Me，经 TalkingHead 分发 | CC BY-NC 4.0 | 仅限非商业，未采用 |
| `avatarsdk.glb` | AvatarSDK | 仅限非商业使用 | 未采用 |
| `vroid.glb` | VRoid Studio | 仅限非商业使用 | 未采用 |
| `realistic-woman-innerscene.glb` | Innerscene | CC0，但无面部 morph target | 已从项目移除，避免与默认角色混淆 |

## 外部上传模型

上传的 GLB 会在本机加载并尝试复用同一套身体/手指/表情映射；骨骼或 morph target 命名不兼容时，页面会在结构检查中报告可映射的骨段与表情通道数量，不会静默假装已驱动。
