# 苏格拉底如是说 · 设计总览（Phase 1）

> Phase 1 范围限定为：**让 Insta360 Link 2 成为一个有"目光"的伙伴**——能找到人脸、能让人脸保持在画面中央、能用点头/摇头/缓慢旋转表达态度。
> 后续 Phase 2 才接入苏格拉底式诘问对话与思考时长可视化。

---

## 1. 项目定位

**苏格拉底如是说（socsay）** 是面向 AI 协作时代的意图对齐工具：在用户开始让 AI "执行" 之前，先用追问帮 ta 想清楚。

物理层面，我们让一台 Insta360 Link 2 真正"看着"用户：
- 它会**找到你的脸**，把镜头对准你
- 它会用**点头/摇头/缓慢旋转**，作为一种轻量的非语言反馈
- 它不是装饰品，而是"陪伴你思考"的物理化身

Phase 1 的成功标准：**人在镜头前移动，相机能跟住；当系统想表达"嗯"或"我不太懂"时，相机能用动作回应。**

---

## 2. 已具备的基础

来自姊妹项目 [`insta360-link2-ptz-demo`](../../insta360-link2-ptz-demo/README.md)，已经验证可用：

- **PTZ 控制**：通过 UVC `pantilt`（selector `0x0d`，8 字节 LE int32 pan + tilt）直接操控云台。pan/tilt 当前安全软边界 `±360000 / ±270000`，硬件 step ≈ `3600`。
- **唤醒**：相机进入隐私待机时云台命令"成功但不动"，必须先有视频流读取才会真正动。macOS 用 `ffmpeg -f avfoundation -pixel_format uyvy422 ...` 抓流即可。
- **AVFoundation 设备名**：`Insta360 Link 2:none`，不要写死序号。
- **MiMo 多模态**：`https://api.xiaomimimo.com/v1/chat/completions`，`api-key` 头，模型 `mimo-v2.5` 已实测能对截图做识别（含 image_tokens 计费）。

> 注意一个重要的限速预期：MiMo 是云端推理，单次图像理解响应在秒级。**不能把它当 30fps 的人脸检测器用。** 见 [perception.md](perception.md) 的"分层感知"设计。

---

## 3. Phase 1 系统结构

```
┌────────────────────────────────────────────────────────────────┐
│                       socsayd (单进程)                          │
│                                                                │
│  ┌──────────┐   frame  ┌──────────────┐   target  ┌─────────┐  │
│  │ Capture  │─────────▶│  Perception  │──────────▶│ Behavior│  │
│  │ (AVF →   │  JPEG/   │  (face loc)  │  (cx,cy,  │  (FSM)  │  │
│  │  MJPEG)  │  RGB     │              │   conf)   │         │  │
│  └──────────┘          └──────────────┘           └────┬────┘  │
│       ▲                       │                        │       │
│       │                       │ low-rate (~1 Hz)       │       │
│       │                       ▼                        ▼       │
│       │              ┌─────────────────┐      ┌────────────┐   │
│       │              │ MiMo image API  │      │  PTZ Ctrl  │   │
│       │              │ (远程, 备用/校准)│      │ (uvc_set)  │   │
│       │              └─────────────────┘      └────────────┘   │
│       │                                              │         │
│       └────── (相机被 PTZ 命令物理移动) ◀────────────┘         │
└────────────────────────────────────────────────────────────────┘
```

三层职责（每层一个独立模块，可单独换实现）：

| 层 | 职责 | Phase 1 实现 | 替换空间 |
|---|---|---|---|
| **Capture** | 拿到稳定帧 + 不让相机睡着 | FFmpeg/AVFoundation 抓 1280×720 / 15fps | OpenCV、libuvc |
| **Perception** | 输出"目标点 (cx, cy, conf)" | 本地快检测 + MiMo 慢校准（双速率） | 纯本地 / 纯云 |
| **Behavior** | 决定相机怎么动 | FSM：Idle / Track / Nod / Shake / Roam | 接 LLM 决策 |

详细见各分文档：
- [hardware-control.md](hardware-control.md)
- [perception.md](perception.md)
- [behavior.md](behavior.md)

---

## 4. 工程边界（Phase 1 不做什么）

为了避免一开始就过度设计，Phase 1 **明确不做**：

- ❌ 苏格拉底诘问对话（Phase 2）
- ❌ 思考时间可视化、Duolingo 式激励（Phase 2）
- ❌ 微表情识别（Phase 3，需要专门模型）
- ❌ 多人场景下的"主讲者识别"（先做"画面里最大那张脸"）
- ❌ 4K / HDR / 自动对焦干预
- ❌ Web UI（直接命令行驱动，必要时输出调试 MJPEG 流）
- ❌ Windows / Linux 适配（先在 macOS 跑通，再讨论）

---

## 5. 里程碑

| M | 内容 | 验收 |
|---|---|---|
| **M0** | 目录骨架 + 设计文档（本次） | 4 篇 .md 齐 |
| **M1** | Capture + 持续唤醒 | 命令行启动后，相机不进隐私待机；可 dump 单帧 |
| **M2** | Perception 出"人脸中心点" | 终端实时打印 `(cx, cy, conf, dt)`，1 Hz 以上 |
| **M3** | Track 闭环（最小可用产品） | 人在镜头前缓慢移动，脸能保持在画面中央 ±10% |
| **M4** | Nod / Shake / Roam 三种姿态 | 命令行可触发；动作不撞软边界、不卡云台 |
| **M5** | 简单决策接口 | 暴露 HTTP `/intent` 让 Phase 2 的对话层调用 |

---

## 6. 仓库结构（建议）

```
socsay/
├── docs/                    ← 当前 phase 的设计
│   ├── README.md            ← 本文件
│   ├── hardware-control.md
│   ├── perception.md
│   └── behavior.md
├── socsay/                  ← Python 源码（M1 起新建）
│   ├── capture.py
│   ├── perception.py
│   ├── behavior.py
│   ├── ptz.py               ← 包装 uvc_ptz_set/get
│   └── main.py
├── bin/                     ← 复用 insta360-link2-ptz-demo 编出的二进制
│   ├── uvc_ptz_get
│   └── uvc_ptz_set
└── .env                     ← MIMO_API_KEY 等
```

---

## 7. 风险与对策（务实清单）

| 风险 | 对策 |
|---|---|
| MiMo 云端延迟 ≥ 1s，不足以做闭环跟踪 | 本地 OpenCV/Vision 做高频检测，MiMo 仅做低频"语义校准"（确认是不是同一个人 / 是不是真人脸） |
| 相机进入隐私待机后 PTZ 失灵 | Capture 进程在整个生命周期内**始终持有**视频流 |
| AI Tracking / Auto Framing 抢控制权 | 启动时打印警告，要求用户在 Insta360 Link Controller 里关掉自动跟踪 |
| PTZ 撞软/硬边界后状态不同步 | 控制层夹紧到 `pan ±300000 / tilt ±200000`（比 demo 还保守），并在每次大跳转后回读 `uvc_ptz_get` 校准 |
| 多个进程同时打开摄像头掉线 | 单进程持流，对外只暴露 HTTP/IPC，不让别人直接占设备 |
| 误把椅背、海报当人脸 | Perception 层用置信度阈值 + MiMo 兜底校验（"这看上去是真人吗"）|

---

## 8. 与 Phase 2 的接口预留

Phase 2 的诘问 / 可视化模块会以**消费者**身份接入：

- **订阅** Behavior 输出的"用户状态信号"（专注 / 走神 / 不在画面 / 长时间静止）
- **下发** 高层意图给 Behavior：`{"gesture":"nod","reason":"我听懂了"}` / `{"gesture":"shake","reason":"再想想"}`
- 接口形态预留 HTTP + JSON，避免今天就锁定 WebSocket / gRPC

这样 Phase 1 可以独立完成、独立验证，Phase 2 不用改 Phase 1 的代码。
