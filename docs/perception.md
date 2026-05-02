# 感知层设计 · perception

> 目标：每隔 ~200ms 输出一个 `FaceTarget(cx, cy, conf, ts)`，让行为层据此驱动云台。

---

## 1. 核心矛盾

我们要回答两个问题，但答案速率完全不同：

| 问题 | 需要的延迟 | 适合的工具 |
|---|---|---|
| "脸现在在画面哪？" | < 200ms（不然跟不动） | 本地检测器 |
| "这真的是要被陪伴的那个人吗？" | 几秒可接受 | MiMo 多模态 |

→ **分层感知**：本地高频跟踪 + 云端低频校准。

---

## 2. 数据契约

```python
@dataclass
class FaceTarget:
    cx: float          # 0..1，画面坐标，0 = 左，1 = 右
    cy: float          # 0..1，0 = 上，1 = 下
    bbox: tuple[float, float, float, float]  # x, y, w, h, 归一化
    conf: float        # 0..1
    ts: float          # time.monotonic()
    source: str        # "local" | "mimo"
```

行为层只看 `cx, cy, conf, ts`。`source` 只用于日志和调试。

---

## 3. 本地检测器（高频，~5 Hz）

### 3.1 选型决策

候选：
1. **macOS Vision `VNDetectFaceRectanglesRequest`**（PyObjC 调用）
2. OpenCV Haar / DNN
3. mediapipe

**Phase 1 选 1（macOS Vision）**，理由：
- 系统自带，无额外模型下载
- 在 M 系列芯片上几乎零 CPU
- 准确度对"画面中一张近距离脸"完全够用
- 与 ffmpeg 抓流配合，全链路无外部模型依赖

如果 PyObjC 集成成本超出预期（M2 实测后判断），回退到 OpenCV DNN（`opencv-zoo` 里的 yunet 模型，约 2MB）。

### 3.2 输入输出

- 输入：来自 Capture 的最新 JPEG bytes
- 解码：`PIL.Image.open(BytesIO(jpeg))` → `np.ndarray`
- 输出：取**置信度最高的一张脸**（Phase 1 不处理多人）

### 3.3 平滑

原始检测每帧都会抖几像素，会让云台无意义地动。两层平滑：

1. **指数滑动平均**（α=0.4）：`cx_smooth = α*cx_new + (1-α)*cx_prev`
2. **死区由行为层做**（见 [hardware-control.md](hardware-control.md) §5.2），感知层只输出原始平滑值

### 3.4 丢失处理

连续 N 帧（N=5）没检测到脸 → 输出 `FaceTarget(conf=0)`。
行为层据此切换到 Roam 模式（缓慢左右扫视寻找）。

---

## 4. 云端校准器（低频，~0.2 Hz）

### 4.1 何时调用 MiMo

不要每帧都调，会爆 quota 也会拖慢系统。触发条件（任一）：

- 启动后第一次成功检测到脸（"这是不是真人？"）
- 连续 30 秒没有变化（确认还在 / 在干嘛）
- 本地检测器报告 conf 突变（可能换人 / 误检）
- 行为层主动请求（例如 Phase 2 想知道"用户表情是迷茫还是顿悟"）

### 4.2 调用方式

复用已验证的 endpoint：
```
POST https://api.xiaomimimo.com/v1/chat/completions
Header: api-key: $MIMO_API_KEY
Body:  model=mimo-v2.5, messages=[image_url(base64), text(prompt)]
```

Phase 1 的 prompt 模板（保持极简、要求结构化输出）：

```
你是一个视觉助手。请只用 JSON 回答，不要多余文字。
判断画面中是否有真人人脸，输出：
{
  "has_person": true|false,
  "is_real_face": true|false,
  "n_faces": 整数,
  "primary_face_box": [x, y, w, h]   // 归一化 0..1，找不到时为 null
}
```

解析失败（非 JSON / 字段缺失）→ 直接丢弃这次结果，不影响本地链路。

### 4.3 调用频控

- 单实例最多 1 个 in-flight 请求（用 `asyncio.Semaphore(1)`）
- 最小间隔 5 秒
- 失败重试上限 1 次，再失败就静默降级到"只信本地"

---

## 5. 融合策略

简单优先级，**绝不**让云端结果阻塞跟踪：

```
if local.conf > 0.5:
    target = local         # 主信号
elif mimo_recent and mimo_recent.has_person:
    target = mimo_recent   # 兜底
else:
    target = LOST
```

云端来的 `primary_face_box` 也送给本地检测器作为下次 ROI 提示（可选优化，Phase 1 可不做）。

---

## 6. 性能预算

| 阶段 | 预算 | 备注 |
|---|---|---|
| JPEG 解码 | < 10 ms | PIL 即可 |
| Vision 人脸检测 | < 30 ms | M 系列芯片 |
| 平滑 + 输出 | < 1 ms | |
| **本地总延迟** | **< 50 ms** | 留余量给 GIL |
| MiMo 一次往返 | 1–4 s | 不在关键路径 |

总目标：本地链路稳定 5–10 Hz 输出 `FaceTarget`。

---

## 7. 调试可见性

- 在每次输出 `FaceTarget` 时，把脸框画到 JPEG 上，存到 `/tmp/socsay_debug.jpg`（覆盖写）。
- 提供 `--save-mjpeg PATH` 选项，把带框的帧吐成 MJPEG 文件，方便事后回放。
- Phase 1 不做实时 Web 预览，避免与摄像头互抢资源。

---

## 8. 隐私边界（一开始就要想）

- **任何送往 MiMo 的图片**都要先在本地降采样到 ≤ 640px 长边，并去掉 EXIF。
- 提供配置开关 `cloud_perception: off`，关闭后系统只用本地检测，完全不出网。
- 调试帧默认存 `/tmp` 而非项目目录，避免误提交。
- README 里必须写明"图像会发送到小米 MiMo 服务"——这是 Phase 1 文档的硬性要求。
