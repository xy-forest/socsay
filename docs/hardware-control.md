# 硬件控制层设计 · hardware-control

> 目标：把 Insta360 Link 2 抽象成一个"我让你看哪里你就看哪里"的可编程头部。

---

## 1. 控制原语

只暴露 4 个原语，行为层不需要知道 UVC 细节：

```python
class PTZ:
    def get(self) -> tuple[int, int]: ...         # 当前 (pan, tilt)
    def move_abs(self, pan: int, tilt: int): ...  # 绝对移动，自动夹紧
    def move_rel(self, dpan: int, dtilt: int): ...# 相对移动
    def center(self): ...                         # 回中 (0, 0)
```

实现方式：直接 `subprocess.run(["./bin/uvc_ptz_set", str(pan), str(tilt)])`。
- 不要在 Phase 1 就重写 C 代码为 Python+libusb，**复用已验证的二进制**。
- 调用频率 ≤ 5 Hz（每次 control transfer 几十毫秒，过高会塞 USB）。

---

## 2. 坐标与边界

| 名称 | 值 | 说明 |
|---|---|---|
| `PAN_MIN/MAX` | `-300_000 / +300_000` | 比 demo 的 ±360000 更保守，留安全余量 |
| `TILT_MIN/MAX` | `-200_000 / +200_000` | 同上 |
| `STEP` | `3600` | 硬件最小步进，所有目标值会 round 到 step 的倍数 |
| `MAX_RATE_HZ` | `5` | 单位时间最多发 5 次 set |

**坐标语义约定**（务必统一，后面行为层全靠这个）：
- pan：**正值 = 相机看向画面右侧**（即用户视角向左）
- tilt：**正值 = 相机抬头**

如果实测和约定不符，在 `ptz.py` 里加一次反号，**不要**让上层去关心。

---

## 3. "唤醒"问题

> 关键事实：相机进入隐私待机时，UVC PTZ 命令会"成功但物理不动"。

策略：
1. **Capture 进程是唯一的"唤醒守门员"**——它从启动到退出始终持有 AVFoundation 输入。
2. 启动顺序强制：先 Capture，3 秒后才允许 Behavior 发第一条 PTZ 指令。
3. 健康检查：每 10 秒读一次 `uvc_ptz_get`，如果连续 3 次返回相同值且我们刚发过移动指令 → 判定相机睡了 → 重启 Capture。

---

## 4. Capture 子模块

### 4.1 取流

用 `ffmpeg` 子进程把帧吐到 stdout 的 MJPEG 流，Python 再切帧。理由：
- 不引入 OpenCV 的 AVFoundation 依赖（macOS 上 OpenCV 抓 AVF 容易踩权限坑）
- MJPEG 帧自带 SOI/EOI 易切分
- 输出码流可直接转发给调试 viewer

启动命令（已验证可用）：
```bash
ffmpeg -loglevel error \
  -f avfoundation \
  -pixel_format uyvy422 \
  -framerate 15 \
  -video_size 1280x720 \
  -i "Insta360 Link 2:none" \
  -an -f mjpeg -q:v 5 -
```

### 4.2 分发

帧到达后，写入一个**单槽位最新帧缓冲**（不是队列），消费者按需取最新一张：

```python
class LatestFrame:
    def put(self, jpeg: bytes): ...     # 覆盖
    def get(self, timeout=0.5) -> bytes:... # 阻塞拿最新
```

理由：感知层永远只关心"最新"，落后的帧扔掉比排队更对。

### 4.3 设备发现

**禁止**写 `0:none` / `1:none`。启动时先：

```bash
ffmpeg -f avfoundation -list_devices true -i ""
```

正则匹配 `\[(\d+)\] Insta360 Link 2`，拿到序号再拼输入 URI。匹配不到则报错退出，明确提示用户检查 USB 连接。

---

## 5. PTZ 控制策略（细节）

### 5.1 平滑跟随（Track 模式专用）

直接 `move_abs(target)` 会让云台瞬移、画面剧烈抖。改为：

```
每 200ms 算一次 target
err = target - current
step = clamp(err * 0.35, -SLEW_MAX, +SLEW_MAX)   # SLEW_MAX = 30000
move_abs(current + step)
```

参数 `0.35` 和 `30000` 都先写在配置里，调试时再改。

### 5.2 死区

人脸落在画面中央 ±8% 范围内时，**不发任何指令**。
否则相机会因为人脸轻微抖动而一直微调，看起来很神经质。

### 5.3 序列化

`PTZ` 类内部一把 `threading.Lock`。Behavior 层任何线程调用 `move_*` 都会被串行化。
原因：UVC control transfer 不是线程安全；并发会出现 `rc=-1` 偶发失败。

---

## 6. 故障与回退

| 现象 | 检测 | 处置 |
|---|---|---|
| `uvc_ptz_set` 返回非 0 | exit code | 重试 1 次；仍失败则 log + 跳过本次 |
| 连续 5 次 set 都失败 | 计数器 | 暂停跟踪 30 秒，等待 USB 自愈 |
| 相机被其他 App 抢走 | ffmpeg 进程退出 | 重启 ffmpeg；超过 3 次/分钟则报错给用户 |
| 软边界被达到 | clamp 触发 | 行为层应该收到通知（见 [behavior.md](behavior.md) 的 Roam）|

---

## 7. 不做的事

- ❌ Phase 1 不做 zoom 和 roll，虽然 probe 显示设备支持。
- ❌ 不做相对移动 API（`move_rel` 内部由 `move_abs` 实现，不暴露给行为层）。
- ❌ 不做"轨迹规划"（贝塞尔、缓动）；线性逼近 + 死区已经够用。
