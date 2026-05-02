# 行为层设计 · behavior

> 目标：把"目标点"和"高层意图"翻译成一连串 PTZ 动作，让相机看起来像一个**有态度的伙伴**，而不是一个伺服系统。

---

## 1. 状态机（FSM）

Phase 1 五个状态，转移由感知信号 + 外部命令共同驱动：

```
                ┌─────────┐
        boot ──▶│  Idle   │◀──────────── done(any gesture)
                └────┬────┘
       face_found    │     no_face_for_5s
                ▼    │             ▼
         ┌─────────┐ │       ┌─────────┐
         │  Track  │◀┘       │  Roam   │
         └────┬────┘         └────┬────┘
              │  intent.gesture    │ face_found
              ▼                    ▼
        ┌──────────┐         (back to Track)
        │ Nod /    │
        │ Shake    │
        └────┬─────┘
             │ 动作完成
             ▼
          回到上一状态
```

| 状态 | 进入条件 | 行为 | 离开条件 |
|---|---|---|---|
| **Idle** | 启动时 / 上一动作收尾 | 维持当前位置，不发指令 | 检测到脸 → Track；外部 intent → 对应手势 |
| **Track** | 有人脸 (conf > 0.5) | 平滑跟随脸到画面中央 | 连续 5s 丢脸 → Roam；外部手势 intent → Nod/Shake |
| **Roam** | 长时间无脸 | 缓慢左右扫视，1°/s 量级 | 检测到脸 → 立即 Track |
| **Nod** | intent="nod" | 上下点头 2 次后回原位 | 动作完成 |
| **Shake** | intent="shake" | 左右摇头 2 次后回原位 | 动作完成 |

**关键约束**：手势期间感知层依然在工作，但 PTZ 控制权被手势"独占"——这避免点头到一半被跟踪打断。

---

## 2. 跟踪算法（Track）

### 2.1 误差到位移

人脸目标是 `(cx, cy)`（画面坐标 0..1），中心是 `(0.5, 0.5)`。
误差 `ex = cx - 0.5`, `ey = cy - 0.5`。

经验映射（首版常量，调好以后再写进配置）：

```python
PAN_PER_UNIT  = 400_000   # 满画面宽 → 40 万 pan 单位
TILT_PER_UNIT = 250_000   # 满画面高 → 25 万 tilt 单位
GAIN          = 0.35      # 单步只走 35% 误差，避免过冲

dpan  = -ex * PAN_PER_UNIT  * GAIN  # 注意符号：脸偏右 → 镜头要往右转
dtilt = -ey * TILT_PER_UNIT * GAIN  # 脸偏上 → 镜头要抬头
```

> 符号在第一次实测时极容易反，**专门留一组开关**：`INVERT_PAN`, `INVERT_TILT`。M3 调试时打开/关闭即可。

### 2.2 死区与节拍

- 死区：`|ex| < 0.08` 且 `|ey| < 0.08` → 不动
- 控制节拍：固定 5 Hz；即使感知输出更快，也合并到下一个 tick
- 单步上限：`|dpan| ≤ 30000`, `|dtilt| ≤ 25000`，防止瞬移

### 2.3 边界感知

如果 clamp 被触发（即想动但已经到 PAN_MAX 之类），把这个事件 emit 给行为层主循环。
含义：**用户已经走到镜头视野边缘了**——这本身是一个信号，Phase 2 可以用来提示"嘿，回来"。

---

## 3. 手势：Nod / Shake / Roam

每个手势 = 一组带时间戳的相对位移序列，**不依赖**当前感知输入：

```python
# 单位：(dpan, dtilt, sleep_ms)
NOD = [
    (0, -25000, 250),
    (0, +50000, 250),
    (0, -25000, 250),
    (0, -25000, 250),
    (0, +50000, 250),
    (0, -25000, 250),
]
SHAKE = [
    (-30000, 0, 220),
    (+60000, 0, 220),
    (-60000, 0, 220),
    (+60000, 0, 220),
    (-30000, 0, 220),
]
```

执行流程：
1. 进入手势状态前，**记录当前位置** `(p0, t0)`
2. 顺序执行序列
3. 序列结束后强制 `move_abs(p0, t0)` 回原位
4. emit `gesture_done(name)` 事件，FSM 回到上一个状态

**Roam** 不同，它是"持续行为"而非"一次性手势"：

```
方向 = ±1（每次走到边界就翻号）
每 500ms 发一次 (dir * 12000, 0)
直到 face_found
```

---

## 4. 对外接口（为 Phase 2 预留）

最小 HTTP，本地 only，方便 Phase 2 的对话层 / 可视化层接入：

```
GET  /state            → {"state":"Track","face":{...},"pos":[pan,tilt]}
POST /intent           → body: {"gesture":"nod"|"shake","reason":"..."}
GET  /events  (SSE)    → 流式输出 face_found / face_lost / gesture_done / boundary_hit
```

约定：
- `POST /intent` 是**建议而非命令**——FSM 自己决定什么时候执行（例如正在 Roam 找人时会延后）
- `reason` 字段会写入日志，方便事后回放"为什么相机点头了"
- 端口默认 `8788`（避开 demo 的 `8787`）

---

## 5. 时间与节拍统一

整个进程一个主循环，5 Hz tick：

```python
while running:
    tick_start = time.monotonic()

    target = perception.latest()          # 不阻塞
    intent = intent_queue.pop_or_none()
    fsm.step(target, intent, ptz)         # 唯一的状态转移点

    sleep_until(tick_start + 0.2)
```

好处：
- 所有控制频率上限是固定的 5 Hz
- 容易插入"录制 + 回放"用于 Phase 2 调试
- 单线程改动 PTZ，不需要锁（PTZ 类内部还是有锁兜底）

---

## 6. 验收脚本（M3 / M4 用）

写在 `socsay/tests/manual_*.md` 里，**人工执行**即可，不写自动化：

1. **Track 收敛**：人站在镜头前 1 米，左右各走 1 步停 3 秒。期望：脸始终落在画面中央 ±10%。
2. **Lost → Roam → Re-acquire**：人离开画面 10 秒。期望：相机开始缓慢扫视；人回来后 2 秒内重新跟住。
3. **Nod / Shake**：`curl -XPOST localhost:8788/intent -d '{"gesture":"nod"}'`。期望：相机点头 2 次后回到刚才的位置（不是 0,0）。
4. **边界报告**：人沿水平方向走出镜头视角。期望：日志出现 `boundary_hit pan_max`。

---

## 7. 不做的事（再次强调）

- ❌ 不做表情/情绪识别驱动的自动 Nod/Shake——Phase 1 的手势全部由**外部 intent 触发**或**Roam 切换触发**，避免我们对一个还没建出来的"懂人"模型做假设
- ❌ 不做 PID 调参——P 控制器 + 死区 + 步长上限对桌面跟随完全够
- ❌ 不在行为层做 LLM 调用——保持本层完全确定性，方便 debug
