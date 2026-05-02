"""微表情/情绪识别循环：本地检测 → 裁脸 → MiMo 多模态。

分工（与 perception.py/behavior.py 协作）：
  - perception.py   ：本地 haar 30Hz 找脸 → 喂 behavior 跟踪云台
  - expression.py   ：每 INTERVAL 秒（默认 5s）抓最新帧 + 最新 face bbox，
                      裁出脸（带 padding），base64 → MiMo 多模态，
                      解析微表情 JSON，可选 TTS 朗读 comment。
  - 不再用 MiMo 做人脸定位。

输出 JSON schema（assistant 必须严格只输出这一行）：
  {
    "emotion":      "neutral|happy|sad|angry|surprised|disgust|fear|confused|focused|tired|distracted|calm|relaxed|excited|anxious|nervous|stressed|frustrated|annoyed|irritated|bored|embarrassed|ashamed|guilty|skeptical|curious|interested|relieved|proud|confident|hopeful|disappointed|worried|overwhelmed|impatient|serious|playful|amused|grateful|lonely|melancholic|apathetic|tense|alert|startled|shocked|concerned|doubtful|suspicious|resistant|defensive|open_minded|trusting|engaged|disengaged|attentive|inattentive|listening|speaking|reading|idle|thinking|deep_thinking|shallow_thinking|reflective|contemplative|evaluating|analyzing|processing|reasoning|problem_solving|brainstorming|imagining|remembering|recalling|comparing|deciding|hesitating|uncertain|unsure|ambivalent|stuck|blocked|lost|blanking|insightful|realizing|understanding|misunderstanding|agreeing|disagreeing|accepting|rejecting|questioning|seeking_clarification|needs_prompting|ready_to_continue|ready_to_answer|not_ready|thinking_hard|cognitive_load_high|cognitive_load_low|overloaded|mentally_fatigued|physically_tired|sleepy|restless|zoned_out|daydreaming|avoidant|avoidance|resisting|withdrawing|leaning_in|leaning_back|nodding|shaking_head|smiling|frowning|eye_contact|looking_away|looking_down|looking_up|looking_left|looking_right|blinking|rapid_blinking|squinting|brow_furrowed|brow_raised|mouth_tight|lip_pressed|jaw_tense|micro_smile|forced_smile|genuine_smile|puzzled|perplexed|inspired|motivated|unmotivated|determined|hesitant_but_engaged|confused_but_curious|focused_but_tired|engaged_but_uncertain|ready_for_execution|needs_deeper_questioning|intent_clear|intent_unclear|alignment_high|alignment_low|flow_state|interrupted|recovering_focus|waiting|pausing|long_pause|short_pause|self_correcting|reconsidering|changing_mind|deflecting|explaining|justifying|summarizing|clarifying|exploring|committing|concluding",
    "intensity":    0.0..1.0,  # 紧张度：0=极放松，1=极紧张/强烈情绪
    "attention":    "high|medium|low",
    "gaze":         "at_camera|away|down|up|side",
    "comment":      "<=20 字中文，对当下表情的简短观察"
  }

CLI（独立跑，用于调通）：
    python -m socsay.expression --interval 5 --max-iters 3 -v
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .capture import Capture
from .perception import FaceTarget, Perception
from .voice import _load_dotenv  # 复用 .env 加载

log = logging.getLogger("socsay.expression")

DEFAULT_INTERVAL = 3.0
MIN_CONF = 0.12                 # 低于这个置信度跳过这一轮
FACE_PAD = 0.40                 # 脸框外扩比例（占 bbox 宽/高）
CROP_MAX_EDGE = 384             # 上传前裁脸最长边
DEBUG_PATH = Path("/tmp/socsay_expr.jpg")
FACE_DIR = Path("/tmp/socsay_faces")
FACE_DIR.mkdir(parents=True, exist_ok=True)
FACE_HISTORY_MAX = 30
MIMO_VLM_MODEL = "mimo-v2.5"
MIMO_VLM_TIMEOUT = 30
MIMO_VLM_MAX_TOKENS = 1500

# 一步到位：看脸 → 直接以温柔观察者口吻描述对方的微表情
SOCRATIC_VISION_PROMPT_old = (
    "你是一位安静、敏锐的微表情观察者，正面对面看着这张脸。\n"
    "请像在做现场旁白一样，用一句口语中文，描述你此刻在对方脸上看到的细节。\n"
    "\n"
    "要求：\n"
    " 1) speech 是一句中文 ≤35 字，温柔、第二人称（用“你”），像在轻声告诉对方你看到什么；\n"
    " 2) 只描述微表情和神态——眉头、眼角、嘴角、视线、呼吸节奏、面部张力等；\n"
    " 3) 不要提问、不要建议、不要分析意图、不要套话、不要 emoji、不要引号；\n"
    " 4) 每次尽量描述不同的细节，避免重复上一句；\n"
    " 5) 严格输出一行 JSON，不要 markdown、不要多余解释。\n"
    "\n"
    "字段：\n"
    "  speech    : 你要说出口的那句中文描述\n"
    "  observe   : ≤20 中文，更冷静客观的描述（给日志用）\n"
    "  emotion   : 从上面 emotion 枚举中选一个最切合的标签\n"
    "  attention : high|medium|low"
)

EMOTION_ENUM = (
    "neutral|happy|sad|angry|surprised|disgust|fear|confused|focused|tired|distracted|"
    "calm|relaxed|excited|anxious|nervous|stressed|frustrated|annoyed|irritated|bored|"
    "embarrassed|ashamed|guilty|skeptical|curious|interested|relieved|proud|confident|"
    "hopeful|disappointed|worried|overwhelmed|impatient|serious|playful|amused|grateful|"
    "lonely|melancholic|apathetic|tense|alert|startled|shocked|concerned|doubtful|"
    "suspicious|resistant|defensive|open_minded|trusting|engaged|disengaged|attentive|"
    "inattentive|listening|speaking|reading|idle|thinking|deep_thinking|shallow_thinking|"
    "reflective|contemplative|evaluating|analyzing|processing|reasoning|problem_solving|"
    "brainstorming|imagining|remembering|recalling|comparing|deciding|hesitating|"
    "uncertain|unsure|ambivalent|stuck|blocked|lost|blanking|insightful|realizing|"
    "understanding|misunderstanding|agreeing|disagreeing|accepting|rejecting|questioning|"
    "seeking_clarification|needs_prompting|ready_to_continue|ready_to_answer|not_ready|"
    "thinking_hard|cognitive_load_high|cognitive_load_low|overloaded|mentally_fatigued|"
    "physically_tired|sleepy|restless|zoned_out|daydreaming|avoidant|avoidance|"
    "resisting|withdrawing|leaning_in|leaning_back|nodding|shaking_head|smiling|frowning|"
    "eye_contact|looking_away|looking_down|looking_up|looking_left|looking_right|blinking|"
    "rapid_blinking|squinting|brow_furrowed|brow_raised|mouth_tight|lip_pressed|jaw_tense|"
    "micro_smile|forced_smile|genuine_smile|puzzled|perplexed|inspired|motivated|"
    "unmotivated|determined|hesitant_but_engaged|confused_but_curious|focused_but_tired|"
    "engaged_but_uncertain|ready_for_execution|needs_deeper_questioning|intent_clear|"
    "intent_unclear|alignment_high|alignment_low|flow_state|interrupted|recovering_focus|"
    "waiting|pausing|long_pause|short_pause|self_correcting|reconsidering|changing_mind|"
    "deflecting|explaining|justifying|summarizing|clarifying|exploring|committing|concluding"
)

SOCRATIC_VISION_PROMPT = (
    "你是苏格拉底，正面对面看着这张脸。\n"
    "请像在做现场旁白一样，用一句口语中文，描述你此刻在对方脸上看到的细节。\n"
    "\n"
    "要求：\n"
    " 1) speech 是一句中文 ≤35 字，在轻声告诉对方你看到什么；\n"
    " 2) 描述微表情和神态；\n"
    " 3) 不要提问、不要建议、不要分析意图、不要套话、不要 emoji、不要引号；\n"
    " 4) 每次尽量描述不同的细节，避免重复上一句；\n"
    " 5) 严格输出一行 JSON，不要 markdown、不要多余解释。\n"
    "\n"
    "字段：\n"
    "  speech    : 你要说出口的那句中文描述\n"
    "  observe   : ≤20 中文，更冷静客观的描述（给日志用）\n"
    "  emotion   : 从以下枚举中选最切合的一个标签\n"
    + "             " + EMOTION_ENUM + "\n"
    "  intensity : 0.0..1.0 的浮点数，量化当下脸部表情的「强度/紧张度」。\n"
    "             - 0.00–0.20 非常放松、平静；\n"
    "             - 0.20–0.45 轻微专注/轻度中性；\n"
    "             - 0.45–0.70 明显思考、轻度蹙眉、口微抑；\n"
    "             - 0.70–1.00 强烈紧张、困惑、压力、烦躁、足以被一眼看出的情绪。\n"
    "  attention : high|medium|low"
)


@dataclass
class ExpressionResult:
    ts: float
    emotion: str
    intensity: float
    attention: str
    gaze: str
    comment: str            # observe 字段
    speech: str = ""        # 苏格拉底要说的那句话
    face_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    face_path: str = ""     # /tmp/socsay_faces/face_<ts>.jpg
    raw: str = ""
    usage: dict = field(default_factory=dict)
    latency_ms: int = 0


# ---------- MiMo 客户端 ----------

class _MimoVLM:
    def __init__(self, model: str = MIMO_VLM_MODEL):
        self.model = model
        self.api_key = os.environ.get("MIMO_API_KEY", "")
        self.api_base = (os.environ.get("MIMO_API_BASE")
                         or "https://api.xiaomimimo.com").rstrip("/")
        if not self.api_key:
            raise RuntimeError("MIMO_API_KEY not set (.env or env)")

    def socratic_vision(self, jpeg_bytes: bytes) -> tuple[str, dict]:
        """一步到位：看脸 → 返回 JSON {speech,observe,emotion,attention}。"""
        b64 = base64.b64encode(jpeg_bytes).decode()
        url = f"{self.api_base}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SOCRATIC_VISION_PROMPT},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }
            ],
            "max_completion_tokens": MIMO_VLM_MAX_TOKENS,
            "temperature": 0.6,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "api-key": self.api_key},
        )
        with urllib.request.urlopen(req, timeout=MIMO_VLM_TIMEOUT) as resp:
            raw = resp.read()
        d = json.loads(raw)
        msg = d["choices"][0]["message"]
        content = msg.get("content") or ""
        return content, d.get("usage", {})


# ---------- 工具 ----------

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _crop_face(frame_bgr: np.ndarray,
               bbox_norm: tuple[float, float, float, float],
               pad: float = FACE_PAD) -> Optional[np.ndarray]:
    H, W = frame_bgr.shape[:2]
    x, y, w, h = bbox_norm
    if w <= 0 or h <= 0:
        return None
    cx = x + w / 2
    cy = y + h / 2
    side = max(w, h) * (1 + pad)
    half = side / 2
    x0 = int(max(0, (cx - half) * W))
    y0 = int(max(0, (cy - half) * H))
    x1 = int(min(W, (cx + half) * W))
    y1 = int(min(H, (cy + half) * H))
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None
    crop = frame_bgr[y0:y1, x0:x1]
    # 限制最长边
    fh, fw = crop.shape[:2]
    long_edge = max(fh, fw)
    if long_edge > CROP_MAX_EDGE:
        scale = CROP_MAX_EDGE / long_edge
        crop = cv2.resize(crop, (int(fw * scale), int(fh * scale)),
                          interpolation=cv2.INTER_AREA)
    return crop


def _encode_jpeg(img: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return bytes(buf)


# ---- 把 intensity 映射到 MiMo TTS 的情绪 ----
#   user-message 里写自然语言风格 + assistant 文本前缀音频标签 (..)
_TENSE_EMOTIONS = {
    "angry", "anxious", "nervous", "stressed", "frustrated", "annoyed",
    "irritated", "impatient", "tense", "alert", "shocked", "startled",
    "overwhelmed", "worried", "defensive", "resistant", "resisting",
    "jaw_tense", "brow_furrowed", "mouth_tight", "lip_pressed",
    "cognitive_load_high", "overloaded", "thinking_hard",
    "hesitating", "stuck", "blocked", "perplexed", "puzzled",
}


def style_for_intensity(intensity: float, emotion: str = ""
                         ) -> tuple[str, str]:
    """按 intensity 选 (user-style 自然语言, assistant 前缀标签)。"""
    emo = (emotion or "").lower()
    bump = 0.10 if emo in _TENSE_EMOTIONS else 0.0
    x = max(0.0, min(1.0, float(intensity) + bump))
    if x < 0.25:
        return ("用极轻、放松、温柔的耳语，气声、慢速，像哄人入睡，尾音下沉。",
                "(平静)")
    if x < 0.50:
        return ("语调平和温暖，像与朋友低声交谈，节奏从容，略带关切。",
                "(温柔)")
    if x < 0.72:
        return ("语气专注略带凝重，节奏稍慢，像在认真劝说，重音略加。",
                "(严肃)")
    return ("带一点不耐烦和挑衅，语速偏快，尾音上扬，像在催促对方："
            "够了，醒醒。",
            "(冷笑 不耐烦)")


# ---------- 主类 ----------

class Expression:
    """周期性裁脸 → MiMo 微表情分析。

    可选 voice：传入则把 comment 朗读出来（同 comment 60s 内不复读）。
    """

    def __init__(self, capture: Capture, perception: Perception,
                 voice=None, interval: float = DEFAULT_INTERVAL,
                 speak_threshold: float = 0.40,
                 model: str = MIMO_VLM_MODEL,
                 socratic: bool = True):
        self.capture = capture
        self.perception = perception
        self.voice = voice
        self.interval = interval
        self.speak_threshold = speak_threshold
        self.socratic = socratic
        self._client = _MimoVLM(model=model)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest: Optional[ExpressionResult] = None
        self._latest_lock = threading.Lock()
        self._history: list[ExpressionResult] = []
        self._history_max = 20
        self._fail_count = 0
        self._call_count = 0
        self._last_spoken = ""
        self._last_spoken_ts = 0.0
        # 滑窗：(ts, intensity) 用来给主控 / 回车拦截做近 60s 平均
        self._intensity_log: list[tuple[float, float]] = []
        self._paused = False  # 默认启动，可在网页上暂停省额度

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="expression", daemon=True
        )
        self._thread.start()
        log.info("expression loop started, interval=%.1fs model=%s",
                 self.interval, self._client.model)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def latest(self) -> Optional[ExpressionResult]:
        with self._latest_lock:
            return self._latest

    def snapshot(self) -> dict:
        last = self.latest()
        avg, n, mx = self.recent_intensity(60.0)
        return {
            "interval": self.interval,
            "calls": self._call_count,
            "fails": self._fail_count,
            "history_size": len(self._history),
            "latest": None if last is None else self._result_dict(last),
            "recent_avg": round(avg, 3),
            "recent_max": round(mx, 3),
            "recent_samples": n,
        }

    def history(self, limit: int = 30) -> list[dict]:
        with self._latest_lock:
            items = list(self._history[-limit:])
        return [self._result_dict(r) for r in reversed(items)]

    def recent_intensity(self, window_sec: float = 60.0
                          ) -> tuple[float, int, float]:
        """返回近 window_sec 内 (avg, n_samples, max)。无样本时 (0,0,0)。"""
        cutoff = time.time() - window_sec
        with self._latest_lock:
            samples = [v for (t, v) in self._intensity_log if t >= cutoff]
        if not samples:
            return 0.0, 0, 0.0
        return sum(samples) / len(samples), len(samples), max(samples)

    @staticmethod
    def _result_dict(r: "ExpressionResult") -> dict:
        return {
            "ts": r.ts,
            "emotion": r.emotion,
            "intensity": r.intensity,
            "attention": r.attention,
            "gaze": r.gaze,
            "comment": r.comment,
            "speech": r.speech,
            "face_path": r.face_path,
            "face_url": (
                f"/expression/face/{int(r.ts * 1000)}.jpg"
                if r.face_path else ""
            ),
            "latency_ms": r.latency_ms,
        }

    # ---------- internals ----------

    def pause(self):
        self._paused = True
        log.info("expression paused")

    def resume(self):
        self._paused = False
        log.info("expression resumed")

    @property
    def paused(self) -> bool:
        return self._paused

    def _loop(self):
        # 起始延一拍，让 perception 先出帧
        self._stop.wait(0.5)
        while not self._stop.is_set():
            if self._paused:
                self._stop.wait(0.5)
                continue
            t0 = time.monotonic()
            try:
                self._tick()
            except Exception:
                self._fail_count += 1
                log.exception("expression tick failed")
            elapsed = time.monotonic() - t0
            self._stop.wait(max(0.0, self.interval - elapsed))

    def _tick(self):
        # 1) 等到一张稳定的人脸：在最长 WAIT_BUDGET 内，要求 GAZE_RATIO 比例
        #    的采样都能看到脸，且位置漂移有限。容忍 cascade 偶发漏检（因此
        #    用"占比"而不是"严格连续"）。
        DWELL_SEC = 1.2           # 至少盯住的窗口长度
        DWELL_DRIFT = 0.22        # 窗口内位置漂移上限
        GAZE_RATIO = 0.55         # 窗口内至少这个比例的样本要看到脸
        WAIT_BUDGET = 8.0         # 最长等 8 秒
        SAMPLE = 0.12

        wait_deadline = time.monotonic() + WAIT_BUDGET
        # 滑动窗口：(t, face_or_None)
        window: list[tuple[float, "FaceTarget | None"]] = []
        face = None
        progress_log_ts = 0.0
        while time.monotonic() < wait_deadline and not self._stop.is_set():
            f = self.perception.latest()
            now = time.monotonic()
            window.append((now, f if (f and f.conf >= MIN_CONF) else None))
            # 丢掉窗口外样本
            cutoff = now - DWELL_SEC
            window = [(t, x) for (t, x) in window if t >= cutoff]

            seen = [x for (_, x) in window if x is not None]
            span = (window[-1][0] - window[0][0]) if window else 0.0
            if span >= DWELL_SEC * 0.9 and len(window) >= 6:
                ratio = len(seen) / len(window)
                if ratio >= GAZE_RATIO and seen:
                    cxs = [s.cx for s in seen]
                    cys = [s.cy for s in seen]
                    drift = max(max(cxs) - min(cxs), max(cys) - min(cys))
                    if drift <= DWELL_DRIFT:
                        face = seen[-1]
                        break
            if now - progress_log_ts >= 1.0:
                hits = len(seen); total = len(window)
                log.info("等待人脸稳定：近 %.1fs 命中 %d/%d 帧",
                         span, hits, total)
                progress_log_ts = now
            time.sleep(SAMPLE)

        if face is None:
            log.info("跳过：8s 内未看到稳定人脸 (window=%d hits=%d/%d)",
                     len(window),
                     sum(1 for (_, x) in window if x is not None),
                     len(window))
            return
        log.info("镜头锁住人脸 ✓  cx=%.2f cy=%.2f conf=%.2f",
                 face.cx, face.cy, face.conf)

        # 2) 等云台把脸推到画面中央再截图（最多 2.5s）
        CENTER_TOL_X = 0.12
        CENTER_TOL_Y = 0.18
        center_deadline = time.monotonic() + 2.5
        while not self._stop.is_set():
            dx = abs(face.cx - 0.5)
            dy = abs(face.cy - 0.5)
            if dx <= CENTER_TOL_X and dy <= CENTER_TOL_Y:
                break
            if time.monotonic() >= center_deadline:
                log.info("center timeout, snap anyway (dx=%.2f dy=%.2f)",
                         dx, dy)
                break
            time.sleep(0.12)
            f2 = self.perception.latest()
            if f2 is not None and f2.conf >= MIN_CONF:
                face = f2
        else:
            return  # stop set

        # 3) 居中后再"安定一下"，让画面真的稳下来再拍
        time.sleep(0.35)
        f3 = self.perception.latest()
        if f3 is not None and f3.conf >= MIN_CONF:
            face = f3
        log.info("人脸已居中 ✓  cx=%.2f cy=%.2f conf=%.2f → 准备截图",
                 face.cx, face.cy, face.conf)

        jpeg, ts = self.capture.latest.get_latest()
        if jpeg is None:
            log.info("跳过：还没有镜头帧")
            return
        # 帧与 face 时间差太大就跳过（避免裁错位置）
        if abs(ts - face.ts) > 3.0:
            log.info("跳过：镜头帧与人脸时间不同步 (%.1fs)", abs(ts - face.ts))
            return

        arr = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            log.warning("镜头帧解码失败")
            return

        crop = _crop_face(frame, face.bbox, pad=FACE_PAD)
        if crop is None:
            log.info("跳过：人脸框太小")
            return
        crop_jpeg = _encode_jpeg(crop, quality=82)
        # 调试落盘 + 历史
        result_ts = time.time()
        face_path = FACE_DIR / f"face_{int(result_ts * 1000)}.jpg"
        try:
            DEBUG_PATH.write_bytes(crop_jpeg)
            face_path.write_bytes(crop_jpeg)
        except Exception:
            log.warning("face dump failed", exc_info=True)
            face_path = Path("")
        try:
            files = sorted(FACE_DIR.glob("face_*.jpg"))
            for old in files[:-FACE_HISTORY_MAX]:
                old.unlink(missing_ok=True)
        except Exception:
            pass

        self._call_count += 1
        log.info("[#%d] 向 MiMo 发送人脸 (%dx%d, %d 字节)…",
                 self._call_count, crop.shape[1], crop.shape[0], len(crop_jpeg))
        t1 = time.monotonic()
        try:
            content, usage = self._client.socratic_vision(crop_jpeg)
        except urllib.error.HTTPError as e:
            err = e.read()[:200].decode(errors="replace")
            self._fail_count += 1
            log.warning("[#%d] MiMo HTTP %d: %s", self._call_count, e.code, err)
            return
        except Exception as e:
            self._fail_count += 1
            log.warning("[#%d] MiMo 请求异常: %s", self._call_count, e)
            return
        latency_ms = int((time.monotonic() - t1) * 1000)
        # 延迟超过 6s 警告，超过 12s 更重警告
        slow_tag = ""
        if latency_ms >= 12000:
            slow_tag = " ⚠严重偏慢"
        elif latency_ms >= 6000:
            slow_tag = " ⚠偏慢"
        parsed = _extract_json(content)
        log.info("[#%d] MiMo 返回 ⏱ %dms%s usage=%s",
                 self._call_count, latency_ms, slow_tag, json.dumps(usage))
        if not parsed:
            self._fail_count += 1
            log.warning("[#%d] 解析 JSON 失败，raw=%r",
                        self._call_count, content[:200])
            return

        speech = str(parsed.get("speech", "")).strip("\"'`「」“”　").strip()
        if speech:
            speech = speech.splitlines()[0]

        result = ExpressionResult(
            ts=result_ts,
            emotion=str(parsed.get("emotion", "neutral"))[:32],
            intensity=float(parsed.get("intensity", 0.0) or 0.0),
            attention=str(parsed.get("attention", "medium"))[:16],
            gaze=str(parsed.get("gaze", "at_camera"))[:16],
            comment=str(parsed.get("observe", ""))[:80],
            speech=speech[:120],
            face_bbox=face.bbox,
            face_path=str(face_path) if str(face_path) else "",
            raw=content,
            usage=usage,
            latency_ms=latency_ms,
        )
        with self._latest_lock:
            self._latest = result
            self._history.append(result)
            if len(self._history) > self._history_max:
                self._history.pop(0)
            self._intensity_log.append((result_ts, result.intensity))
            # 只留近 5min，防止无限增长
            cutoff = result_ts - 300.0
            self._intensity_log = [
                (t, v) for (t, v) in self._intensity_log if t >= cutoff
            ]

        log.info("[#%d] 表情=%s | 紧张=%.2f | 注意力=%s | 观察：%s | 话：%s",
                 self._call_count, result.emotion, result.intensity,
                 result.attention, result.comment, result.speech)

        # 朗读 observe（更冷静客观的描述），同一句话 60s 内不复读
        to_speak = (result.comment or speech).strip()
        if self.voice is None or not to_speak:
            return
        now = time.time()
        if to_speak != self._last_spoken or now - self._last_spoken_ts > 60:
            style_prompt, tag_prefix = style_for_intensity(
                result.intensity, result.emotion
            )
            spoken = f"{tag_prefix}{to_speak}" if tag_prefix else to_speak
            log.info("[#%d] → TTS（%s 紧张=%.2f）：%s",
                     self._call_count, tag_prefix or "中性",
                     result.intensity, to_speak)
            self.voice.say(spoken, style=style_prompt, tag="socratic")
            self._last_spoken = to_speak
            self._last_spoken_ts = now


# ---------- CLI ----------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="socsay.expression")
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    ap.add_argument("--max-iters", type=int, default=0,
                    help="0 = 无限")
    ap.add_argument("--speak", action="store_true",
                    help="同时启动 voice 把 comment 朗读出来")
    ap.add_argument("--mimo-voice", default="冰糖")
    ap.add_argument("--test-image", help="跳过相机，用这张本地图片走一轮 MiMo微表情")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _load_dotenv()

    if args.test_image:
        # 直接调 MiMo，不启动相机。这个路径是为了验证表情识别本身。
        client = _MimoVLM()
        img = cv2.imread(args.test_image, cv2.IMREAD_COLOR)
        if img is None:
            print(f"cannot read image: {args.test_image}")
            return 2
        # 限长边
        h, w = img.shape[:2]
        if max(h, w) > CROP_MAX_EDGE:
            s = CROP_MAX_EDGE / max(h, w)
            img = cv2.resize(img, (int(w*s), int(h*s)))
        jpeg = _encode_jpeg(img, quality=82)
        DEBUG_PATH.write_bytes(jpeg)
        log.info("sending test image %dx%d %d bytes…",
                 img.shape[1], img.shape[0], len(jpeg))
        t0 = time.monotonic()
        content, usage = client.socratic_vision(jpeg)
        dt = int((time.monotonic() - t0) * 1000)
        log.info("latency=%dms usage=%s", dt, json.dumps(usage))
        log.info("raw=%r", content)
        log.info("parsed=%s", _extract_json(content))
        return 0

    cap = Capture()
    cap.start()
    per = Perception(cap)
    per.start()

    voice = None
    if args.speak:
        from .voice import Voice
        voice = Voice(engine="mimo", mimo_voice=args.mimo_voice)
        voice.start()

    expr = Expression(cap, per, voice=voice, interval=args.interval)
    expr.start()

    stop = threading.Event()

    def shutdown(*_):
        log.info("stopping…")
        stop.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if args.max_iters > 0:
        # 等待大约 max_iters 次回调
        deadline = time.time() + args.max_iters * args.interval + 30
        while not stop.is_set() and expr._call_count < args.max_iters \
                and time.time() < deadline:
            time.sleep(0.5)
    else:
        while not stop.is_set():
            time.sleep(0.5)

    expr.stop()
    if voice is not None:
        voice.stop()
    per.stop()
    cap.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
