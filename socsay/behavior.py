"""行为层：5 状态 FSM + Track / Roam / Nod / Shake。"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from queue import Empty, Queue
from typing import Optional

from .perception import FaceTarget, Perception
from .ptz import PTZ, PAN_MAX, PAN_MIN, TILT_MAX, TILT_MIN

log = logging.getLogger("socsay.behavior")


class State(str, Enum):
    IDLE = "Idle"
    TRACK = "Track"
    ROAM = "Roam"
    NOD = "Nod"
    SHAKE = "Shake"


@dataclass
class Intent:
    gesture: str          # "nod" | "shake" | "center"
    reason: str = ""


# 跟踪参数
PAN_PER_UNIT = 400_000
TILT_PER_UNIT = 250_000
GAIN = 0.20             # ← 从 0.35 降低，避免超调
DEADZONE = 0.10         # ← 从 0.08 加大
MAX_DPAN = 18_000       # ← 从 30_000
MAX_DTILT = 12_000      # ← 从 25_000
MOVE_COOLDOWN = 0.40    # ← 新增：一次跟踪后等 perception 跟上
INVERT_PAN = False
INVERT_TILT = False

# Roam 参数
ROAM_STEP = 12_000
ROAM_INTERVAL = 0.5

# 手势序列：(dpan, dtilt, sleep_ms)
NOD_SEQ = [
    (0, -25_000, 220),
    (0, +50_000, 220),
    (0, -25_000, 220),
    (0, -25_000, 220),
    (0, +50_000, 220),
    (0, -25_000, 250),
]
SHAKE_SEQ = [
    (-30_000, 0, 200),
    (+60_000, 0, 200),
    (-60_000, 0, 200),
    (+60_000, 0, 200),
    (-30_000, 0, 220),
]

LOST_TO_ROAM_SEC = 12.0
TICK = 0.2  # 5 Hz


class Behavior:
    def __init__(self, ptz: PTZ, perception: Perception, auto_track: bool = True):
        self.ptz = ptz
        self.perception = perception
        self.auto_track = auto_track
        self.state = State.IDLE
        self.intents: Queue[Intent] = Queue()
        self._stop = threading.Event()
        self._last_face_ts = 0.0
        self._roam_dir = 1
        self._roam_last = 0.0
        self._last_move_ts = 0.0
        self._thread: Optional[threading.Thread] = None
        self._events: list[dict] = []  # 简单事件历史
        self._events_lock = threading.Lock()

    # ----- 公开 -----
    def start(self):
        self._thread = threading.Thread(
            target=self._loop, name="behavior", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()

    def push_intent(self, intent: Intent):
        self.intents.put(intent)

    def snapshot(self) -> dict:
        face = self.perception.latest()
        return {
            "state": self.state.value,
            "pos": list(self.ptz.last_pos),
            "face": None if not face else {
                "cx": round(face.cx, 3),
                "cy": round(face.cy, 3),
                "conf": round(face.conf, 3),
                "ts": face.ts,
            },
            "events_tail": self._events[-10:],
        }

    # ----- 主循环 -----
    def _loop(self):
        log.info("behavior loop start, state=%s auto_track=%s", self.state, self.auto_track)
        while not self._stop.is_set():
            tick_start = time.monotonic()
            try:
                self._step()
            except Exception:
                log.exception("step failed")
            elapsed = time.monotonic() - tick_start
            time.sleep(max(0.0, TICK - elapsed))

    def _step(self):
        # 优先处理 intent
        intent = None
        try:
            intent = self.intents.get_nowait()
        except Empty:
            pass

        face = self.perception.latest()
        now = time.monotonic()
        if face and face.conf > 0.0:
            self._last_face_ts = now

        if intent:
            self._handle_intent(intent)
            return

        # 如果关闭了自动追踪，就不做 roam/track
        if not self.auto_track:
            return

        # 状态转移
        if self.state in (State.IDLE, State.ROAM, State.TRACK):
            if face and face.conf > 0.5:
                if self.state != State.TRACK:
                    self._set_state(State.TRACK)
                self._do_track(face)
            else:
                if now - self._last_face_ts > LOST_TO_ROAM_SEC:
                    if self.state != State.ROAM:
                        self._set_state(State.ROAM)
                    self._do_roam()
                else:
                    if self.state == State.TRACK:
                        # 短暂丢脸不切状态，停手等等
                        pass

    # ----- 子动作 -----
    def _do_track(self, face: FaceTarget):
        # 刚动过就等 perception/干顺不才发下一个指令，避免超调反弹
        if time.monotonic() - self._last_move_ts < MOVE_COOLDOWN:
            return
        ex = face.cx - 0.5
        ey = face.cy - 0.5
        if abs(ex) < DEADZONE and abs(ey) < DEADZONE:
            return
        dpan = -ex * PAN_PER_UNIT * GAIN
        dtilt = -ey * TILT_PER_UNIT * GAIN
        if INVERT_PAN:
            dpan = -dpan
        if INVERT_TILT:
            dtilt = -dtilt
        dpan = max(-MAX_DPAN, min(MAX_DPAN, int(dpan)))
        dtilt = max(-MAX_DTILT, min(MAX_DTILT, int(dtilt)))
        cp, ct = self.ptz.last_pos
        new_p, new_t = self.ptz.move_abs(cp + dpan, ct + dtilt)
        self._last_move_ts = time.monotonic()
        # 边界检测
        if new_p in (PAN_MIN, PAN_MAX) or new_t in (TILT_MIN, TILT_MAX):
            self._emit("boundary_hit", pos=[new_p, new_t])

    def _do_roam(self):
        now = time.monotonic()
        if now - self._roam_last < ROAM_INTERVAL:
            return
        self._roam_last = now
        cp, ct = self.ptz.last_pos
        target = cp + self._roam_dir * ROAM_STEP
        if target >= PAN_MAX or target <= PAN_MIN:
            self._roam_dir *= -1
            target = cp + self._roam_dir * ROAM_STEP
        self.ptz.move_abs(target, ct)

    def _handle_intent(self, intent: Intent):
        log.info("intent: %s reason=%r", intent.gesture, intent.reason)
        g = intent.gesture.lower()
        if g == "nod":
            self._do_gesture(State.NOD, NOD_SEQ)
        elif g == "shake":
            self._do_gesture(State.SHAKE, SHAKE_SEQ)
        elif g == "center":
            self.ptz.center()
        else:
            log.warning("unknown gesture: %s", g)
        self._emit("gesture_done", name=g, reason=intent.reason)

    def _do_gesture(self, state: State, seq: list[tuple[int, int, int]]):
        prev = self.state
        self._set_state(state)
        p0, t0 = self.ptz.last_pos
        cp, ct = p0, t0
        for dp, dt_, ms in seq:
            cp += dp
            ct += dt_
            self.ptz.move_abs(cp, ct)
            time.sleep(ms / 1000.0)
        self.ptz.move_abs(p0, t0)
        self._set_state(prev)

    # ----- utils -----
    def _set_state(self, s: State):
        if s == self.state:
            return
        log.info("state: %s -> %s", self.state, s)
        self.state = s
        self._emit("state", value=s.value)

    def _emit(self, kind: str, **kw):
        ev = {"ts": time.time(), "kind": kind, **kw}
        with self._events_lock:
            self._events.append(ev)
            if len(self._events) > 200:
                del self._events[:100]
