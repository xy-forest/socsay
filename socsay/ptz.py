"""PTZ 控制：包装 bin/uvc_ptz_get / uvc_ptz_set。"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from pathlib import Path

BIN_DIR = Path(__file__).resolve().parent.parent / "bin"
GET_BIN = str(BIN_DIR / "uvc_ptz_get")
SET_BIN = str(BIN_DIR / "uvc_ptz_set")

PAN_MIN, PAN_MAX = -300_000, 300_000
TILT_MIN, TILT_MAX = -200_000, 200_000
STEP = 3600

# 控制频率上限（秒）
MIN_INTERVAL = 0.18

_NUM_RE = re.compile(r"-?\d+")


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _round_step(v: int) -> int:
    return int(round(v / STEP)) * STEP


class PTZError(RuntimeError):
    pass


class PTZ:
    def __init__(self, invert_pan: bool = False, invert_tilt: bool = False):
        self._lock = threading.Lock()
        self._last_set_ts = 0.0
        self._last_pos = (0, 0)
        self.invert_pan = invert_pan
        self.invert_tilt = invert_tilt

    # ---- 低层 ----
    def _run(self, *args: str, timeout: float = 1.5) -> str:
        try:
            r = subprocess.run(
                args, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired as e:
            raise PTZError(f"timeout: {' '.join(args)}") from e
        if r.returncode != 0:
            raise PTZError(f"rc={r.returncode} stderr={r.stderr.strip()}")
        return r.stdout

    # ---- 公开 API ----
    def get(self) -> tuple[int, int]:
        out = self._run(GET_BIN)
        # 输出形如：{"pan":0,"tilt":0}
        m = re.search(r'"pan"\s*:\s*(-?\d+).*?"tilt"\s*:\s*(-?\d+)', out)
        if not m:
            raise PTZError(f"unparsable: {out!r}")
        pan, tilt = int(m.group(1)), int(m.group(2))
        with self._lock:
            self._last_pos = (pan, tilt)
        return pan, tilt

    def move_abs(self, pan: int, tilt: int) -> tuple[int, int]:
        if self.invert_pan:
            pan = -pan
        if self.invert_tilt:
            tilt = -tilt
        pan = _round_step(_clamp(pan, PAN_MIN, PAN_MAX))
        tilt = _round_step(_clamp(tilt, TILT_MIN, TILT_MAX))
        with self._lock:
            now = time.monotonic()
            wait = MIN_INTERVAL - (now - self._last_set_ts)
            if wait > 0:
                time.sleep(wait)
            self._run(SET_BIN, str(pan), str(tilt), timeout=2.0)
            self._last_set_ts = time.monotonic()
            self._last_pos = (pan, tilt)
            return pan, tilt

    def move_rel(self, dpan: int, dtilt: int) -> tuple[int, int]:
        with self._lock:
            cp, ct = self._last_pos
        return self.move_abs(cp + dpan, ct + dtilt)

    def center(self) -> tuple[int, int]:
        return self.move_abs(0, 0)

    @property
    def last_pos(self) -> tuple[int, int]:
        return self._last_pos


if __name__ == "__main__":
    # 自检：读 → 回中 → 读
    p = PTZ()
    print("before:", p.get())
    print("center:", p.center())
    time.sleep(0.5)
    print("after :", p.get())
