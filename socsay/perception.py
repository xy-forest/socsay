"""感知层：本地 OpenCV haar 人脸检测 + 平滑输出 FaceTarget。

Phase 1 暂不接 MiMo 校准（见 docs/perception.md §4），先把本地链路跑通。
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .capture import Capture

log = logging.getLogger("socsay.perception")


@dataclass
class FaceTarget:
    cx: float                    # 0..1
    cy: float                    # 0..1
    bbox: tuple[float, float, float, float]  # x,y,w,h normalized
    conf: float                  # 0..1
    ts: float
    source: str = "local"


# 平滑系数
ALPHA = 0.75            # ← 从 0.5 调高，给跟踪循环更新鲜的位置
LOST_THRESHOLD_FRAMES = 8
# 检测时下采样到这个宽度（60w 上 haar 快 4 倍）
DETECT_WIDTH = 640
# 脸宽 / 图宽 在 这个区间线性映射到 conf 0.2..0.95
CONF_FACE_RATIO_LOW = 0.06   # ~80px / 1280
CONF_FACE_RATIO_HIGH = 0.30  # ~380px / 1280


class Perception:
    def __init__(self, capture: Capture, save_debug: bool = True):
        self.capture = capture
        self.save_debug = save_debug
        # alt2 召回优于 default，对侧脸/透视变形更鲁棒
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        )
        if self._cascade.empty():
            raise RuntimeError("haar cascade load failed")
        # 侧脸兑底，profileface 只判右侧脸，左侧需镜像后再跑一次
        self._profile = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        self._latest: Optional[FaceTarget] = None
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._smooth_cx: Optional[float] = None
        self._smooth_cy: Optional[float] = None
        self._lost_count = 0
        self._frame_count = 0
        self._dt_sum = 0.0
        self._fps_log_ts = time.monotonic()

    def start(self):
        self._thread = threading.Thread(
            target=self._loop, name="perception", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()

    def latest(self) -> Optional[FaceTarget]:
        with self._latest_lock:
            return self._latest

    def _loop(self):
        last_gen = -1
        while not self._stop.is_set():
            buf, ts, gen = self.capture.latest.wait_new(last_gen, timeout=1.0)
            if buf is None or gen == last_gen:
                continue
            last_gen = gen
            t0 = time.monotonic()
            try:
                target = self._detect(buf, ts)
            except Exception as e:
                log.exception("detect failed: %s", e)
                continue
            dt = (time.monotonic() - t0) * 1000
            with self._latest_lock:
                self._latest = target
            self._frame_count += 1
            self._dt_sum += dt
            now = time.monotonic()
            if now - self._fps_log_ts >= 5.0:
                fps = self._frame_count / (now - self._fps_log_ts)
                avg = self._dt_sum / max(1, self._frame_count)
                log.info("perception fps=%.1f avg=%.1fms last_conf=%.2f",
                         fps, avg, target.conf)
                self._frame_count = 0
                self._dt_sum = 0.0
                self._fps_log_ts = now

    def _detect(self, jpeg: bytes, ts: float) -> FaceTarget:
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return FaceTarget(0.5, 0.5, (0, 0, 0, 0), 0.0, ts)
        h, w = img.shape[:2]
        # 下采样加速
        scale = DETECT_WIDTH / float(w) if w > DETECT_WIDTH else 1.0
        if scale != 1.0:
            small = cv2.resize(img, (DETECT_WIDTH, int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            small = img
        sh, sw = small.shape[:2]
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        # minSize 也要按下采样后尺寸；40px@640w 对应 80px@1280w
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=2,
            minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE,
        )
        # 正脸没找到 → 试侧脸（右）及镜像后的左侧
        if len(faces) == 0 and not self._profile.empty():
            pf_r = self._profile.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=4,
                minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE,
            )
            pf_l = self._profile.detectMultiScale(
                cv2.flip(gray, 1), scaleFactor=1.2, minNeighbors=4,
                minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE,
            )
            # 左侧检测是在镜像上，还原坐标
            pf_l = [(sw - x - w, y, w, h) for (x, y, w, h) in pf_l]
            faces = list(pf_r) + list(pf_l)
        if len(faces) == 0:
            self._lost_count += 1
            if self._lost_count >= LOST_THRESHOLD_FRAMES:
                self._smooth_cx = self._smooth_cy = None
            self._maybe_save_debug(img, None)
            return FaceTarget(0.5, 0.5, (0, 0, 0, 0), 0.0, ts)

        # 取最大的脸（下采样坐标系）
        x_s, y_s, fw_s, fh_s = max(faces, key=lambda b: b[2] * b[3])
        # 还原到原图坐标
        x = int(x_s / scale); y = int(y_s / scale)
        fw = int(fw_s / scale); fh = int(fh_s / scale)
        cx_raw = (x + fw / 2) / w
        cy_raw = (y + fh / 2) / h
        if self._smooth_cx is None:
            self._smooth_cx, self._smooth_cy = cx_raw, cy_raw
        else:
            self._smooth_cx = ALPHA * cx_raw + (1 - ALPHA) * self._smooth_cx
            self._smooth_cy = ALPHA * cy_raw + (1 - ALPHA) * self._smooth_cy
        self._lost_count = 0

        # 置信度：按脸宽占图宽比例线性映射到 0.2..0.95
        ratio = fw / w
        if ratio <= CONF_FACE_RATIO_LOW:
            conf = 0.2
        elif ratio >= CONF_FACE_RATIO_HIGH:
            conf = 0.95
        else:
            conf = 0.2 + 0.75 * (ratio - CONF_FACE_RATIO_LOW) / (
                CONF_FACE_RATIO_HIGH - CONF_FACE_RATIO_LOW
            )
        target = FaceTarget(
            cx=self._smooth_cx,
            cy=self._smooth_cy,
            bbox=(x / w, y / h, fw / w, fh / h),
            conf=conf,
            ts=ts,
        )
        self._maybe_save_debug(img, (x, y, fw, fh))
        return target

    def _maybe_save_debug(self, img, box):
        if not self.save_debug:
            return
        # 限速：~2 Hz 写一次
        now = time.monotonic()
        last = getattr(self, "_dbg_last", 0.0)
        if now - last < 0.5:
            return
        self._dbg_last = now
        out = img.copy()
        h, w = out.shape[:2]
        cv2.line(out, (w // 2, 0), (w // 2, h), (80, 80, 80), 1)
        cv2.line(out, (0, h // 2), (w, h // 2), (80, 80, 80), 1)
        if box is not None:
            x, y, fw, fh = box
            cv2.rectangle(out, (x, y), (x + fw, y + fh), (0, 200, 0), 2)
            cx, cy = x + fw // 2, y + fh // 2
            cv2.circle(out, (cx, cy), 6, (0, 0, 255), -1)
        try:
            cv2.imwrite("/tmp/socsay_debug.jpg", out)
        except Exception:
            pass


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cap = Capture()
    cap.start()
    per = Perception(cap)
    per.start()
    try:
        for _ in range(20):
            time.sleep(0.5)
            t = per.latest()
            if t and t.conf > 0:
                print(f"face cx={t.cx:.3f} cy={t.cy:.3f} conf={t.conf:.2f}")
            else:
                print("no face")
    finally:
        per.stop()
        cap.stop()
