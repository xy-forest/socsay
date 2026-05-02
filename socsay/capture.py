"""Capture：用 ffmpeg 从 AVFoundation 抓 MJPEG，并提供"最新帧"获取。

也是相机的"唤醒守门员"：只要本类在跑，相机就不会进隐私待机。
"""
from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
from typing import Optional

log = logging.getLogger("socsay.capture")

DEVICE_NAME = "Insta360 Link 2"
SOI = b"\xff\xd8"
EOI = b"\xff\xd9"


def _find_device_index() -> Optional[str]:
    """返回 AVFoundation 设备索引字符串，找不到返回 None。"""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "avfoundation",
             "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=5,
        )
    except Exception as e:
        log.error("list_devices failed: %s", e)
        return None
    text = (r.stderr or "") + (r.stdout or "")
    # ffmpeg 输出形如：[AVFoundation indev @ 0x...] [1] Insta360 Link 2
    for line in text.splitlines():
        if DEVICE_NAME in line:
            m = re.search(r"\[(\d+)\]\s*" + re.escape(DEVICE_NAME), line)
            if m:
                return m.group(1)
    return None


class LatestFrame:
    """单槽位最新帧缓冲。"""

    def __init__(self):
        self._cv = threading.Condition()
        self._buf: Optional[bytes] = None
        self._ts = 0.0
        self._gen = 0

    def put(self, jpeg: bytes):
        with self._cv:
            self._buf = jpeg
            self._ts = time.monotonic()
            self._gen += 1
            self._cv.notify_all()

    def get_latest(self) -> tuple[Optional[bytes], float]:
        with self._cv:
            return self._buf, self._ts

    def wait_new(self, last_gen: int, timeout: float = 1.0
                 ) -> tuple[Optional[bytes], float, int]:
        with self._cv:
            self._cv.wait_for(lambda: self._gen != last_gen, timeout=timeout)
            return self._buf, self._ts, self._gen


class Capture:
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        # 注意：Insta360 Link 2 在某些固件/模式下 1280x720@15 不可用，30 稳。
        self.width = width
        self.height = height
        self.fps = fps
        self.latest = LatestFrame()
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self):
        idx = _find_device_index()
        if idx is None:
            raise RuntimeError(
                f"未找到 AVFoundation 设备 '{DEVICE_NAME}'，请检查 USB 连接。"
            )
        log.info("Insta360 device index = %s", idx)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "avfoundation",
            "-pixel_format", "uyvy422",
            "-framerate", str(self.fps),
            "-video_size", f"{self.width}x{self.height}",
            "-i", f"{idx}:none",
            "-an", "-f", "mjpeg", "-q:v", "5", "-",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._thread = threading.Thread(
            target=self._reader, name="capture-reader", daemon=True
        )
        self._thread.start()
        # 等第一帧
        for _ in range(50):  # 5s
            if self.latest.get_latest()[0] is not None:
                log.info("first frame received")
                return
            time.sleep(0.1)
        raise RuntimeError("ffmpeg 启动 5s 内没有收到第一帧")

    def _reader(self):
        assert self._proc and self._proc.stdout
        buf = bytearray()
        stdout = self._proc.stdout
        while not self._stop.is_set():
            chunk = stdout.read(65536)
            if not chunk:
                log.warning("ffmpeg stdout closed")
                break
            buf.extend(chunk)
            # 切 JPEG
            while True:
                start = buf.find(SOI)
                if start < 0:
                    break
                end = buf.find(EOI, start + 2)
                if end < 0:
                    if start > 0:
                        del buf[:start]
                    break
                end += 2
                jpeg = bytes(buf[start:end])
                del buf[:end]
                self.latest.put(jpeg)

    def stop(self):
        self._stop.set()
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.kill()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cap = Capture()
    cap.start()
    try:
        for i in range(5):
            time.sleep(1)
            buf, ts = cap.latest.get_latest()
            print(f"[{i}] frame size={len(buf) if buf else 0} bytes ts={ts:.3f}")
        # dump 一张
        buf, _ = cap.latest.get_latest()
        if buf:
            with open("/tmp/socsay_capture.jpg", "wb") as f:
                f.write(buf)
            print("saved /tmp/socsay_capture.jpg")
    finally:
        cap.stop()
