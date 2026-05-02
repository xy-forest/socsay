"""MiMo-in-the-loop 盯人模式。

每 INTERVAL 秒：
  1. 从 Capture 拿一张最新帧
  2. 缩到 ≤640px 长边，base64 → MiMo 多模态接口
  3. 解析 JSON：人脸归一化 bbox、建议方向
  4. 计算 PTZ 位移，move_abs
  5. 循环

设计选择：
- 完全不依赖本地 haar 检测，纯 LLM 驱动（用户原话）
- 失败（无人 / 解析失败 / 网络错）一律 no-op，不让相机乱转
- 单步 PTZ 位移做 clamp，避免一次抡到天花板
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import urllib.request
import urllib.error

from .capture import Capture
from .ptz import PTZ

log = logging.getLogger("socsay.gaze")

# ---- 跟踪映射常量（沿用 behavior.py） ----
PAN_PER_UNIT = 400_000
TILT_PER_UNIT = 250_000
GAIN = 0.55              # MiMo 周期长，单步可以激进些
DEADZONE = 0.10
MAX_DPAN = 80_000
MAX_DTILT = 60_000

DEFAULT_INTERVAL = 3.0
RESIZE_LONG_EDGE = 640

# 扫描参数（没看到人时）
SEARCH_PAN_STEP = 90_000      # 每次pan跳这么多
SEARCH_TILT_LEVELS = [0, 60_000, -40_000, 120_000]  # 依次换个仕角
SEARCH_PAN_RANGE = 270_000    # 水平扫描范围 ±


def _load_dotenv(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


PROMPT = (
    "Output ONLY a JSON object. No prose, no markdown.\n"
    "Schema:\n"
    "{\"has_person\":bool,\"is_face_visible\":bool,"
    "\"face_box\":[x,y,w,h]|null,\"person_center\":[cx,cy]|null,"
    "\"suggestion\":\"<<=20 chars CN>>\"}\n"
    "Coordinates normalized 0..1, origin top-left.\n"
    "person_center: prefer face center; else torso center.\n"
)


class MiMoClient:
    def __init__(self):
        self.api_key = os.environ["MIMO_API_KEY"]
        self.endpoint = os.environ.get(
            "MIMO_OPENAI_CHAT_COMPLETIONS",
            "https://api.xiaomimimo.com/v1/chat/completions",
        )
        self.model = "mimo-v2.5"

    def understand(self, jpeg_bytes: bytes, timeout: float = 30.0) -> dict:
        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
            "max_completion_tokens": 2048,
            "temperature": 0.1,
            "stream": False,
        }
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "api-key": self.api_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        msg = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage", {})
        parsed = _extract_json(msg)
        return {"raw": msg, "parsed": parsed, "usage": usage}


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    # 去掉 ```json ... ``` 包裹
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _resize_for_upload(jpeg_bytes: bytes) -> bytes:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jpeg_bytes
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= RESIZE_LONG_EDGE:
        return jpeg_bytes
    scale = RESIZE_LONG_EDGE / long_edge
    nh, nw = int(h * scale), int(w * scale)
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return jpeg_bytes
    return enc.tobytes()


def _annotate(jpeg_bytes: bytes, parsed: dict | None, suggestion: str = ""):
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return
    h, w = img.shape[:2]
    cv2.line(img, (w // 2, 0), (w // 2, h), (80, 80, 80), 1)
    cv2.line(img, (0, h // 2), (w, h // 2), (80, 80, 80), 1)
    if parsed:
        box = parsed.get("face_box")
        if box and len(box) == 4:
            x = int(box[0] * w); y = int(box[1] * h)
            bw = int(box[2] * w); bh = int(box[3] * h)
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 200, 0), 2)
        c = parsed.get("person_center")
        if c and len(c) == 2:
            cx = int(c[0] * w); cy = int(c[1] * h)
            cv2.circle(img, (cx, cy), 8, (0, 0, 255), -1)
    if suggestion:
        cv2.putText(img, suggestion, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 220, 220), 2, cv2.LINE_AA)
    cv2.imwrite("/tmp/socsay_gaze.jpg", img)


def _decide_move(parsed: dict | None) -> tuple[int, int, tuple[float, float] | None]:
    """返回 (dpan, dtilt, target_norm_or_None)。"""
    if not parsed:
        return 0, 0, None
    if not parsed.get("has_person"):
        return 0, 0, None
    c = parsed.get("person_center")
    if not c or len(c) != 2:
        return 0, 0, None
    try:
        cx = float(c[0]); cy = float(c[1])
    except Exception:
        return 0, 0, None
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    ex = cx - 0.5
    ey = cy - 0.5
    if abs(ex) < DEADZONE and abs(ey) < DEADZONE:
        return 0, 0, (cx, cy)
    dpan = -ex * PAN_PER_UNIT * GAIN
    dtilt = -ey * TILT_PER_UNIT * GAIN
    dpan = max(-MAX_DPAN, min(MAX_DPAN, int(dpan)))
    dtilt = max(-MAX_DTILT, min(MAX_DTILT, int(dtilt)))
    return dpan, dtilt, (cx, cy)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="socsay.gaze")
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                    help="MiMo 调用间隔秒（默认 3）")
    ap.add_argument("--max-iters", type=int, default=0,
                    help="最大迭代次数，0=无限")
    ap.add_argument("--dry-run", action="store_true",
                    help="只调用 MiMo，不发 PTZ 命令")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    _load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    cap = Capture()
    cap.start()
    ptz = PTZ()
    try:
        cur = ptz.get()
        log.info("initial ptz pos = %s", cur)
    except Exception as e:
        log.warning("ptz get at boot failed: %s", e)
    mimo = MiMoClient()

    stop = False

    def _sig(*_):
        nonlocal stop
        stop = True
        log.info("stop requested")

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    log.info("entering gaze loop, interval=%.1fs", args.interval)
    iters = 0
    # 扫描状态
    search_pan = 0
    search_dir = 1
    search_tilt_idx = 0
    last_seen_ts = 0.0
    try:
        while not stop:
            iters += 1
            t0 = time.monotonic()
            buf, ts = cap.latest.get_latest()
            if buf is None:
                log.warning("no frame yet, skip")
                time.sleep(0.5)
                continue
            small = _resize_for_upload(buf)
            log.info("[%d] sending frame (%d -> %d bytes)…",
                     iters, len(buf), len(small))
            try:
                res = mimo.understand(small)
            except urllib.error.HTTPError as e:
                log.error("MiMo HTTP %s: %s", e.code, e.read()[:200])
                time.sleep(args.interval)
                continue
            except Exception as e:
                log.error("MiMo failed: %s", e)
                time.sleep(args.interval)
                continue
            parsed = res["parsed"]
            usage = res.get("usage", {})
            if parsed is None:
                log.warning("[%d] parse failed, raw=%r", iters, res["raw"][:300])
            log.info("[%d] mimo parsed=%s usage=%s",
                     iters, parsed, json.dumps(usage, ensure_ascii=False))
            dpan, dtilt, target = _decide_move(parsed)
            suggestion = ""
            if parsed and isinstance(parsed.get("suggestion"), str):
                suggestion = parsed["suggestion"]
            _annotate(buf, parsed, suggestion)

            if dpan == 0 and dtilt == 0:
                if target is None:
                    # 没人：主动扫下一个位置
                    if args.dry_run:
                        log.info("[%d] DRY: 扫描下一位", iters)
                    else:
                        search_pan += search_dir * SEARCH_PAN_STEP
                        if abs(search_pan) > SEARCH_PAN_RANGE:
                            search_dir *= -1
                            search_pan = max(-SEARCH_PAN_RANGE,
                                             min(SEARCH_PAN_RANGE, search_pan))
                            search_tilt_idx = (search_tilt_idx + 1) % len(SEARCH_TILT_LEVELS)
                        tilt_v = SEARCH_TILT_LEVELS[search_tilt_idx]
                        ptz.move_abs(search_pan, tilt_v)
                        log.info("[%d] 未见人，扫到 (pan=%d, tilt=%d)",
                                 iters, search_pan, tilt_v)
                else:
                    log.info("[%d] 在死区内 (cx=%.2f, cy=%.2f)，保持",
                             iters, *target)
                    last_seen_ts = time.monotonic()
            else:
                cp, ct = ptz.last_pos
                np_, nt_ = cp + dpan, ct + dtilt
                if args.dry_run:
                    log.info("[%d] DRY: would move dpan=%d dtilt=%d -> (%d,%d)",
                             iters, dpan, dtilt, np_, nt_)
                else:
                    new_pos = ptz.move_abs(np_, nt_)
                    # 同步 search_pan 到1下最新位置，以免下次丢人后从远点重扫
                    search_pan = new_pos[0]
                    log.info("[%d] 跟随 → %s   '%s'",
                             iters, new_pos, suggestion)
                    last_seen_ts = time.monotonic()

            if args.max_iters and iters >= args.max_iters:
                break
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, args.interval - elapsed))
    finally:
        cap.stop()
    log.info("gaze loop finished after %d iters", iters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
