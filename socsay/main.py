"""socsay 主入口：拉起 Capture → Perception → Behavior，并启动一个最小 HTTP 接口。

HTTP（默认 127.0.0.1:8788）：
  GET  /state             当前状态/位置/脸/最近事件
  POST /intent            body: {"gesture":"nod"|"shake"|"center","reason":"..."}
  GET  /debug.jpg         最新带脸框的调试帧（来自 /tmp/socsay_debug.jpg）
  GET  /voice/state       发声器状态
  POST /voice/say         body: {"text":"...","voice":"Tingting","rate":180,"interrupt":false,"tag":""}
  POST /voice/play        body: {"path":"/abs/path.wav","volume":1.0,"interrupt":false,"tag":""}
  POST /voice/stop        停掉当前一条发声
  POST /voice/clear       清空待播队列
  GET  /expression/state  最近一次微表情识别结果 + 调用计数
  GET  /expression/face.jpg 最近一次裁出的脸帧（送给 MiMo 的那张）
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from .behavior import Behavior, Intent
from .capture import Capture
from .expression import Expression
from .perception import Perception
from .ptz import PTZ
from .voice import Voice, _load_dotenv
from .attention import Attention, EnterSuppressor, random_nudge, random_provoke
from .expression import style_for_intensity
from .web import install_log_bus, write_mjpeg, write_sse, INDEX_HTML

log = logging.getLogger("socsay.main")


def make_handler(behavior: Behavior, voice: Voice,
                 expression: Optional[Expression],
                 attention: Optional[Attention] = None,
                 suppressor: Optional[EnterSuppressor] = None,
                 capture: Optional[Capture] = None,
                 log_bus=None):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # 静默默认 access log
            return

        def _json(self, code: int, payload):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                data = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if self.path.startswith("/stream.mjpg"):
                if capture is None:
                    self._json(503, {"error": "no capture"})
                    return
                write_mjpeg(self,
                            lambda: capture.latest.get_latest(),
                            fps=15.0)
                return
            if self.path.startswith("/logs"):
                if log_bus is None:
                    self._json(503, {"error": "log bus disabled"})
                    return
                write_sse(self, log_bus)
                return
            if self.path == "/attention/state":
                if attention is None:
                    self._json(200, {"enabled": False})
                    return
                snap = attention.snapshot()
                snap["suppressor"] = (suppressor.snapshot()
                                       if suppressor else None)
                self._json(200, {"enabled": True, **snap})
                return
            if self.path == "/state":
                self._json(200, behavior.snapshot())
            elif self.path == "/voice/state":
                self._json(200, voice.snapshot())
            elif self.path == "/expression/state":
                if expression is None:
                    self._json(200, {"enabled": False})
                else:
                    self._json(200, {"enabled": True, "paused": expression.paused, **expression.snapshot()})
            elif self.path == "/expression/face.jpg":
                p = Path("/tmp/socsay_expr.jpg")
                if not p.exists():
                    self._json(404, {"error": "no expr face yet"})
                    return
                data = p.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path.startswith("/expression/face/") \
                    and self.path.endswith(".jpg"):
                # /expression/face/<unix_ms>.jpg
                stem = self.path.rsplit("/", 1)[-1].split(".", 1)[0]
                if not stem.isdigit():
                    self._json(400, {"error": "bad ts"})
                    return
                p = Path("/tmp/socsay_faces") / f"face_{stem}.jpg"
                if not p.exists():
                    self._json(404, {"error": "no such face"})
                    return
                data = p.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "max-age=3600")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path.startswith("/expression/history"):
                if expression is None:
                    self._json(200, {"enabled": False, "items": []})
                else:
                    self._json(200, {"enabled": True,
                                      "items": expression.history(limit=30)})
            elif self.path == "/debug.jpg":
                p = Path("/tmp/socsay_debug.jpg")
                if not p.exists():
                    self._json(404, {"error": "no debug frame yet"})
                    return
                data = p.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            try:
                body = self._read_json()
            except Exception:
                self._json(400, {"error": "bad json"})
                return

            if self.path == "/intent":
                g = (body.get("gesture") or "").strip()
                if g not in ("nod", "shake", "center"):
                    self._json(400, {"error": "gesture must be nod|shake|center"})
                    return
                behavior.push_intent(Intent(gesture=g, reason=body.get("reason", "")))
                self._json(200, {"ok": True, "queued": g})
                return

            if self.path == "/voice/say":
                text = (body.get("text") or "").strip()
                if not text:
                    self._json(400, {"error": "text required"})
                    return
                voice.say(
                    text,
                    voice=body.get("voice"),
                    rate=body.get("rate"),
                    style=body.get("style"),
                    engine=body.get("engine"),
                    interrupt=bool(body.get("interrupt", False)),
                    tag=str(body.get("tag", "http")),
                )
                self._json(200, {"ok": True, "queued": "tts",
                                 "queue_size": voice.snapshot()["queue_size"]})
                return

            if self.path == "/voice/play":
                path = (body.get("path") or "").strip()
                if not path:
                    self._json(400, {"error": "path required"})
                    return
                try:
                    voice.play_file(
                        path,
                        volume=body.get("volume"),
                        interrupt=bool(body.get("interrupt", False)),
                        tag=str(body.get("tag", "http")),
                    )
                except FileNotFoundError:
                    self._json(404, {"error": f"file not found: {path}"})
                    return
                self._json(200, {"ok": True, "queued": "file",
                                 "queue_size": voice.snapshot()["queue_size"]})
                return

            if self.path == "/voice/stop":
                killed = voice.stop_current()
                self._json(200, {"ok": True, "killed": killed})
                return

            if self.path == "/voice/clear":
                n = voice.clear()
                self._json(200, {"ok": True, "cleared": n})
                return

            if self.path == "/expression/start":
                if expression is not None:
                    expression.resume()
                    self._json(200, {"ok": True, "paused": False})
                else:
                    self._json(503, {"error": "expression not enabled"})
                return

            if self.path == "/expression/stop":
                if expression is not None:
                    expression.pause()
                    self._json(200, {"ok": True, "paused": True})
                else:
                    self._json(503, {"error": "expression not enabled"})
                return

            self._json(404, {"error": "not found"})

    return H


def main(argv=None):
    ap = argparse.ArgumentParser(prog="socsay")
    ap.add_argument("--no-track", action="store_true",
                    help="只跑 capture+perception，不发 PTZ 命令（调试用）")
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--hello", action="store_true",
                    help="启动时朗读一句招呼")
    ap.add_argument("--voice-engine", choices=["mimo", "system"], default="mimo",
                    help="默认 TTS 后端")
    ap.add_argument("--mimo-voice", default="冰糖",
                    help="mimo 预置音色")
    ap.add_argument("--no-expression", action="store_true",
                    help="不启动微表情循环")
    ap.add_argument("--expression-interval", type=float, default=3.0,
                    help="微表情循环间隔秒")
    ap.add_argument("--expression-speak", action="store_true",
                    help="表情 comment 自动读出来")
    ap.add_argument("--no-attention", action="store_true",
                    help="不启动前台 App 检测/回车拦截")
    ap.add_argument("--target-app", action="append", default=[],
                    help="变动拦截目标 App名，多次传入。默认 Code/Visual Studio Code")
    ap.add_argument("--no-suppressor", action="store_true",
                    help="只检测不拦截回车键")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log_bus = install_log_bus(
        level=logging.DEBUG if args.verbose else logging.INFO
    )
    _load_dotenv()  # 让 MIMO_API_KEY 等可被 voice 读到

    cap = Capture()
    cap.start()
    per = Perception(cap)
    per.start()
    ptz = PTZ()
    try:
        ptz.get()  # 唤醒读一次
    except Exception as e:
        log.warning("ptz get failed at boot: %s", e)

    behavior = Behavior(ptz, per, auto_track=not args.no_track)
    behavior.start()
    if args.no_track:
        log.warning("--no-track 模式：不自动追踪/漫游，但仍然处理 nod/shake 手势")

    voice = Voice(engine=args.voice_engine, mimo_voice=args.mimo_voice)
    voice.start()
    if args.hello:
        voice.say("苏格拉底已就绪", tag="boot")

    expression: Optional[Expression] = None
    if not args.no_expression:
        expression = Expression(
            cap, per,
            voice=voice if args.expression_speak else None,
            interval=args.expression_interval,
        )
        expression.start()

    attention: Optional[Attention] = None
    suppressor: Optional[EnterSuppressor] = None
    if not args.no_attention:
        target_apps = set(args.target_app) if args.target_app else None
        attention = Attention(
            expression=expression,
            target_apps=target_apps if target_apps else {"Code", "Visual Studio Code", "Code - Insiders", "Electron", "Cursor"},
        )
        attention.start()
        if not args.no_suppressor:
            def on_blocked():
                # 看近 1min 平均 intensity：偏高 → 挑衅；否则温柔
                avg, n, mx = (0.0, 0, 0.0)
                if expression is not None:
                    avg, n, mx = expression.recent_intensity(60.0)
                tense = (n >= 2 and (avg >= 0.55 or mx >= 0.70))
                if tense:
                    msg = random_provoke()
                    style_prompt, tag_prefix = style_for_intensity(
                        max(avg, mx, 0.80), "impatient"
                    )
                else:
                    msg = random_nudge()
                    style_prompt, tag_prefix = style_for_intensity(0.35, "")
                spoken = f"{tag_prefix}{msg}" if tag_prefix else msg
                log.info("[suppressor] 吞掉回车 → %s（近1min intensity avg=%.2f"
                         " max=%.2f n=%d %s）",
                         msg, avg, mx, n,
                         "挑衅档" if tense else "温柔档")
                voice.say(spoken, style=style_prompt,
                          tag="suppressor", interrupt=True)
                # 🎯 摇头：物理拒绝！
                behavior.push_intent(Intent(gesture="shake", reason="enter_blocked"))
            suppressor = EnterSuppressor(
                active_fn=lambda: (attention.should_intervene()[0]
                                   if attention else False),
                on_blocked=on_blocked,
            )
            suppressor.start()

        # 🎯 点头监视器：紧张→平静时摄像头点头 + MiMo 语音认可
        def _nod_watcher():
            import time as _t
            was_tense = False
            while True:
                _t.sleep(3.0)
                if expression is None or expression.paused:
                    continue
                avg, n, mx = expression.recent_intensity(30.0)
                if n < 2:
                    continue
                is_tense = (avg >= 0.45 or mx >= 0.60)
                if was_tense and not is_tense:
                    log.info("[nod_watcher] 紧张→平静，点头 ✓")
                    behavior.push_intent(Intent(gesture="nod", reason="calm_after_tense"))
                    voice.say("嗯，你想通了。", style="语调温柔放松，像在肯定对方", tag="nod")
                was_tense = is_tense

        threading.Thread(target=_nod_watcher, name="nod_watcher", daemon=True).start()

        # 🎯 后台预热：把所有话术提前合成到缓存，路演时秒出声
        def _warmup():
            import time as _t
            _t.sleep(2.0)
            from .attention import SHORT_NUDGES, PROVOKE_NUDGES
            # 温柔档 style
            _, gentle_prefix = style_for_intensity(0.35, "")
            gentle_style, _ = style_for_intensity(0.35, "")
            # 挑衅档 style
            provoke_style, provoke_prefix = style_for_intensity(0.80, "impatient")

            all_items = []
            for line in SHORT_NUDGES:
                spoken = f"{gentle_prefix}{line}" if gentle_prefix else line
                all_items.append((spoken, gentle_style))
            for line in PROVOKE_NUDGES:
                spoken = f"{provoke_prefix}{line}" if provoke_prefix else line
                all_items.append((spoken, provoke_style))
            all_items.append(("嗯，你想通了。", "语调温柔放松，像在肯定对方"))

            log.info("[warmup] 开始预热 %d 句话术 TTS 缓存…", len(all_items))
            ok = 0
            for i, (text, style) in enumerate(all_items):
                try:
                    voice.mimo.synthesize(text=text, voice=None, style=style)
                    ok += 1
                except Exception as e:
                    log.debug("[warmup] %d/%d 失败: %s", i+1, len(all_items), e)
                _t.sleep(0.3)
            log.info("[warmup] 预热完成 %d/%d", ok, len(all_items))

        threading.Thread(target=_warmup, name="tts_warmup", daemon=True).start()


    httpd = ThreadingHTTPServer(
        ("127.0.0.1", args.port),
        make_handler(behavior, voice, expression,
                     attention=attention, suppressor=suppressor,
                     capture=cap, log_bus=log_bus),
    )
    http_thread = threading.Thread(
        target=httpd.serve_forever, name="http", daemon=True
    )
    http_thread.start()
    log.info("HTTP listening on http://127.0.0.1:%d", args.port)
    log.info("打开浏览器: http://127.0.0.1:%d/", args.port)
    log.info("try: curl -s localhost:%d/state | jq", args.port)
    log.info("     curl -X POST localhost:%d/intent -d '{\"gesture\":\"nod\"}'",
             args.port)

    stop = threading.Event()

    def shutdown(*_):
        log.info("shutting down…")
        stop.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        while not stop.is_set():
            stop.wait(0.5)
    finally:
        try:
            httpd.shutdown()
        except Exception:
            pass
        if expression is not None:
            expression.stop()
        if attention is not None:
            attention.stop()
        if suppressor is not None:
            suppressor.stop()
        behavior.stop()
        per.stop()
        cap.stop()
        voice.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
