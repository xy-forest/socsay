"""socsay.voice — 让伙伴有"嘴"。

封装 macOS 自带的 `say` (TTS) 和 `afplay` (放音频文件)，统一成一个串行播放队列：
后到的发声请求默认排队，避免互相打断；也可以用 interrupt=True 立刻打断当前发声。

公开 API：
    v = Voice()
    v.start()
    v.say("再想想，你为什么觉得这是真正的问题？")            # 排队
    v.say("等一下，先听这句", interrupt=True)               # 打断
    v.play_file("/path/to/click.wav")                      # 排队播音频
    v.stop_current()                                       # 停掉正在发的那一条
    v.clear()                                              # 清空队列（不停当前）
    v.snapshot()                                           # 看状态
    v.stop()                                               # 关闭

CLI（自测）：
    python -m socsay.voice "你好，我是苏格拉底"
    python -m socsay.voice --voice Tingting --rate 180 "你在想什么？"
    python -m socsay.voice --file /System/Library/Sounds/Tink.aiff
    python -m socsay.voice --list-voices
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from base64 import b64decode
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Optional

log = logging.getLogger("socsay.voice")

DEFAULT_VOICE = "Tingting"   # zh_CN
DEFAULT_RATE = 180           # 词/分钟，say 默认 ~175
SAY_BIN = "/usr/bin/say"
AFPLAY_BIN = "/usr/bin/afplay"

# ---- MiMo TTS 默认参数 ----
MIMO_TTS_MODEL = "mimo-v2.5-tts"
MIMO_TTS_VOICE = "冰糖"          # 预置中文女声
# CLI/HTTP 允许传 ASCII 别名（避免 zsh/curl 中文引号麻烦）
MIMO_VOICE_ALIASES = {
    "bingtang": "冰糖", "binghua": "冰糖",
    "moli": "茉莉", "jasmine": "茉莉",
    "suda": "苏打", "soda": "苏打",
    "baihua": "白桦", "birch": "白桦",
    "default": "mimo_default",
}


def _normalize_mimo_voice(name: Optional[str]) -> Optional[str]:
    if not name:
        return name
    return MIMO_VOICE_ALIASES.get(name.strip().lower(), name)
MIMO_TTS_FORMAT = "wav"
MIMO_TTS_TIMEOUT = 30
MIMO_TTS_CACHE_DIR = Path("/tmp/socsay_tts")


@dataclass
class Utterance:
    kind: str                       # "tts" | "file"
    text: str = ""
    path: str = ""
    voice: Optional[str] = None     # system: say -v ; mimo: 预置音色名
    rate: Optional[int] = None      # 仅 system
    style: Optional[str] = None     # mimo: user-message 风格提示
    engine: Optional[str] = None    # "mimo" | "system"；None = 使用 Voice 默认
    volume: Optional[float] = None  # 0.0 ~ 2.0 (afplay)
    tag: str = ""                   # 调用方自定义标签，便于日志/追踪
    enqueued_at: float = field(default_factory=time.time)


class Voice:
    """串行发声/播放器。一个工作线程消费队列，subprocess 执行。"""

    def __init__(self, default_voice: str = DEFAULT_VOICE,
                 default_rate: int = DEFAULT_RATE,
                 engine: str = "mimo",
                 mimo_voice: str = MIMO_TTS_VOICE,
                 mimo_model: str = MIMO_TTS_MODEL,
                 mimo_style: Optional[str] = None,
                 mimo_api_key: Optional[str] = None,
                 mimo_api_base: Optional[str] = None):
        if not Path(SAY_BIN).exists():
            raise RuntimeError(f"missing {SAY_BIN} (macOS only)")
        self.default_voice = default_voice
        self.default_rate = default_rate
        self.engine = engine if engine in ("mimo", "system") else "mimo"
        self.mimo = MimoTTS(
            voice=_normalize_mimo_voice(mimo_voice),
            model=mimo_model,
            style=mimo_style,
            api_key=mimo_api_key,
            api_base=mimo_api_base,
        )
        self._q: "Queue[Utterance]" = Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc_lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._current: Optional[Utterance] = None
        self._last: Optional[Utterance] = None
        self._spoken_count = 0
        self._mimo_fail_count = 0

    # ---------- lifecycle ----------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="voice", daemon=True)
        self._thread.start()
        log.info("voice worker started (engine=%s mimo_voice=%s sys_voice=%s rate=%d)",
                 self.engine, self.mimo.voice, self.default_voice, self.default_rate)

    def stop(self, drain: bool = False):
        if not drain:
            self.clear()
        self.stop_current()
        self._stop.set()
        # 推一个 sentinel 让线程立即醒
        self._q.put(Utterance(kind="__stop__"))
        if self._thread:
            self._thread.join(timeout=2.0)

    # ---------- public ops ----------
    def say(self, text: str, *, voice: Optional[str] = None,
            rate: Optional[int] = None, style: Optional[str] = None,
            engine: Optional[str] = None, interrupt: bool = False,
            tag: str = "") -> None:
        text = (text or "").strip()
        if not text:
            return
        u = Utterance(kind="tts", text=text, voice=voice, rate=rate,
                      style=style, engine=engine, tag=tag)
        self._enqueue(u, interrupt=interrupt)

    def play_file(self, path: str, *, volume: Optional[float] = None,
                  interrupt: bool = False, tag: str = "") -> None:
        if not Path(path).exists():
            raise FileNotFoundError(path)
        u = Utterance(kind="file", path=str(path), volume=volume, tag=tag)
        self._enqueue(u, interrupt=interrupt)

    def stop_current(self) -> bool:
        """杀掉当前 subprocess（如果有）。返回是否真的杀了一个。"""
        with self._proc_lock:
            p = self._proc
            if p and p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
                return True
        return False

    def clear(self) -> int:
        """清空待播队列，返回清掉的条数。"""
        n = 0
        try:
            while True:
                self._q.get_nowait()
                n += 1
        except Empty:
            pass
        return n

    def is_speaking(self) -> bool:
        with self._proc_lock:
            return self._proc is not None and self._proc.poll() is None

    def snapshot(self) -> dict:
        cur = self._current
        last = self._last
        return {
            "speaking": self.is_speaking(),
            "queue_size": self._q.qsize(),
            "spoken_count": self._spoken_count,
            "mimo_fail_count": self._mimo_fail_count,
            "engine": self.engine,
            "mimo_voice": self.mimo.voice,
            "mimo_model": self.mimo.model,
            "default_voice": self.default_voice,
            "default_rate": self.default_rate,
            "current": _u_to_dict(cur) if cur else None,
            "last": _u_to_dict(last) if last else None,
        }

    # ---------- internal ----------
    def _enqueue(self, u: Utterance, interrupt: bool) -> None:
        if interrupt:
            self.clear()
            self.stop_current()
        self._q.put(u)
        log.debug("enqueued %s tag=%s qsize=%d interrupt=%s",
                  u.kind, u.tag, self._q.qsize(), interrupt)

    def _loop(self):
        while not self._stop.is_set():
            try:
                u = self._q.get(timeout=0.5)
            except Empty:
                continue
            if u.kind == "__stop__":
                break
            self._current = u
            try:
                self._run(u)
                self._spoken_count += 1
                self._last = u
            except Exception as e:
                log.warning("voice run failed (%s): %s", u.kind, e)
            finally:
                self._current = None

    def _run(self, u: Utterance):
        if u.kind == "tts":
            engine = u.engine or self.engine
            if engine == "mimo":
                wav_path = self._synthesize_mimo(u)
                if wav_path is None:
                    # 降级 system
                    log.warning("mimo tts failed, fallback to system say (tag=%s)", u.tag)
                    cmd = self._build_say_cmd(u)
                else:
                    cmd = [AFPLAY_BIN, str(wav_path)]
            else:
                cmd = self._build_say_cmd(u)
        elif u.kind == "file":
            cmd = [AFPLAY_BIN]
            if u.volume is not None:
                cmd += ["-v", f"{max(0.0, min(2.0, u.volume)):.2f}"]
            cmd += [u.path]
        else:
            return
        log.info("▶ %s tag=%s %s", u.kind, u.tag,
                 (u.text[:60] + ("…" if len(u.text) > 60 else ""))
                 if u.kind == "tts" else u.path)
        with self._proc_lock:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        rc = self._proc.wait()
        with self._proc_lock:
            self._proc = None
        if rc not in (0, -15, 143):  # 15/-15 = SIGTERM (我们主动 stop)
            log.warning("voice cmd exited rc=%d cmd=%s", rc, cmd)

    def _build_say_cmd(self, u: Utterance) -> list:
        voice = u.voice or self.default_voice
        rate = u.rate or self.default_rate
        return [SAY_BIN, "-v", voice, "-r", str(int(rate)), "--", u.text]

    def _synthesize_mimo(self, u: Utterance) -> Optional[Path]:
        try:
            return self.mimo.synthesize(
                text=u.text,
                voice=u.voice,           # 覆盖预置音色
                style=u.style,
            )
        except Exception as e:
            self._mimo_fail_count += 1
            log.warning("mimo synth error: %s", e)
            return None


def _u_to_dict(u: Utterance) -> dict:
    return {
        "kind": u.kind,
        "text": u.text,
        "path": u.path,
        "voice": u.voice,
        "rate": u.rate,
        "style": u.style,
        "engine": u.engine,
        "tag": u.tag,
        "age_ms": int((time.time() - u.enqueued_at) * 1000),
    }


# ---------- MiMo TTS 后端 ----------

class MimoTTS:
    """调用 MiMo-V2.5-TTS 合成 wav，本地缓存。"""

    def __init__(self, voice: str = MIMO_TTS_VOICE,
                 model: str = MIMO_TTS_MODEL,
                 style: Optional[str] = None,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 cache_dir: Path = MIMO_TTS_CACHE_DIR,
                 timeout: float = MIMO_TTS_TIMEOUT):
        self.voice = voice
        self.model = model
        self.style = style or "语调平和、温暖、像与朋友低声交谈"
        self.api_key = api_key or os.environ.get("MIMO_API_KEY", "")
        self.api_base = (api_base or os.environ.get("MIMO_API_BASE")
                         or "https://api.xiaomimimo.com").rstrip("/")
        self.timeout = timeout
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, text: str, voice: str, style: str) -> str:
        h = hashlib.sha1()
        h.update(self.model.encode())
        h.update(b"|")
        h.update(voice.encode())
        h.update(b"|")
        h.update(style.encode())
        h.update(b"|")
        h.update(text.encode())
        return h.hexdigest()

    def synthesize(self, text: str, voice: Optional[str] = None,
                   style: Optional[str] = None) -> Path:
        if not self.api_key:
            raise RuntimeError("MIMO_API_KEY not set")
        v = _normalize_mimo_voice(voice or self.voice)
        s = style or self.style
        key = self._cache_key(text, v, s)
        out = self.cache_dir / f"{key}.wav"
        if out.exists() and out.stat().st_size > 44:
            log.debug("mimo cache hit %s", out.name)
            return out
        url = f"{self.api_base}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": s},
                {"role": "assistant", "content": text},
            ],
            "audio": {"format": MIMO_TTS_FORMAT, "voice": v},
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "api-key": self.api_key,
            },
            method="POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            err_body = e.read()[:300].decode(errors="replace")
            raise RuntimeError(f"http {e.code}: {err_body}") from e
        data = json.loads(raw)
        try:
            audio_b64 = data["choices"][0]["message"]["audio"]["data"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"no audio in response: {raw[:200]!r}")
        wav_bytes = b64decode(audio_b64)
        # 原子写入
        tmp = out.with_suffix(".wav.partial")
        tmp.write_bytes(wav_bytes)
        tmp.replace(out)
        log.info("mimo synth ok voice=%s bytes=%d in %.2fs (cache=%s)",
                 v, len(wav_bytes), time.time() - t0, out.name)
        return out


# ---------- CLI ----------

def _list_voices():
    if not shutil.which(SAY_BIN):
        print("say not found")
        return
    out = subprocess.run([SAY_BIN, "-v", "?"], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if "zh_" in line or "en_US" in line:
            print(line)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="socsay.voice")
    ap.add_argument("text", nargs="*", help="要朗读的文本（不传则需 --file）")
    ap.add_argument("--engine", choices=["mimo", "system"], default="mimo")
    ap.add_argument("--voice", default=DEFAULT_VOICE,
                    help="system 后端的 say -v 名字")
    ap.add_argument("--rate", type=int, default=DEFAULT_RATE)
    ap.add_argument("--mimo-voice", default=MIMO_TTS_VOICE,
                    help="mimo 预置音色：冰糖/茆莉/苏打/白桦/Mia/Chloe/Milo/Dean")
    ap.add_argument("--mimo-style", default=None,
                    help="mimo 风格提示（user message）")
    ap.add_argument("--file", help="改为播放音频文件")
    ap.add_argument("--volume", type=float, default=None,
                    help="afplay 音量 0~2，默认 1")
    ap.add_argument("--list-voices", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.list_voices:
        _list_voices()
        return 0

    # 加载 .env（简易实现）
    _load_dotenv()

    v = Voice(default_voice=args.voice, default_rate=args.rate,
              engine=args.engine, mimo_voice=args.mimo_voice,
              mimo_style=args.mimo_style)
    v.start()

    def shutdown(*_):
        v.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    if args.file:
        v.play_file(args.file, volume=args.volume, tag="cli")
    else:
        text = " ".join(args.text).strip()
        if not text:
            ap.error("需要传 text 或 --file")
        v.say(text, tag="cli")

    # 等队列放完（包括正在合成的 in-flight 项）
    time.sleep(0.2)
    while v.is_speaking() or v._q.qsize() > 0 or v._current is not None:
        time.sleep(0.1)
    v.stop(drain=True)
    return 0


def _load_dotenv(path: str = ".env"):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, val = line.partition("=")
        k = k.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(k, val)


if __name__ == "__main__":
    sys.exit(main())
