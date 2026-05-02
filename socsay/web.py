"""网页 UI / MJPEG / SSE 日志 模块。

提供：
  - install_log_handler(): 把内存环形缓冲挂到 root logger，所有 log 实时进队列
  - LogBus.subscribe(): 给 SSE 用的订阅器
  - INDEX_HTML: 单页应用，左视频右快照下日志，调用 /state /expression/state /attention/state
  - mjpeg_handler / sse_handler: 给 BaseHTTPRequestHandler 调用
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from queue import Queue, Empty, Full
from pathlib import Path
from typing import Iterable, Optional


# ---------- log bus ----------

class LogBus:
    """单进程内的日志广播。subscribe 得到一个 Queue，put 满即丢老的。"""

    def __init__(self, history: int = 200):
        self._lock = threading.Lock()
        self._history: deque[str] = deque(maxlen=history)
        self._subs: list[Queue] = []

    def push(self, line: str):
        with self._lock:
            self._history.append(line)
            dead = []
            for q in self._subs:
                try:
                    q.put_nowait(line)
                except Full:
                    dead.append(q)
            for q in dead:
                try:
                    self._subs.remove(q)
                except ValueError:
                    pass

    def subscribe(self) -> Queue:
        q: Queue = Queue(maxsize=500)
        with self._lock:
            for line in list(self._history):
                try:
                    q.put_nowait(line)
                except Full:
                    break
            self._subs.append(q)
        return q

    def unsubscribe(self, q: Queue):
        with self._lock:
            try:
                self._subs.remove(q)
            except ValueError:
                pass


class _BusHandler(logging.Handler):
    def __init__(self, bus: LogBus):
        super().__init__()
        self.bus = bus
        self.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%H:%M:%S",
        ))

    def emit(self, record):
        try:
            msg = self.format(record)
        except Exception:
            return
        self.bus.push(msg)


def install_log_bus(level: int = logging.INFO) -> LogBus:
    bus = LogBus()
    h = _BusHandler(bus)
    h.setLevel(level)
    logging.getLogger().addHandler(h)
    return bus


# ---------- MJPEG ----------

def write_mjpeg(handler, get_jpeg_fn, fps: float = 15.0):
    """把 ffmpeg 持续推出的 jpeg 当 multipart 流出去。

    get_jpeg_fn(): -> (bytes|None, ts:float)；阻塞或快速返回都行。
    """
    boundary = "frame"
    handler.send_response(200)
    handler.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
    handler.send_header("Pragma", "no-cache")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Content-Type",
                        f"multipart/x-mixed-replace; boundary={boundary}")
    handler.end_headers()
    interval = 1.0 / max(1.0, fps)
    last_ts = -1.0
    try:
        while True:
            jpeg, ts = get_jpeg_fn()
            if jpeg is None:
                time.sleep(0.05)
                continue
            if ts == last_ts:
                time.sleep(interval / 2)
                continue
            last_ts = ts
            try:
                handler.wfile.write(b"--" + boundary.encode() + b"\r\n")
                handler.wfile.write(b"Content-Type: image/jpeg\r\n")
                handler.wfile.write(
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                )
                handler.wfile.write(jpeg)
                handler.wfile.write(b"\r\n")
                handler.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            time.sleep(interval)
    except Exception:
        # 客户端断开
        return


# ---------- SSE ----------

def write_sse(handler, bus: LogBus, max_idle_sec: float = 30.0):
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.end_headers()
    q = bus.subscribe()
    last_send = time.monotonic()
    try:
        while True:
            try:
                line = q.get(timeout=2.0)
                payload = f"data: {line}\n\n".encode("utf-8")
                handler.wfile.write(payload)
                handler.wfile.flush()
                last_send = time.monotonic()
            except Empty:
                # 心跳，避免代理断开
                if time.monotonic() - last_send > max_idle_sec:
                    return
                try:
                    handler.wfile.write(b": ping\n\n")
                    handler.wfile.flush()
                except Exception:
                    return
            except (BrokenPipeError, ConnectionResetError):
                return
    finally:
        bus.unsubscribe(q)


# ---------- HTML ----------

INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Socratic Companion · Insta360 Link 2</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:    #f3efe7;
    --ink:   #111111;
    --dim:   #6b6b66;
    --line:  #1a1a1a;
    --rule:  rgba(17,17,17,0.18);
    --panel: #ffffff;
    --warn:  #b34a18;
    --ok:    #2c6e3a;
    --bad:   #b3001b;
    --frame: transparent;
    --frame-color: #2a9d8f; /* 由 JS 根据 intensity 实时改 */
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; }
  body {
    background: var(--bg); color: var(--ink);
    font: 14px/1.55 "Inter", -apple-system, "PingFang SC",
                    "Helvetica Neue", Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    transition: background 0.6s ease;
  }
  .mono { font-family: "JetBrains Mono", "SF Mono", Consolas, Menlo, monospace; }

  /* 整页边框：跨越 alert/normal。颜色由 --frame-color 提供，厚度/脉冲由 mode 提供 */
  .edge {
    position: fixed; inset: 0; pointer-events: none; z-index: 9999;
    box-shadow: inset 0 0 0 6px var(--frame-color);
    transition: box-shadow 0.4s ease;
    opacity: 0.85;
  }
  body[data-mode="alert"] {
    --frame: var(--frame-color);
  }
  body[data-mode="alert"] .edge {
    box-shadow:
      inset 0 0 0 12px var(--frame-color),
      inset 0 0 80px 0 rgba(193,49,15,0.45);
    animation: pulse 1.4s ease-in-out infinite;
    opacity: 1;
  }
  body[data-mode="watching"] .edge {
    box-shadow: inset 0 0 0 8px var(--frame-color);
  }
  body[data-mode="idle"] .edge {
    opacity: 0.55;
  }

  @keyframes pulse {
    0%,100% { box-shadow: inset 0 0 0 12px var(--frame-color), inset 0 0 80px 0 rgba(193,49,15,0.40); }
    50%     { box-shadow: inset 0 0 0 16px var(--frame-color), inset 0 0 110px 0 rgba(193,49,15,0.65); }
  }

  header {
    display: flex; align-items: baseline; gap: 18px;
    padding: 22px 32px 18px;
    border-bottom: 1px solid var(--line);
    background: transparent;
  }
  header .brand {
    font-family: "JetBrains Mono", monospace;
    font-size: 11px; letter-spacing: 0.18em;
    color: var(--dim); text-transform: uppercase;
  }
  header h1 {
    font-size: 22px; font-weight: 600; letter-spacing: -0.01em;
  }
  header h1 .accent { color: var(--dim); font-weight: 500; }
  header .grow { flex: 1; }
  header .meta {
    font-family: "JetBrains Mono", monospace;
    font-size: 11px; color: var(--dim); letter-spacing: 0.05em;
  }
  header .dot {
    display: inline-block; width: 7px; height: 7px;
    background: var(--ok); border-radius: 50%;
    margin-right: 6px; transform: translateY(-1px);
  }
  #toggleBtn {
    padding: 8px 22px; border: 2px solid var(--ink); border-radius: 6px;
    background: var(--ink); color: #fff;
    font: 600 13px/1 "Inter", sans-serif; letter-spacing: 0.08em;
    cursor: pointer; transition: all 0.25s ease; text-transform: uppercase;
  }
  #toggleBtn:hover { background: #333; }
  #toggleBtn.running { background: var(--bad); border-color: var(--bad); }
  #toggleBtn.running:hover { background: #8b0015; }

  #enterLock {
    display: inline-block; padding: 10px 24px; border-radius: 8px;
    font: 700 18px/1 "JetBrains Mono", monospace;
    letter-spacing: 0.08em; text-transform: uppercase;
    transition: all 0.3s ease; margin-bottom: 16px;
  }
  #enterLock.locked {
    background: var(--bad); color: #fff;
    box-shadow: 0 0 20px rgba(179,0,27,0.35);
    animation: lockPulse 1.2s ease-in-out infinite;
  }
  #enterLock.unlocked {
    background: var(--ok); color: #fff;
    box-shadow: 0 0 12px rgba(44,110,58,0.25);
  }
  @keyframes lockPulse {
    0%,100% { opacity: 1; }
    50%     { opacity: 0.7; }
  }

  /* ---------------- BIG EMOTION BAND ---------------- */
  #bigEmo {
    padding: 28px 32px 24px;
    border-bottom: 1px solid var(--rule);
    background: transparent;
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 24px; align-items: center;
  }
  #bigEmo .lab {
    font-family: "JetBrains Mono", monospace;
    font-size: 11px; letter-spacing: 0.2em;
    color: var(--dim); text-transform: uppercase;
    margin-bottom: 6px;
  }
  #bigEmo .word {
    font-family: "Inter", sans-serif;
    font-weight: 800;
    font-size: clamp(60px, 11vw, 160px);
    line-height: 0.95;
    letter-spacing: -0.03em;
    color: #b6b6b1;
    transition: color 0.4s ease;
    word-break: break-word;
  }
  #bigEmo .meter {
    width: 220px;
    text-align: right;
  }
  #bigEmo .meter .num {
    font-family: "JetBrains Mono", monospace;
    font-size: 42px; font-weight: 700;
    color: var(--ink); letter-spacing: -0.02em;
  }
  #bigEmo .meter .bar {
    margin-top: 10px; height: 10px;
    background: rgba(17,17,17,0.08);
    border: 1px solid var(--rule);
    overflow: hidden;
  }
  #bigEmo .meter .bar > span {
    display: block; height: 100%; width: 0%;
    background: var(--ink);
    transition: width 0.45s ease, background 0.4s ease;
  }
  #bigEmo .meter .sub {
    font-family: "JetBrains Mono", monospace;
    font-size: 11px; color: var(--dim);
    letter-spacing: 0.06em; margin-top: 6px;
  }

  /* ---------------- MAIN GRID ---------------- */
  main {
    display: grid;
    grid-template-columns: 1.4fr 1fr;
    grid-template-rows: minmax(360px, auto) auto auto;
    gap: 0;
    border-bottom: 1px solid var(--line);
  }
  section {
    padding: 22px 28px;
    border-right: 1px solid var(--rule);
    border-bottom: 1px solid var(--rule);
    background: transparent;
    display: flex; flex-direction: column;
    min-width: 0;
  }
  section:nth-child(2n) { border-right: none; }
  .num {
    font-family: "JetBrains Mono", monospace;
    font-size: 11px; letter-spacing: 0.18em; color: var(--dim);
  }
  h2 {
    margin-top: 4px; margin-bottom: 14px;
    font-size: 15px; font-weight: 600; letter-spacing: 0.02em;
    text-transform: uppercase;
  }

  /* live */
  #live { grid-column: 1; grid-row: 1; }
  #live .frame {
    flex: 1; min-height: 320px; background: #0a0a0a;
    border: 1px solid var(--line); position: relative;
    overflow: hidden;
  }
  #live .frame img { width: 100%; height: 100%; object-fit: contain; display: block; }
  #live .badge {
    position: absolute; top: 10px; left: 10px;
    background: rgba(0,0,0,0.55); color: #fff;
    font-family: "JetBrains Mono", monospace;
    font-size: 11px; padding: 4px 8px; letter-spacing: 0.05em;
  }

  /* face */
  #face { grid-column: 2; grid-row: 1; }
  #face .frame {
    flex: 0 0 auto; aspect-ratio: 1/1; max-height: 280px;
    background: #efe9dc; border: 1px solid var(--line);
    display: flex; align-items: center; justify-content: center;
    overflow: hidden; margin-bottom: 14px;
  }
  #face .frame img { width: 100%; height: 100%; object-fit: cover; }
  #face .frame .ph { color: var(--dim); font-family: "JetBrains Mono", monospace; font-size: 11px; }

  /* 大字描述行 */
  .speech {
    margin-top: 10px; padding: 16px 18px;
    border-left: 3px solid var(--ink);
    background: var(--panel);
    font-size: 22px; line-height: 1.5; font-weight: 500;
    letter-spacing: 0.005em;
  }
  .speech.empty { color: var(--dim); font-style: italic; border-left-color: var(--rule); font-size: 14px; font-weight: 400; }
  .observe {
    margin-top: 10px; padding: 12px 14px;
    background: rgba(17,17,17,0.04);
    font-size: 18px; line-height: 1.5;
    color: #2c2c28;
  }
  .observe .lab {
    font-family: "JetBrains Mono", monospace;
    font-size: 10px; letter-spacing: 0.18em; color: var(--dim);
    text-transform: uppercase; display: block; margin-bottom: 4px;
  }

  .kv { display: grid; grid-template-columns: 90px 1fr;
        gap: 4px 12px; font-size: 12px; margin-top: 12px; }
  .kv dt { color: var(--dim); font-family: "JetBrains Mono", monospace;
           font-size: 10px; letter-spacing: 0.06em; text-transform: uppercase;
           padding-top: 2px; }
  .kv dd { font-weight: 500; }

  /* history grid (上提到第二行) */
  #history { grid-column: 1 / span 2; grid-row: 2; }
  .gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
    gap: 10px;
  }
  .tile {
    position: relative; aspect-ratio: 1/1;
    background: #efe9dc; border: 1px solid var(--rule);
    overflow: hidden; cursor: pointer;
  }
  .tile img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .tile .meta {
    position: absolute; left: 0; right: 0; bottom: 0;
    background: linear-gradient(0deg, rgba(0,0,0,0.7), rgba(0,0,0,0));
    color: #fff; font-family: "JetBrains Mono", monospace;
    font-size: 10px; padding: 16px 6px 4px; letter-spacing: 0.04em;
    display: flex; justify-content: space-between;
  }
  .tile.active { outline: 2px solid var(--ink); outline-offset: -2px; }

  /* signals strip */
  #status { grid-column: 1 / span 2; grid-row: 3; padding: 18px 28px; border-bottom: none; }
  #status .strip {
    display: grid; grid-template-columns: repeat(6, 1fr);
    gap: 16px;
  }
  #status .cell { border-left: 1px solid var(--rule); padding-left: 12px; }
  #status .cell:first-child { border-left: 1px solid var(--ink); }
  #status .lab {
    font-family: "JetBrains Mono", monospace;
    font-size: 10px; letter-spacing: 0.1em;
    color: var(--dim); text-transform: uppercase; margin-bottom: 4px;
  }
  #status .val {
    font-family: "JetBrains Mono", monospace;
    font-size: 14px; font-weight: 500; color: var(--ink);
  }
  #status .val.warn { color: var(--warn); }
  #status .val.ok   { color: var(--ok); }
  #status .val.bad  { color: var(--bad); }

  /* logs */
  #logs { padding: 22px 28px 28px; }
  #logs .body {
    height: 200px; overflow: auto;
    background: #fbfaf7;
    border: 1px solid var(--rule);
    padding: 10px 12px;
    font: 12px/1.55 "JetBrains Mono", monospace;
    color: #2a2a26;
  }
  #logs .body .line { white-space: pre-wrap; word-break: break-all; }
  #logs .body .line.WARNING { color: var(--warn); }
  #logs .body .line.ERROR   { color: var(--bad); }
  #logs .body .line.DEBUG   { color: #8a8a83; }
  #logs .body .line .t { color: #8a8a83; margin-right: 8px; }

  footer {
    padding: 14px 28px; font-family: "JetBrains Mono", monospace;
    font-size: 11px; color: var(--dim); letter-spacing: 0.05em;
    display: flex; gap: 18px; flex-wrap: wrap; align-items: center;
  }
  footer a { color: var(--ink); text-decoration: none; border-bottom: 1px solid var(--rule); }
  footer a:hover { border-color: var(--ink); }

  @media (max-width: 960px) {
    main { grid-template-columns: 1fr; }
    section { border-right: none; }
    #live, #face { grid-column: 1; grid-row: auto; }
    #history { grid-column: 1; }
    #status { grid-column: 1; }
    #status .strip { grid-template-columns: repeat(3, 1fr); }
    #bigEmo { grid-template-columns: 1fr; }
    #bigEmo .meter { width: 100%; text-align: left; }
  }
</style>
</head>
<body data-mode="idle">
<div class="edge"></div>

<header>
  <span class="brand">CASE / 04 — INK</span>
  <h1>Socratic Companion <span class="accent">// face-aware reflective dialogue</span></h1>
  <span class="grow"></span>
  <button id="toggleBtn" class="running" onclick="toggleExpr()">■ 停止监测</button>
  <span class="meta"><span class="dot" id="hb"></span>INSTA360 LINK 2 · MIMO V2.5 · MIMO TTS</span>
</header>

<!-- ============== BIG EMOTION ============== -->
<div id="bigEmo">
  <div>
    <div class="lab">last emotion · 当前情绪</div>
    <div class="word" id="bigEmoWord">—</div>
  </div>
  <div class="meter">
    <div class="lab">intensity</div>
    <div class="num" id="bigEmoInt">0.00</div>
    <div class="bar"><span id="bigEmoBar"></span></div>
    <div class="sub" id="bigEmoSub">awaiting first detection…</div>
  </div>
</div>

<main>

  <section id="live">
    <div class="num">01 · LIVE FEED</div>
    <h2>Camera</h2>
    <div class="frame">
      <img id="liveImg" alt="live">
      <div class="badge mono" id="liveBadge">— · — fps</div>
    </div>
  </section>

  <section id="face">
    <div class="num">02 · THINKING DEPTH</div>
    <h2>思考深度</h2>
    <div id="enterLock" class="unlocked">⏎ ENTER 已解锁</div>
    <div class="speech" id="speechBox" style="font-size:15px;">— Socratic line will appear here —</div>
    <div class="observe" id="observeBox" style="display:none">
      <span class="lab">observe</span>
      <span id="observeText">—</span>
    </div>
    <dl class="kv">
      <dt>近1min 平均紧张度</dt> <dd id="eAvg" class="mono" style="font-size:28px;font-weight:700;">0.00</dd>
      <dt>近1min 最高紧张度</dt> <dd id="eMax" class="mono" style="font-size:28px;font-weight:700;">0.00</dd>
      <dt>采样数</dt>            <dd id="eSamples" class="mono">0</dd>
      <dt>attention</dt>         <dd id="eAtt">—</dd>
      <dt>gaze</dt>              <dd id="eGaze">—</dd>
      <dt>latency</dt>           <dd id="eLat" class="mono">—</dd>
    </dl>
  </section>

  <section id="history">
    <div class="num">04 · HISTORY</div>
    <h2>Past Snapshots</h2>
    <div class="gallery" id="gallery">
      <div class="ph mono" style="color:var(--dim); padding:30px 4px;">empty</div>
    </div>
  </section>

  <section id="status">
    <div class="num">03 · SIGNALS</div>
    <h2>Live State</h2>
    <div class="strip">
      <div class="cell"><div class="lab">behavior</div><div class="val" id="mState">—</div></div>
      <div class="cell"><div class="lab">face conf</div><div class="val" id="mConf">—</div></div>
      <div class="cell"><div class="lab">PTZ pan/tilt</div><div class="val" id="mPos">—</div></div>
      <div class="cell"><div class="lab">front app</div><div class="val" id="mApp">—</div></div>
      <div class="cell"><div class="lab">intervene</div><div class="val" id="mInt">—</div></div>
      <div class="cell"><div class="lab">↵ blocked</div><div class="val" id="mBlk">0</div></div>
    </div>
  </section>

</main>

<section id="logs">
  <div class="num">05 · SYSTEM LOG · SSE</div>
  <h2>Runtime</h2>
  <div class="body" id="logBox"></div>
</section>

<footer>
  <span>© 2026 socsay</span>
  <a href="/state" target="_blank">/state</a>
  <a href="/expression/state" target="_blank">/expression/state</a>
  <a href="/expression/history" target="_blank">/expression/history</a>
  <a href="/attention/state" target="_blank">/attention/state</a>
  <a href="/voice/state" target="_blank">/voice/state</a>
  <span class="grow"></span>
  <span>BUILT IN THE OPEN · v0.3</span>
</footer>

<script>
let exprRunning = true;
async function toggleExpr() {
  const btn = document.getElementById('toggleBtn');
  const endpoint = exprRunning ? '/expression/stop' : '/expression/start';
  try {
    await fetch(endpoint, {method: 'POST'});
    exprRunning = !exprRunning;
    btn.textContent = exprRunning ? '■ 停止监测' : '▶ 开始监测';
    btn.className = exprRunning ? 'running' : '';
  } catch(e) { console.error(e); }
}
(() => {
  // ---------- emotion classification (frontend-only) ----------
  // 紧张/思考/蹙眉系列 → 橙红
  const TENSE = new Set([
    "confused","perplexed","puzzled","frustrated","stressed","anxious",
    "nervous","tense","worried","overwhelmed","stuck","blocked","lost",
    "blanking","thinking","deep_thinking","thinking_hard","reasoning",
    "analyzing","evaluating","problem_solving","reflective","contemplative",
    "hesitant_but_engaged","engaged_but_uncertain","cognitive_load_high",
    "mentally_fatigued","brow_furrowed","frowning","jaw_tense","mouth_tight",
    "lip_pressed","squinting","hesitating","uncertain","unsure","ambivalent",
    "skeptical","doubtful","suspicious","resistant","defensive",
    "needs_deeper_questioning","concerned","disappointed","impatient",
    "irritated","annoyed","angry","fear","startled","shocked","sad",
    "melancholic","ashamed","guilty","embarrassed","tired","sleepy",
    "physically_tired","restless","focused_but_tired","interrupted",
    "recovering_focus","overloaded","needs_prompting","not_ready",
    "rapid_blinking","disengaged","zoned_out","daydreaming","avoidant",
    "avoidance","resisting","withdrawing","leaning_back","shaking_head",
    "forced_smile","misunderstanding","disagreeing","rejecting"
  ]);

  // 把 emotion + intensity 映射成颜色 hex
  // 主轴是 intensity：从「蓝绿(冷静)」 → 「橙红(紧张)」线性渐变
  // emotion 是否在 TENSE 集只用来在 intensity 接近时偏向暖色
  // 返回 {color, hue} 给整页主题色用
  function intensityColor(inten, isTense) {
    let t = Math.max(0, Math.min(1, +inten || 0));
    // 紧张系再加 0.15 偏移，但不超过 1
    if (isTense) t = Math.min(1, t + 0.15);
    // 0.0 → teal/cyan #2a9d8f
    // 0.5 → 中性暖黄 #d6a23a
    // 1.0 → 强烈橙红 #c1310f
    let r, g, b;
    if (t < 0.5) {
      const k = t / 0.5;            // 0..1
      r = Math.round( 42 + (214- 42) * k);
      g = Math.round(157 + (162-157) * k);
      b = Math.round(143 + ( 58-143) * k);
    } else {
      const k = (t - 0.5) / 0.5;     // 0..1
      r = Math.round(214 + (193-214) * k);
      g = Math.round(162 + ( 49-162) * k);
      b = Math.round( 58 + ( 15- 58) * k);
    }
    return `rgb(${r},${g},${b})`;
  }
  // 同色系的浅淡背景（给 body 用）
  function intensityTint(inten, isTense) {
    let t = Math.max(0, Math.min(1, +inten || 0));
    if (isTense) t = Math.min(1, t + 0.15);
    if (t < 0.5) {
      const k = t / 0.5;
      // 蓝绿淡 #e6f1ee → 米白 #f3efe7
      const r = Math.round(230 + (243-230) * k);
      const g = Math.round(241 + (239-241) * k);
      const b = Math.round(238 + (231-238) * k);
      return `rgb(${r},${g},${b})`;
    } else {
      const k = (t - 0.5) / 0.5;
      // 米白 #f3efe7 → 浅橙 #f7d9c4
      const r = Math.round(243 + (247-243) * k);
      const g = Math.round(239 + (217-239) * k);
      const b = Math.round(231 + (196-231) * k);
      return `rgb(${r},${g},${b})`;
    }
  }
  function emoColor(emo, inten) {
    return intensityColor(inten, TENSE.has((emo || "").toLowerCase()));
  }
  function isTense(emo) { return TENSE.has((emo || "").toLowerCase()); }

  // 当前最新 intensity，给 pollState 共用
  let curIntensity = 0;
  let curIsTense   = false;

  // ---------- live mjpeg ----------
  const liveImg = document.getElementById('liveImg');
  liveImg.src = '/stream.mjpg?ts=' + Date.now();

  const speechBox   = document.getElementById('speechBox');
  const observeBox  = document.getElementById('observeBox');
  const observeText = document.getElementById('observeText');
  const bigWord     = document.getElementById('bigEmoWord');
  const bigInt      = document.getElementById('bigEmoInt');
  const bigBar      = document.getElementById('bigEmoBar');
  const bigSub      = document.getElementById('bigEmoSub');

  function applyTheme() {
    const col  = intensityColor(curIntensity, curIsTense);
    const tint = intensityTint(curIntensity, curIsTense);
    document.documentElement.style.setProperty('--bg',    tint);
    document.documentElement.style.setProperty('--frame-color', col);
    // bar / word / edge 全部同色
    bigBar.style.background = col;
    bigWord.style.color = col;
  }

  function paintLatest(item) {
    if (!item) return;
    const emo = item.emotion || '—';
    const inten = +item.intensity || 0;

    bigWord.textContent = emo;
    bigInt.textContent  = inten.toFixed(2);
    bigBar.style.width  = (inten * 100).toFixed(0) + '%';
    bigSub.textContent  = `attention=${item.attention || '—'} · gaze=${item.gaze || '—'} · ${item.latency_ms || 0}ms`;

    document.getElementById('eAtt').textContent  = item.attention || '—';
    document.getElementById('eGaze').textContent = item.gaze || '—';
    document.getElementById('eLat').textContent  = (item.latency_ms || 0) + ' ms';

    if (item.speech) {
      speechBox.textContent = '“ ' + item.speech + ' ”';
      speechBox.classList.remove('empty');
    }
    if (item.comment) {
      observeText.textContent = item.comment;
      observeBox.style.display = 'block';
    }

    curIntensity = inten;
    curIsTense   = isTense(emo);
    applyTheme();
  }

  // ---------- history gallery ----------
  // history[0] 是最新的；Face Snapshot 直接复用第 0 项
  const gallery = document.getElementById('gallery');
  let lastTopTs = 0;
  function pollHistory() {
    fetch('/expression/history').then(r => r.json()).then(j => {
      const items = j.items || [];
      if (!items.length) return;
      // 用最新的去覆盖 BIG/FACE/SPEECH
      const top = items[0];
      if (top.ts !== lastTopTs) {
        paintLatest(top);
        lastTopTs = top.ts;
      }
      // 重渲染 gallery
      gallery.innerHTML = '';
      items.forEach((it, idx) => {
        if (!it.face_url) return;
        const tile = document.createElement('div');
        tile.className = 'tile' + (idx === 0 ? ' active' : '');
        const t = new Date(it.ts * 1000);
        const hh = String(t.getHours()).padStart(2,'0');
        const mm = String(t.getMinutes()).padStart(2,'0');
        const ss = String(t.getSeconds()).padStart(2,'0');
        tile.innerHTML = `
          <img src="${it.face_url}" alt="">
          <div class="meta">
            <span>${hh}:${mm}:${ss}</span>
            <span>${it.emotion}</span>
          </div>`;
        tile.title = (it.speech || it.comment || '');
        tile.onclick = () => paintLatest(it);
        gallery.appendChild(tile);
      });
    }).catch(()=>{});
  }
  setInterval(pollHistory, 2500);
  setTimeout(pollHistory, 600);

  // ---------- main / attention state, drives EDGE ----------
  function pollState() {
    fetch('/state').then(r => r.json()).then(j => {
      document.getElementById('mState').textContent = j.state || '—';
      const c = j.face ? j.face.conf : 0;
      document.getElementById('mConf').textContent = (c||0).toFixed(2);
      if (j.pos) {
        document.getElementById('mPos').textContent = `${j.pos[0]} / ${j.pos[1]}`;
      }
    }).catch(()=>{});

    fetch('/attention/state').then(r => r.json()).then(j => {
      const app = j.frontmost_app || '—';
      document.getElementById('mApp').textContent = app;
      const p = document.getElementById('mInt');
      if (j.should_intervene) {
        p.textContent = 'YES · ' + (j.intervene_reason || '');
        p.className = 'val bad';
      } else {
        p.textContent = j.intervene_reason || 'idle';
        p.className = 'val';
      }
      document.getElementById('mBlk').textContent =
        (j.suppressor && j.suppressor.block_count) || 0;

      // 更新回车锁定状态
      const lockEl = document.getElementById('enterLock');
      if (j.is_target_app && j.should_intervene) {
        lockEl.className = 'locked';
        lockEl.textContent = '⏎ ENTER 已锁定';
      } else {
        lockEl.className = 'unlocked';
        lockEl.textContent = '⏎ ENTER 已解锁';
      }

      // 边沿染色逻辑：
      //   非 VSCode → idle (透明)
      //   VSCode + 紧张/应当干预 → alert (橙红，禁止回车的视觉)
      //   VSCode + 平静 → watching (绿)
      let mode = 'idle';
      if (j.is_target_app) {
        mode = j.should_intervene ? 'alert' : 'watching';
      }
      // 也参考一下当前 emotion 防止 attention 短期没刷新
      if (j.is_target_app && bigWord.textContent && isTense(bigWord.textContent)) {
        mode = 'alert';
      }
      document.body.dataset.mode = mode;
    }).catch(()=>{});
  }
  setInterval(pollState, 1500);

  // ---------- thinking depth ----------
  const eAvg     = document.getElementById('eAvg');
  const eMax     = document.getElementById('eMax');
  const eSamples = document.getElementById('eSamples');
  function pollDepth() {
    fetch('/expression/state').then(r => r.json()).then(j => {
      if (!j.enabled) return;
      eAvg.textContent     = (j.recent_avg || 0).toFixed(2);
      eMax.textContent     = (j.recent_max || 0).toFixed(2);
      eSamples.textContent = j.recent_samples || 0;
      // 如果 paused 同步按钮状态
      const btn = document.getElementById('toggleBtn');
      if (j.paused && exprRunning) {
        exprRunning = false;
        btn.textContent = '▶ 开始监测';
        btn.className = '';
      } else if (!j.paused && !exprRunning) {
        exprRunning = true;
        btn.textContent = '■ 停止监测';
        btn.className = 'running';
      }
    }).catch(() => {});
  }
  pollDepth();
  setInterval(pollDepth, 2000);
  pollState();

  // ---------- log SSE ----------
  const logBox = document.getElementById('logBox');
  const hb = document.getElementById('hb');
  function appendLog(line) {
    const m = line.match(/^(\d{2}:\d{2}:\d{2}) (\w+) (\S+) (.*)$/);
    let cls = 'INFO';
    if (m) cls = m[2];
    const div = document.createElement('div');
    div.className = 'line ' + cls;
    if (m) {
      div.innerHTML = `<span class="t">${m[1]}</span>${m[2]} ${m[3]} ${
        m[4].replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))
      }`;
    } else {
      div.textContent = line;
    }
    logBox.appendChild(div);
    if (logBox.children.length > 400) logBox.removeChild(logBox.firstChild);
    logBox.scrollTop = logBox.scrollHeight;
  }
  function connectSSE() {
    const es = new EventSource('/logs');
    es.onmessage = (e) => {
      hb.style.background = 'var(--ok)';
      appendLog(e.data);
    };
    es.onerror = () => {
      hb.style.background = 'var(--bad)';
      es.close();
      setTimeout(connectSSE, 2000);
    };
  }
  connectSSE();
})();
</script>

</body>
</html>
"""
