"""注意力 / 干预模块。

职责：
  1) 通过 osascript 周期性查询 macOS 前台 App 名称
  2) 暴露 should_intervene()：当前台 = VSCode 且最近一次表情 = 困惑/蹙眉
  3) 启动 EnterSuppressor：用 pynput darwin_intercept 在「应当干预」时吞掉 Return/KP_Enter
  4) 提供 30 句苏格拉底式「再想想」催问语，random.choice 给 voice 朗读

需要权限（macOS）：
  - 系统设置 → 隐私与安全性 → 辅助功能 → 把运行 python 的终端勾上
  - 系统设置 → 隐私与安全性 → 输入监控 → 同上
"""
from __future__ import annotations

import logging
import random
import re
import subprocess
import threading
import time
from typing import Callable, Optional

log = logging.getLogger("socsay.attention")

VSCODE_NAMES = {"Code", "Visual Studio Code", "Code - Insiders", "Electron", "Cursor"}

# 30 句催问/挑刺语，全部带苏格拉底口吻
PROMPTS = [
    "等一下，你确定这就是答案吗？",
    "再读一遍代码，你真理解每一行了吗？",
    "如果让你现在解释给别人听，你能说清吗？",
    "你刚才停顿了，是哪里没想透？",
    "这是最简单的写法，还是你想到的第一种？",
    "假设这段代码上线了，会出什么问题？",
    "你确认所有边界都覆盖了吗？",
    "为什么是这个变量名？它准确吗？",
    "如果删掉这一行，会怎样？",
    "这块逻辑能写得更短吗？",
    "如果半年后再看，你还能读懂吗？",
    "你考虑过失败路径吗？",
    "这个抽象，真的必要吗？",
    "你是在解决问题，还是在绕开它？",
    "这一步的假设是什么，写下来过吗?",
    "你真的需要现在回车吗？",
    "再想想，这是最优解吗？",
    "如果有 bug，你会怎么定位？",
    "这段代码的不变式是什么？",
    "你能用一句话概括它在做什么吗？",
    "你是在写代码，还是在等代码替你思考？",
    "退一步看：这是真问题，还是你在自寻烦恼？",
    "再问自己一次：为什么是这样？",
    "如果约束改了，这个方案还成立吗？",
    "你刚才皱眉了，那一刻你在想什么？",
    "别急着回车，先把它说清楚。",
    "这真是你想要的，还是顺手就这样了？",
    "你写的，真的是你想表达的吗？",
    "深呼吸，再问一句：为什么？",
    "再来一次，从头说一遍逻辑。",
]

CONFUSED_KEYWORDS = ("困惑", "蹙眉", "皱眉", "迟疑", "犹豫", "茫然", "微蹙", "紧闭", "专注", "沉思", "思考")

# 30 句很短的"再想想"式提示，专门用于回车被吞时口头打断
SHORT_NUDGES = [
    "再想想。",
    "慢下来。",
    "等会儿呗。",
    "别急。",
    "缓一缓。",
    "再读一遍。",
    "停一下。",
    "深呼吸。",
    "再确认下。",
    "想清楚再走。",
    "稳住。",
    "等等。",
    "回头看一眼。",
    "先别回车。",
    "再多看一秒。",
    "别凑合。",
    "再过一遍。",
    "再问一次为什么。",
    "再走半步。",
    "想透了再说。",
    "别糊弄自己。",
    "先理一理。",
    "缓口气。",
    "你确定吗？",
    "真的吗？",
    "再核一下。",
    "停。",
    "嘘，再想想。",
    "别这么快。",
    "慢一拍。",
]


def _frontmost_app() -> Optional[str]:
    """返回 macOS 前台进程名（如 'Code'）。失败返回 None。"""
    try:
        r = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to get name of first '
             'application process whose frontmost is true'],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode != 0:
            return None
        return (r.stdout or "").strip() or None
    except Exception:
        return None


class Attention:
    """周期性查前台 App，并暴露 is_active_target() / is_confused_recent()。"""

    def __init__(self, expression=None,
                 target_apps: set[str] = VSCODE_NAMES,
                 poll_sec: float = 1.5,
                 confused_window_sec: float = 12.0):
        self.expression = expression
        self.target_apps = set(target_apps)
        self.poll_sec = poll_sec
        self.confused_window_sec = confused_window_sec
        self._app: Optional[str] = None
        self._app_ts = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="attention", daemon=True
        )
        self._thread.start()
        log.info("attention loop start, target_apps=%s", self.target_apps)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self):
        while not self._stop.is_set():
            name = _frontmost_app()
            if name and name != self._app:
                log.info("frontmost app: %s -> %s", self._app, name)
            self._app = name
            self._app_ts = time.time()
            self._stop.wait(self.poll_sec)

    # ------- queries -------
    def frontmost(self) -> Optional[str]:
        return self._app

    def is_active_target(self) -> bool:
        return (self._app or "") in self.target_apps

    def is_confused_recent(self) -> tuple[bool, str]:
        """最近 N 秒内最新表情为 confused 或 comment 包含蹙眉/皱眉关键词。"""
        if self.expression is None:
            return False, ""
        last = self.expression.latest()
        if last is None:
            return False, ""
        if time.time() - last.ts > self.confused_window_sec:
            return False, ""
        emo = (last.emotion or "").lower()
        com = last.comment or ""
        if emo == "confused":
            return True, f"emotion={emo}"
        for kw in CONFUSED_KEYWORDS:
            if kw in com:
                return True, f"comment={kw}"
        return False, ""

    def is_tense_recent(self,
                        window_sec: float = 60.0,
                        avg_threshold: float = 0.20,
                        max_threshold: float = 0.30,
                        ) -> tuple[bool, str]:
        """近 window_sec 内 intensity 平均 ≥avg 或 单次峰值 ≥max → 紧张档。"""
        if self.expression is None:
            return False, ""
        try:
            avg, n, mx = self.expression.recent_intensity(window_sec)
        except Exception:
            return False, ""
        if n < 2:
            return False, f"too_few_samples={n}"
        if avg >= avg_threshold or mx >= max_threshold:
            return True, f"intensity avg={avg:.2f} max={mx:.2f} n={n}"
        return False, f"calm avg={avg:.2f} max={mx:.2f} n={n}"

    def should_intervene(self) -> tuple[bool, str]:
        if not self.is_active_target():
            return False, "not_target_app"
        ok, why = self.is_confused_recent()
        if ok:
            return True, why
        ok2, why2 = self.is_tense_recent()
        if ok2:
            return True, why2
        return False, why2 or "calm"

    def snapshot(self) -> dict:
        ok_int, why_int = self.should_intervene()
        ok_conf, why_conf = self.is_confused_recent()
        return {
            "frontmost_app": self._app,
            "is_target_app": self.is_active_target(),
            "is_confused_recent": ok_conf,
            "confused_reason": why_conf,
            "should_intervene": ok_int,
            "intervene_reason": why_int,
            "target_apps": sorted(self.target_apps),
            "confused_window_sec": self.confused_window_sec,
        }


# ---------- Enter 键拦截 ----------

class EnterSuppressor:
    """当 active_fn() 返回 True，拦截下一个 Return/KP_Enter 并触发 on_blocked。

    注意：需要 macOS 辅助功能权限。失败时静默忽略（log warning）。
    """

    def __init__(self, active_fn: Callable[[], bool],
                 on_blocked: Optional[Callable[[], None]] = None,
                 cooldown_sec: float = 1.5):
        self.active_fn = active_fn
        self.on_blocked = on_blocked
        self.cooldown_sec = cooldown_sec
        self._listener = None
        self._last_block_ts = 0.0
        self._block_count = 0

    def start(self):
        try:
            from pynput import keyboard as kb  # noqa: F401
            from Quartz import (
                kCGEventKeyDown, kCGEventFlagsChanged,  # noqa: F401
            )
        except Exception as e:
            log.warning("enter suppressor disabled (import failed): %s", e)
            return
        try:
            from pynput import keyboard as kb
            from pynput.keyboard import Key

            def darwin_intercept(event_type, event):
                # event_type: pynput 内部 Quartz 类型；只处理 KeyDown
                # 直接用 Quartz 提取 keycode
                try:
                    from Quartz import (
                        CGEventGetIntegerValueField,
                        kCGKeyboardEventKeycode,
                    )
                    keycode = CGEventGetIntegerValueField(
                        event, kCGKeyboardEventKeycode
                    )
                except Exception:
                    return event
                # macOS keycodes: 36=Return, 76=KeypadEnter
                if keycode not in (36, 76):
                    return event
                try:
                    if not self.active_fn():
                        return event
                except Exception:
                    return event
                # 冷却：避免一段时间内反复触发
                now = time.time()
                if now - self._last_block_ts < self.cooldown_sec:
                    return event
                self._last_block_ts = now
                self._block_count += 1
                log.info("Enter intercepted (#%d) — VSCode confused state",
                         self._block_count)
                try:
                    if self.on_blocked is not None:
                        self.on_blocked()
                except Exception:
                    log.exception("on_blocked failed")
                # 返回 None 即吞掉
                return None

            self._listener = kb.Listener(
                on_press=lambda *_: None,  # 不需要常规 on_press
                darwin_intercept=darwin_intercept,
            )
            self._listener.start()
            log.info("enter suppressor started (cooldown=%.1fs) — 需要辅助功能权限",
                     self.cooldown_sec)
        except Exception:
            log.exception("enter suppressor failed to start")

    def stop(self):
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

    def snapshot(self) -> dict:
        return {
            "running": self._listener is not None,
            "block_count": self._block_count,
            "cooldown_sec": self.cooldown_sec,
        }


# 30 句更挑衅/不耐烦版本，专门用在用户「最近 1min intensity 偏高 + 还在猛敲回车」时
PROVOKE_NUDGES = [
    "您再想想啊！",
    "这就完事了？",
    "这么急着回车，想清楚了吗？",
    "停，先别按。",
    "嘿，你确定吗？",
    "再读一遍，再按。",
    "别糊弄我，也别糊弄你自己。",
    "就这？真就这？",
    "诶，再想想呗。",
    "回车不是答案。",
    "你脸上写着没想好。",
    "深呼吸，别硬来。",
    "急啥？再看一眼。",
    "你不慌，我都替你慌。",
    "再问一次：为什么？",
    "你这是赌还是想？",
    "别让回车替你思考。",
    "醒醒，再来一遍。",
    "再憋三秒。",
    "这味儿不对，再想。",
    "别凑合，从头说。",
    "你确定这是你要的？",
    "心不静，先停。",
    "再说服我一次。",
    "急也没用，理一理。",
    "你刚才皱眉了，听见没？",
    "够了，醒醒。",
    "停下，回头。",
    "不准回车，先说人话。",
    "再一次，认真点。",
]


def random_prompt() -> str:
    return random.choice(PROMPTS)


def random_nudge() -> str:
    return random.choice(SHORT_NUDGES)


def random_provoke() -> str:
    return random.choice(PROVOKE_NUDGES)
