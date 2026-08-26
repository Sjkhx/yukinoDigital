"""AI 调用调试日志 + 演出指令解析（s2s 管线与 orchestrator 共用）。

每轮 AI 调用（管线对话 / orchestrator task-done 点评）写一个独立日志文件，
方便调试人设输出、格式漂移、演出编排。目录默认 <repo>/log/ai，
可用环境变量 VOXEMW_AI_LOG_DIR 覆盖。
"""

from __future__ import annotations

import datetime
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# 演出指令：按顺序提取 "type:value" 步骤（容忍 ; | , 分隔、中文冒号、多余文字）
PERF_STEP_RE = re.compile(
    r"(expression|motion|pose|pause)\s*[:：]\s*([0-9A-Za-z一-鿿_-]+)"
)

PERF_MARKER = "【演出】"

# 步骤值白名单（与前端 yukino2d.js 的解析能力对齐）：
# - motion：基础动作 1a~4c（字母=姿态底座）、特效 sweat/uruuru/yami/hatena/ikari/mayu
#   及其英文别名（前端 alias 表）；纯数字按 Action 序号放行（前端按 model3.json 顺序取）
# - expression：yuk_em0~13 对应数字 0~13，或人设英文别名（em 文件名不含语义词，
#   前端查 EXPRESSION_ALIASES 表解析）
# - pose：A/B/C 或 1/2/3，可带 pose 前缀
KNOWN_MOTIONS = {
    "1a", "2a", "3a", "4a", "1b", "2b", "3b", "4b", "1c", "2c", "3c", "4c",
    "sweat", "uruuru", "yami", "hatena", "ikari", "mayu",
    "angry", "confused", "sparkle", "dark", "eyebrow",
}
KNOWN_EXPRESSIONS = {
    "calm", "smile", "serious", "cold", "helpless", "awkward",
    "scrutinize", "sidelong", "surprised", "displeased", "troubled",
    "shy", "blush", "chuckle",
}
_DIGITS_RE = re.compile(r"^\d+$")
_POSE_VALUE_RE = re.compile(r"^(?:pose)?[a-c1-3]$", re.IGNORECASE)
_PAUSE_VALUE_RE = re.compile(r"^\d{1,5}$")


def _step_drop_reason(typ: str, val: str) -> str | None:
    """步骤值不在白名单时返回丢弃原因，合法返回 None。"""
    v = (val or "").strip().lower()
    if typ == "pause":
        return None if _PAUSE_VALUE_RE.fullmatch(v) else "停顿不是整数毫秒"
    if typ == "motion":
        if _DIGITS_RE.fullmatch(v) or v in KNOWN_MOTIONS:
            return None
        return "未知动作"
    if typ == "expression":
        if (_DIGITS_RE.fullmatch(v) and 0 <= int(v) <= 13) or v in KNOWN_EXPRESSIONS:
            return None
        return "未知表情"
    if typ == "pose":
        if _POSE_VALUE_RE.fullmatch(v):
            return None
        return "未知姿态"
    return "未知类型"

_ai_log_counter = 0


def resolve_ai_log_dir() -> Path:
    """日志目录：优先 VOXEMW_AI_LOG_DIR 环境变量，默认 <yukino>/log/ai。"""
    default = str(Path(__file__).resolve().parent.parent / "log" / "ai")
    return Path(os.environ.get("VOXEMW_AI_LOG_DIR", "") or default)


def parse_perf_directive_verbose(text: str) -> tuple[list[dict], list[str]]:
    """解析【演出】指令 → (合法步骤列表, 丢弃记录列表)。

    步骤值为 [{type, value}, ...]；丢弃记录形如 "motion:摇头 (未知动作)"，
    供 ai_log 观察人设格式漂移（此前坏步骤在前端静默丢，后端无痕）。"""
    steps: list[dict] = []
    dropped: list[str] = []
    for m in PERF_STEP_RE.finditer(text):
        typ, val = m.group(1), m.group(2)
        reason = _step_drop_reason(typ, val)
        if reason:
            dropped.append(f"{typ}:{val} ({reason})")
            continue
        if typ == "pause":
            steps.append({"type": typ, "value": int(val)})
        else:
            steps.append({"type": typ, "value": val})
    return steps, dropped


def parse_perf_directive(text: str) -> list[dict]:
    """解析【演出】指令为步骤列表：[{type, value}, ...]。坏步骤跳过。"""
    return parse_perf_directive_verbose(text)[0]


# 编排步骤开头：区分「行内引用【演出】」（后跟散文）和「真演出段」（后跟步骤）
_STEP_START = re.compile(r"\s*(expression|motion|pose|pause)\s*[:：]")


def _perf_marker_idx(text: str) -> int:
    """找【演出】标记位置：行首，或行内但后跟编排步骤 → 真标记；
    行内引用（如译文里写「【演出】的格式」）不算，避免译文被从中间截断。"""
    i = text.find(PERF_MARKER)
    while i >= 0:
        after = text[i + len(PERF_MARKER):]
        at_bol = i == 0 or text[i - 1] == "\n"
        if at_bol or _STEP_START.match(after):
            return i
        i = text.find(PERF_MARKER, i + len(PERF_MARKER))
    return -1


def split_perf_section(text: str) -> tuple[str, str]:
    """从文本里拆出【演出】段。返回 (去演出段文本, 演出段原文)。"""
    idx = _perf_marker_idx(text)
    if idx < 0:
        return text, ""
    return text[:idx].rstrip(), text[idx + len(PERF_MARKER):].strip()


def write_ai_log(
    source: str,
    raw: str,
    ja: str,
    zh: str,
    perf: str,
    steps: list,
    conn_id: str = "",
    dropped: list | None = None,
) -> None:
    """写一份 AI 调用日志（每次调用一个文件）。失败只告警，不阻断主流程。"""
    global _ai_log_counter
    try:
        d = resolve_ai_log_dir()
        d.mkdir(parents=True, exist_ok=True)
        _ai_log_counter += 1
        safe_conn = "".join(
            c for c in (conn_id or "none") if c.isalnum() or c in "-_"
        )[:16] or "none"
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = d / f"ai_{_ai_log_counter:03d}_{ts}_{source}_{safe_conn}.log"
        lines = [
            f"[AI 调用 #{_ai_log_counter}] {ts} | source={source} | conn_id={conn_id}",
            "=" * 60,
            "--- 完整 LLM 原文 ---",
            raw.rstrip(),
            "",
            "--- 切分结果 ---",
            f"日语: {ja!r}",
            f"译文: {zh!r}",
            f"演出: {perf!r}",
            "",
            "--- 编排步骤（vox.choreo） ---",
            str(steps) if steps else "(无)",
        ]
        if dropped:
            lines += ["", "--- 丢弃的演出步骤 ---"] + dropped
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("AI 调用日志已写入 %s", path)
    except Exception as e:
        logger.warning("AI 调用日志写入失败: %s", e)
