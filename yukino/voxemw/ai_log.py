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

_ai_log_counter = 0


def resolve_ai_log_dir() -> Path:
    """日志目录：优先 VOXEMW_AI_LOG_DIR 环境变量，默认 <repo>/log/ai。"""
    default = str(Path(__file__).resolve().parent.parent.parent / "log" / "ai")
    return Path(os.environ.get("VOXEMW_AI_LOG_DIR", "") or default)


def parse_perf_directive(text: str) -> list[dict]:
    """解析【演出】指令为步骤列表：[{type, value}, ...]。坏步骤跳过。"""
    steps: list[dict] = []
    for m in PERF_STEP_RE.finditer(text):
        typ, val = m.group(1), m.group(2)
        if typ == "pause":
            try:
                steps.append({"type": typ, "value": int(val)})
            except ValueError:
                continue  # 停顿值不是数字：跳过该步骤
        else:
            steps.append({"type": typ, "value": val})
    return steps


def split_perf_section(text: str) -> tuple[str, str]:
    """从文本里拆出【演出】段。返回 (去演出段文本, 演出段原文)。"""
    idx = text.find(PERF_MARKER)
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
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("AI 调用日志已写入 %s", path)
    except Exception as e:
        logger.warning("AI 调用日志写入失败: %s", e)
