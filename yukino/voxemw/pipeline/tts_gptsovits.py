# TTS 积木：GPT-SoVITS v4（Yukino 微调权重，经常驻推理服务调用）。
# 作为自定义 handler 由 voxemw.pipeline.launch 在运行时注册进
# huggingface/speech-to-speech 管线（module_kwargs.tts == "gptsovits"）。
#
# 架构：GPT-SoVITS 装在独立 conda 环境（GPTSoVITS），推理由常驻服务
# （scripts/gptsovits_server.py，127.0.0.1:8899）提供——模型常驻 GPU，
# 本 handler 只做 HTTP 请求 + 分块输出，不加载任何模型。
#
# 要点：
# - 非流式合成：GPT-SoVITS 整句推理（~1-3s/句），所以 process() 按
#   句子边界（。！？…」）积累 LLM 增量文本，凑齐一句才请求合成，
#   首音延迟 = 整句合成时间（与 VoxCPM2 的流式 ~0.5s 不同）。
# - 双语模式：复用 tts_voxcpm.BilingualFlow——只合成「【日语】」段，
#   「【译文】」段不朗读；EndOfResponse 兜底全量合成。
# - 服务返回 16kHz int16 mono PCM（服务端已重采样），handler 直接分块。
# - 打断：HTTP 请求发出后如被打断，丢弃结果不输出（服务端合成不可中断）。
# - 服务健康检查：启动时探测 /health，失败仅告警不阻塞（下次请求重试）。

from __future__ import annotations

import json
import logging
import re
import urllib.request
from threading import Event
from typing import Any, Iterator, Optional

import numpy as np
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

from voxemw.pipeline.tts_voxcpm import BilingualFlow

logger = logging.getLogger(__name__)
console = Console()

PIPELINE_SR = 16000  # 管线音频采样率（服务端已重采样到 16k）
DEFAULT_SERVER = "http://127.0.0.1:8899"

# 句子边界：句号/叹号/问号/引号收尾即切句；「…」是句内停顿不是边界
#（踩坑1：把 … 当边界会把「……別に」拆成碎片句 → 电音/重复听感，2026-08-12）。
# 攒句策略（踩坑2：单句超短合成音色不稳+词重复，试听=整段内部切分质量好）：
# 切出的句子先入 pending，攒够 SENT_BATCH 句或累计 SENT_MIN_CHARS 字符才合成，
# 多句合并为一段文本一次请求——由服务端内部切分（复现试听条件）。
_SENT_END = re.compile(r"[。！？」』]|(?<![A-Za-z])\.\s*$")
MIN_CHARS = 2
SENT_BATCH = 2  # 攒几句合成
SENT_MIN_CHARS = 24  # 或累计字符达此值
# 单次合成上限：80 → 40。长段（>60 字）「不切」模式下 AR 发散/拖沓率高
#（实测 95 字崩坏 60.3s、80 字段重复），≤40 字段 20 次合成 0 崩坏（2026-08-12）
MAX_CHARS = 40


class GPTSovitsTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """
    Handles Text-to-Speech via GPT-SoVITS v4 inference server (sentence-level).
    """

    def setup(
        self,
        should_listen: Event,
        server_url: str = DEFAULT_SERVER,
        speed: float = 1.0,
        sample_rate: int = PIPELINE_SR,
        blocksize: int = 512,
        bilingual: bool = False,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.bilingual = bilingual
        self._bilingual = BilingualFlow() if bilingual else None
        if bilingual:
            logger.info("GPT-SoVITS 双语模式: 只合成「【日语】」段,「【译文】」段不朗读")
        self.server_url = server_url.rstrip("/")
        self.speed = speed
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self._buf = ""  # 句子积累缓冲（bilingual 关闭时用）
        self._pending: list[str] = []  # 已切出的完整句（攒批合成）
        self._turn_key = None  # (turn_id, turn_revision)：新一轮开始时清空上一轮余量
        if speed != 1.0:
            logger.info("GPT-SoVITS 语速: %.2f", speed)

        self._check_health()

    def _check_health(self) -> None:
        try:
            with urllib.request.urlopen(f"{self.server_url}/health", timeout=3) as r:
                ok = r.read()
            logger.info("GPT-SoVITS 推理服务在线: %s (%s)", self.server_url, ok)
        except Exception as e:
            logger.warning("GPT-SoVITS 推理服务不可达 %s: %s（启动服务见 scripts/gptsovits_server.py）", self.server_url, e)

    def _synthesize(self, text: str) -> Iterator[np.ndarray]:
        """请求服务合成一句，按 blocksize 产出 int16 音频块。"""
        text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
        # 省略号 → 句号：GPT-SoVITS 对「……」「…」处理差，会触发 AR 循环
        # 生成 40s+ 噪声流（电流声根因，2026-08-12 实测 46.7s 崩坏句）
        text = re.sub(r"…{1,}", "。", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return
        req = json.dumps({"text": text, "speed": self.speed}).encode("utf-8")
        try:
            request = urllib.request.Request(
                f"{self.server_url}/tts",
                data=req,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=60) as r:
                pcm = r.read()
                sr = int(r.headers.get("X-Sample-Rate", self.sample_rate))
        except Exception as e:
            logger.error("GPT-SoVITS 合成请求失败: %s", e)
            return
        audio = np.frombuffer(pcm, dtype=np.int16)
        if sr != self.sample_rate:
            if sr % self.sample_rate == 0:
                # 32k→16k 用 decimate（IIR 8 阶 Chebyshev 抗混叠）——resample_poly
                # 抗混叠不足，SoVITS 8-16kHz 高频混叠进 0-8k = 刺耳电音
                #（2026-08-12 听感 A/B 确认：iir-decimate 电音消失，7-8k 混叠 0）
                from scipy.signal import decimate

                audio = decimate(audio, sr // self.sample_rate, ftype="iir").astype(
                    np.int16
                )
            else:
                # 兜底：非整数倍（当前服务固定 32k，理论不会走到）
                from math import gcd

                from scipy.signal import resample_poly

                g = gcd(sr, self.sample_rate)
                audio = resample_poly(
                    audio, up=self.sample_rate // g, down=sr // g
                ).astype(np.int16)
        logger.info("GPT-SoVITS 合成 %r -> %.2fs", text[:30], len(audio) / self.sample_rate)
        for i in range(0, len(audio), self.blocksize):
            block = audio[i : i + self.blocksize]
            if len(block) < self.blocksize:
                block = np.pad(block, (0, self.blocksize - len(block)))  # 末块补零（管线要求等长块）
            yield block

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = getattr(self, "speculative_turns", None)
        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id, tts_input.turn_revision
            ):
                return
            tail = ""
            if self._bilingual is not None:
                if self._bilingual.mode == "none":
                    tail = self._bilingual.buf.strip()  # 双语兜底：整段回退合成
                self._bilingual.reset()
                # BUGFIX(2026-08-16): 双语模式下 _buf/_pending 仍可能残留
                # MAX_CHARS 硬切余量（如「無駄」被切成「無」+「駄…」）。若不
                # 在本轮结束前合成，这部分会直接丢尾音，并串到下一轮开头，
                # 造成「字幕有、声音没有/下一轮开头冒出上一轮尾巴」。
                if self._buf.strip() or self._pending:
                    jp_tail = "".join(self._pending) + self._buf.strip()
                    self._pending = []
                    self._buf = ""
                    tail = (tail + jp_tail).strip()
            elif self._buf.strip() or self._pending:
                tail = "".join(self._pending) + self._buf.strip()
                self._pending = []
                self._buf = ""
            self._turn_key = None  # 下一轮 normal input 会重新触发清空（双保险）
            if tail:
                logger.warning("GPT-SoVITS 句尾兜底合成: %r", tail[:40])
                gen = self.cancel_scope.generation if self.cancel_scope else None
                try:
                    for chunk in self._synthesize(tail):
                        if self._is_stale(gen):
                            return
                        yield chunk
                except Exception as e:
                    logger.error("GPT-SoVITS 兜底合成失败: %s", e, exc_info=True)
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id, tts_input.turn_revision
        ):
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        # 新一轮（turn/revision 变化）开始：清空上一轮未合成余量，防止串音
        turn_key = (tts_input.turn_id, tts_input.turn_revision)
        if turn_key != self._turn_key:
            self._turn_key = turn_key
            self._buf = ""
            self._pending = []

        gen = self.cancel_scope.generation if self.cancel_scope else None
        text = tts_input.text
        if not text or not text.strip():
            return
        console.print(f"[green]ASSISTANT: {text}")

        # 双语模式：只取「【日语】」段增量
        if self._bilingual is not None:
            ja_d, _ = self._bilingual.feed(text)
            if not ja_d.strip():
                return
            text = ja_d

        # 句子积累：切出完整句入 pending，攒够批量或超长才合成（防单句超短合成）
        self._buf += text
        while True:
            m = _SENT_END.search(self._buf)
            if m and len(self._buf[: m.end()].strip()) >= MIN_CHARS:
                self._pending.append(self._buf[: m.end()].strip())
                self._buf = self._buf[m.end() :]
                continue
            break  # 等更多增量

        force = len(self._buf) >= MAX_CHARS  # 无边界超长：连未完成句一起合成
        pending_chars = sum(len(s) for s in self._pending)
        if not (len(self._pending) >= SENT_BATCH or pending_chars >= SENT_MIN_CHARS or force):
            return
        if force:
            # 无边界超长：连未完成句一起合成。总量超 MAX_CHARS 时硬切，余量留 _buf。
            text_to_synth = "".join(self._pending)
            self._pending = []
            text_to_synth += self._buf.strip()
            self._buf = ""
            if len(text_to_synth) > MAX_CHARS:
                self._buf = text_to_synth[MAX_CHARS:].strip() + self._buf
                text_to_synth = text_to_synth[:MAX_CHARS].strip()
        else:
            # 攒批合成：尽量按完整句边界取，避免把词从中间劈开。
            # 例：4 句共 49 字时，先合成前 3 句 31 字，最后一句留给
            # EndOfResponse 兜底或下一批——好过硬切 40 字把「無駄」劈成两半。
            text_to_synth = ""
            while self._pending:
                nxt = text_to_synth + self._pending[0]
                if len(nxt) > MAX_CHARS and text_to_synth:
                    break
                text_to_synth = nxt
                self._pending.pop(0)
                if len(text_to_synth) >= MAX_CHARS:
                    break
        if text_to_synth.strip():
            try:
                for chunk in self._synthesize(text_to_synth):
                    if self._is_stale(gen):
                        return
                    yield chunk
            except Exception as e:
                logger.error("GPT-SoVITS 合成失败: %s", e, exc_info=True)

    def _is_stale(self, cancel_gen: int | None) -> bool:
        return (
            cancel_gen is not None
            and self.cancel_scope is not None
            and self.cancel_scope.is_stale(cancel_gen)
        )

    def cleanup(self) -> None:
        logger.info("GPT-SoVITS handler cleaned up（推理服务为独立进程，不受影响）")
