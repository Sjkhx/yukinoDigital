# TTS 积木：openbmb/VoxCPM2（流式 API + Ultimate Cloning 音色克隆）。
# 作为自定义 handler 由 voxemw.pipeline.launch 在运行时注册进
# huggingface/speech-to-speech 管线（module_kwargs.tts == "voxcpm"）。
# 实现继承自 v0.3 线上跑通的 extensions/tts_voxcpm_handler.py。
#
# 要点：
# - Ultimate Cloning：参考音频同时作为 prompt（音频续写，配台词文本）和
#   reference（音色克隆）传给模型，官方称相似度最高。启动时为每个音色调
#   tts_model.build_prompt_cache() 预编码（纯音频特征编码，无需 ASR），
#   之后每句合成直接复用 cache，热切换零成本、无磁盘缓存。
# - 流式：tts_model._generate_with_prompt_cache(streaming=True) 逐 chunk
#   产出 48kHz float，边收边重采样到 16kHz、按 blocksize 切块输出——
#   首音延迟只取决于第一个 chunk，不必等整句合成完（这正是换 VoxCPM2 的原因）。
#   写法与 voxcpm 包 core.py 的 generate_streaming 完全一致（wav.squeeze(0).cpu()）。
# - 输出采样率读 model.tts_model.sample_rate（AudioVAE V2 输出 48kHz），不硬编码。
# - 多音色热切换：realtime 客户端 session.update 的 session.audio.output.voice
#   按名字选 voices 里预编码的 cache（读法仿上游 qwen3_tts_handler 的
#   _apply_session_voice_override，只取 session 级），未匹配回退默认音色。

from __future__ import annotations

import json
import logging
import re
from math import gcd
from threading import Event
from time import perf_counter
from typing import Any, Iterator, Optional

import numpy as np
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)
console = Console()

PIPELINE_SR = 16000  # 管线音频采样率（与上游其他 TTS handler 一致）

DEFAULT_VOICE = "default"  # ref_audio/ref_text 启动参数对应的音色名（voices 未命中时的回退）


class BilingualFlow:
    """「【日语】…【译文】…」LLM 输出格式的流式切分器。

    同一逻辑供两处使用：
    - TTS 侧（tts_voxcpm）：feed() 的日语增量送合成、译文增量丢弃（只念日语）
    - 显示侧（launch.py patch on_assistant_text）：日语增量进 transcript、
      译文增量发 vox.translation.delta（聊天框显示翻译）

    标记可能被 LLM 流式输出从中间切开（如 "【日" 后断 chunk）：feed 保留
    尾部安全前缀等下一个 chunk 补全，不产生错误增量。模式：
      none —— 尚未见到【日语】：增量恒空、缓冲全保留（标记前内容丢弃，
              但保留完整缓冲供 EndOfResponse 兜底——LLM 整轮未按格式输出
              时把全文按原行为朗读）
      ja   —— 在【日语】段内（增量输出日语）
      zh   —— 已进入【译文】段（增量输出译文）
    """

    JA = "【日语】"
    ZH = "【译文】"

    def __init__(self):
        self.buf = ""
        self.mode = "none"

    def feed(self, text: str) -> tuple[str, str]:
        """喂入一段流式文本，返回 (日语增量, 译文增量)。"""
        # LLM 标签漂移容错：简体「【日本語】」等价于「【日语】」
        #（不识别会把整段带标签兜底合成 → 长段崩坏/标签进语音，2026-08-12）
        text = text.replace("【日本語】", "【日语】")
        self.buf += text
        ja_d = ""
        zh_d = ""
        while True:
            s = self.buf
            if self.mode == "none":
                if self.JA in s:
                    # 标记前内容（LLM 多余输出）丢弃，从标记后开始
                    self.buf = s[s.index(self.JA) + len(self.JA):]
                    self.mode = "ja"
                    continue  # 立即处理标记后的内容
                break  # 无标记：整段保留，不产出增量（EndOfResponse 兜底用）
            if self.mode == "ja":
                if self.ZH in s:
                    j = s.index(self.ZH)
                    ja_d += s[:j]
                    self.buf = s[j + len(self.ZH):]
                    self.mode = "zh"
                    continue
                safe = len(s) - (len(self.ZH) - 1)
                if safe > 0:
                    ja_d += s[:safe]
                    self.buf = s[safe:]
                break
            # zh：译文是尾段，后续不再有标记，直接全量输出
            zh_d += s
            self.buf = ""
            break
        return ja_d, zh_d

    def reset(self) -> None:
        self.buf = ""
        self.mode = "none"


class _AtempoStretcher:
    """ffmpeg atempo 流式变速（保调）封装：16kHz mono f32 进/出，每句一个实例。

    背景：VoxCPM2 克隆语速实测比参考音快 ~12%（同文本合成 21.4s vs ref.wav 24.2s），
    与分段/cfg/timesteps 无关，模型固有。tts.rate=0.886 即补偿（时长 ×1/0.886）。
    atempo 内部有几十 ms 分析窗，首段输出比输入晚一个窗，属正常流式延迟。
    """

    def __init__(self, sample_rate: int, rate: float):
        import queue
        import subprocess
        import threading

        self._q: queue.Queue = queue.Queue()
        self._p = subprocess.Popen(
            [
                "ffmpeg", "-v", "error",
                "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
                "-af", f"atempo={rate}",
                "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
        )
        self._buf = b""
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        while True:
            data = self._p.stdout.read(65536)
            if not data:
                self._q.put(None)
                return
            self._q.put(data)

    def feed(self, audio_f32) -> "np.ndarray":
        """喂一块 16kHz f32，返回当前可得的拉伸输出（可能为空，窗口延迟）。"""
        import queue

        import numpy as np

        self._p.stdin.write(audio_f32.astype(np.float32).tobytes())
        self._p.stdin.flush()
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                self._buf += item
        out, self._buf = self._buf, b""
        return np.frombuffer(out, dtype=np.float32)

    def flush(self) -> "np.ndarray":
        """句尾收干（关闭 stdin 后读干到 EOF）。"""
        import queue

        import numpy as np

        try:
            self._p.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        while True:
            try:
                item = self._q.get(timeout=10)
            except queue.Empty:
                logger.warning("atempo flush 超时（ffmpeg 异常？），丢弃尾部")
                break
            if item is None:
                break
            self._buf += item
        self._p.wait()
        out, self._buf = self._buf, b""
        return np.frombuffer(out, dtype=np.float32)

    def close(self) -> None:
        """打断废弃：直接杀进程，不等收尾。"""
        try:
            self._p.kill()
        except OSError:
            pass


class VoxCPMTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """
    Handles Text-to-Speech using openbmb/VoxCPM2 (streaming + Ultimate Cloning).
    """

    def setup(
        self,
        should_listen: Event,
        model_name: str = "openbmb/VoxCPM2",
        device: str = "cuda:0",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        voices: Optional[str] = None,
        sample_rate: int = PIPELINE_SR,
        blocksize: int = 512,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        optimize: bool = False,
        load_denoiser: bool = False,
        rate: float = 1.0,
        bilingual: bool = False,
        gen_kwargs: dict[str, Any] | None = None,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        from voxcpm import VoxCPM

        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.bilingual = bilingual
        self._bilingual = BilingualFlow() if bilingual else None
        if bilingual:
            logger.info("VoxCPM 双语模式: 只合成「【日语】」段,「【译文】」段不朗读")
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.gen_kwargs = gen_kwargs or {}
        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps
        self.rate = rate  # 语速补偿（atempo，1.0=不变；0.886 ≈ 抵消克隆语速 +12%）
        if self.rate != 1.0:
            logger.info("VoxCPM 语速补偿开启: atempo rate=%.3f（时长 ×%.3f）",
                        self.rate, 1.0 / self.rate)

        logger.info("Loading VoxCPM model: %s on %s", model_name, device)
        self.model = VoxCPM.from_pretrained(
            model_name,
            load_denoiser=load_denoiser,  # denoiser 需额外下 ModelScope 模型，ref 干净就关
            optimize=optimize,  # torch.compile：启动慢、首次合成快；RTF 0.3 不编译也够用
            device=device,
        )
        # AudioVAE V2 输出采样率（48kHz），不硬编码
        self._model_sr = int(self.model.tts_model.sample_rate)

        # 预编码 Ultimate Cloning prompt cache：默认音色（ref_audio/ref_text）
        # + voices JSON 里的每个命名音色。cache 在显存里，热切换只是选 dict 项。
        self.voice_prompts: dict[str, Any] = {}
        self.default_voice: Optional[str] = None
        if ref_audio:
            self.voice_prompts[DEFAULT_VOICE] = self._build_voice_cache(ref_audio, ref_text)
            self.default_voice = DEFAULT_VOICE
        if voices:
            try:
                voice_map = json.loads(voices)
            except json.JSONDecodeError as e:
                raise ValueError(f"voxcpm_tts_voices is not valid JSON: {e}") from e
            if not isinstance(voice_map, dict):
                raise ValueError(
                    "voxcpm_tts_voices must be a JSON object: {name: {ref_audio, ref_text}}"
                )
            for name, spec in voice_map.items():
                v_ref_audio = (spec or {}).get("ref_audio")
                if not v_ref_audio:
                    logger.warning("Skipping VoxCPM voice %r: missing ref_audio", name)
                    continue
                self.voice_prompts[name] = self._build_voice_cache(
                    v_ref_audio, (spec or {}).get("ref_text")
                )
        self._active_voice = self.default_voice

        # 48kHz -> 16kHz: 16000/48000 = 1/3（与上游 pocket/qwen3 handler 相同的有理重采样）
        g = gcd(self.sample_rate, self._model_sr)
        self._resample_up = self.sample_rate // g
        self._resample_down = self._model_sr // g
        self._needs_resampling = self.sample_rate != self._model_sr

        self.warmup()

    def _build_voice_cache(self, ref_audio: str, ref_text: Optional[str]):
        """为单个音色构建 Ultimate Cloning prompt cache。

        有 ref_text：prompt（音频+台词续写）+ reference（克隆）双路，官方相似度最高；
        无 ref_text：退化为 reference-only 基础克隆（build_prompt_cache 不允许
        prompt_wav_path 单传而不配 prompt_text）。
        """
        if ref_text:
            logger.info("Building VoxCPM Ultimate-Cloning prompt cache from %s", ref_audio)
            return self.model.tts_model.build_prompt_cache(
                prompt_text=ref_text,
                prompt_wav_path=ref_audio,
                reference_wav_path=ref_audio,
            )
        logger.warning(
            "VoxCPM voice %s has no ref_text; falling back to reference-only cloning",
            ref_audio,
        )
        return self.model.tts_model.build_prompt_cache(reference_wav_path=ref_audio)

    def _select_prompt(self, tts_input: TTSIn):
        """按 session.audio.output.voice 选音色 cache；缺省/未命中回退默认音色。

        读法仿上游 qwen3_tts_handler._apply_session_voice_override（只取 session 级，
        不管 response 级覆盖）。
        """
        voice: Optional[str] = None
        runtime_config = getattr(tts_input, "runtime_config", None)
        if runtime_config is not None:
            session = getattr(runtime_config, "session", None)
            audio = getattr(session, "audio", None) if session is not None else None
            output = getattr(audio, "output", None) if audio is not None else None
            sess_voice = getattr(output, "voice", None) if output is not None else None
            voice = str(sess_voice) if sess_voice else None

        name = self.default_voice
        if voice:
            if voice in self.voice_prompts:
                name = voice
            else:
                logger.warning("Unknown VoxCPM voice %r; falling back to %r", voice, name)
        if name != self._active_voice:
            logger.info("VoxCPM voice: %s -> %s", self._active_voice, name)
            self._active_voice = name
        return self.voice_prompts.get(name) if name else None

    @property
    def min_time_to_debug(self) -> float:
        # 音频 chunk 很多，避免刷屏；只记录异常慢的 chunk
        return 0.1

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}")
        try:
            for _ in self._synthesize("Hello, this is a warmup."):
                pass
            logger.info(f"{self.__class__.__name__} warmed up")
        except Exception as e:
            logger.warning(f"Warmup generation failed: {e}")

    def _to_int16(self, audio: np.ndarray) -> np.ndarray:
        return np.clip(audio * 32768, -32768, 32767).astype(np.int16)

    def _is_stale(self, cancel_gen: int | None) -> bool:
        return (
            cancel_gen is not None
            and self.cancel_scope is not None
            and self.cancel_scope.is_stale(cancel_gen)
        )

    def _synthesize(
        self,
        text: str,
        cancel_gen: int | None = None,
        voice_prompt_cache: Any = None,
    ) -> Iterator[np.ndarray]:
        """流式合成一句文本：边生成边重采样、按 blocksize 产出 int16 音频块。

        voice_prompt_cache 缺省用默认音色（warmup 即走这条）。
        """
        if self._is_stale(cancel_gen):
            logger.info("TTS generation cancelled (interruption)")
            return

        # 与 voxcpm core.py 一致的文本预处理（换行/空白折叠）
        text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
        # 滤掉括号动作/旁白（（笑）（拍大腿）等）:人设禁止朗读动作,但 LLM 仍偶尔输出;
        # 限 20 字内的短括号段,避免误伤正常括注。只影响 TTS,转写显示保留原文
        text = re.sub(r"[（(][^（）()]{1,20}[)）]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return

        start = perf_counter()
        if voice_prompt_cache is None and self.default_voice is not None:
            voice_prompt_cache = self.voice_prompts.get(self.default_voice)

        # 流式生成（写法对齐 voxcpm core.py 的 generate_streaming，
        # 区别只是复用启动时预编码的 prompt cache，不每句重编码参考音频）
        gen = self.model.tts_model._generate_with_prompt_cache(
            target_text=text,
            prompt_cache=voice_prompt_cache,
            min_len=2,
            max_len=2000,
            inference_timesteps=self.inference_timesteps,
            cfg_value=self.cfg_value,
            retry_badcase=False,  # streaming 模式不支持（模型内部也会强制关掉）
            streaming=True,
            **self.gen_kwargs,
        )

        from scipy.signal import resample_poly

        # 语速补偿（rate≠1.0）：每句一个 atempo 管道，流式保调变速
        stretcher = _AtempoStretcher(self.sample_rate, self.rate) if self.rate != 1.0 else None
        cancelled = False
        pending = np.empty(0, dtype=np.int16)
        total_out = 0
        first_chunk_at: Optional[float] = None
        try:
            for wav, _, _ in gen:
                if self._is_stale(cancel_gen):
                    cancelled = True
                    logger.info("TTS generation cancelled (interruption)")
                    return
                # core.py: wav.squeeze(0).cpu().numpy()（48kHz 单声道 float）
                audio = np.atleast_1d(
                    np.asarray(wav.squeeze(0).cpu().numpy(), dtype=np.float32).squeeze()
                )
                if audio.size == 0:
                    continue
                if first_chunk_at is None:
                    first_chunk_at = perf_counter() - start
                    logger.info(f"VoxCPM TTFA: {first_chunk_at:.2f}s (streaming first chunk)")
                if self._needs_resampling:
                    audio = resample_poly(audio, up=self._resample_up, down=self._resample_down)
                if stretcher is not None:
                    audio = stretcher.feed(audio)
                    if audio.size == 0:
                        continue  # atempo 窗口延迟，本块暂无输出
                pending = np.concatenate([pending, self._to_int16(audio)])
                while len(pending) >= self.blocksize:
                    yield pending[: self.blocksize]
                    pending = pending[self.blocksize :]
                    total_out += self.blocksize
        finally:
            gen.close()
            if stretcher is not None:
                if cancelled:
                    stretcher.close()  # 打断：丢弃未收尾音频
                else:
                    tail = stretcher.flush()
                    if tail.size:
                        pending = np.concatenate([pending, self._to_int16(tail)])

        if len(pending) > 0:
            # flush 尾巴可能很大（atempo 窗尾可达数千采样），先整块吐完再补零收尾；
            # 之前直接 np.pad(pending, (0, blocksize-len))，pending>blocksize 时
            # pad 宽度为负 → ValueError → 句尾被静默丢弃（「吞音」根因，2026-08-05）
            while len(pending) >= self.blocksize:
                yield pending[: self.blocksize]
                pending = pending[self.blocksize :]
                total_out += self.blocksize
            if len(pending) > 0:
                total_out += len(pending)
                yield np.pad(pending, (0, self.blocksize - len(pending)))

        generation_time = perf_counter() - start
        audio_duration = total_out / self.sample_rate
        rtf = generation_time / audio_duration if audio_duration > 0 else 0
        logger.info(
            f"VoxCPM streamed {audio_duration:.2f}s audio in {generation_time:.2f}s (RTF: {rtf:.2f})"
        )

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = getattr(self, "speculative_turns", None)
        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id,
                tts_input.turn_revision,
            ):
                return
            # 双语兜底：LLM 整轮都没输出「【日语】」标记（未按格式）——
            # 缓冲里的全部文本按原行为合成立即念出，避免静音
            if self._bilingual is not None:
                if self._bilingual.mode == "none":
                    tail = self._bilingual.buf.strip()
                    if tail:
                        logger.warning("双语格式缺失（无【日语】标记），整段回退合成")
                        gen = self.cancel_scope.generation if self.cancel_scope else None
                        voice_prompt_cache = self._select_prompt(tts_input)
                        try:
                            yield from self._synthesize(tail, gen, voice_prompt_cache)
                        except Exception as e:
                            logger.error(f"Error during VoxCPM fallback generation: {e}", exc_info=True)
                # 无条件重置切分状态：否则上一轮结束在 zh 模式（【译文】段），
                # 下一轮 LLM 文本会被全当译文丢弃 → 第二次回复没声音（2026-08-10）
                self._bilingual.reset()
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id,
            tts_input.turn_revision,
        ):
            logger.debug("Dropping stale TTS input for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision)
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        gen = self.cancel_scope.generation if self.cancel_scope else None
        text = tts_input.text
        if not text or not text.strip():
            return

        console.print(f"[green]ASSISTANT: {text}")

        # 双语模式：流式切分「【日语】」段朗读，「【译文】」段只显示不朗读
        if self._bilingual is not None:
            ja_d, _ = self._bilingual.feed(text)
            logger.info("TTS bilingual: in=%r ja_d=%r mode=%s buf=%r",
                        text[:40], ja_d[:40], self._bilingual.mode, self._bilingual.buf[:20])
            if not ja_d.strip():
                return  # 本 chunk 无日语内容（译文段/标记前内容）不合成
            text = ja_d

        voice_prompt_cache = self._select_prompt(tts_input)
        try:
            yield from self._synthesize(text, gen, voice_prompt_cache)
        except Exception as e:
            logger.error(f"Error during VoxCPM generation: {e}", exc_info=True)

    def cleanup(self) -> None:
        try:
            del self.model
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("VoxCPM handler cleaned up")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
