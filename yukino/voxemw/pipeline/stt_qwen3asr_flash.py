# STT 积木：Qwen3-ASR-Flash（阿里云百炼 DashScope，OpenAI 兼容模式）。
# 作为自定义 handler 由 voxemw.pipeline.launch 在运行时注册进
# huggingface/speech-to-speech 管线（module_kwargs.stt == "qwen3asr_flash"）。
#
# 云端推理、本地零显存。HTTP POST 同步调用，与 VAD 分段（判停 500ms）
# 天然契合——每段 <10s 短音频无需 WebSocket 长连接。
# 接口：POST /compatible-mode/v1/chat/completions，Bearer Token 认证。

from __future__ import annotations

import base64
import io
import json
import logging
import os
import tempfile
import urllib.request
import wave
from typing import Any, Iterator

import numpy as np
from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logger = logging.getLogger(__name__)
console = Console()

TARGET_SAMPLE_RATE = 16000

# 情绪侧信道：与 SenseVoice handler 一致，orchestrator 读它选垫音
EMOTION_SIDECAR_PATH = os.path.join(tempfile.gettempdir(), "voxemw_stt_emotion")


def _write_emotion_sidecar(emotion: str) -> None:
    """原子写情绪侧信道文件。"""
    try:
        tmp = f"{EMOTION_SIDECAR_PATH}.tmp"
        with open(tmp, "w") as f:
            f.write(emotion)
        os.replace(tmp, EMOTION_SIDECAR_PATH)
    except OSError as e:
        logger.warning("情绪侧信道写入失败: %s", e)


def _float32_to_wav_base64(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """float32 16kHz mono numpy → WAV 字节 → base64 data URL。"""
    pcm = np.clip(audio * 32768, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


class Qwen3ASRFlashSTTHandler(BaseSTTHandler):
    """DashScope Qwen3-ASR-Flash via OpenAI-compatible /v1/chat/completions."""

    def setup(
        self,
        api_key: str = "",
        model: str = "qwen3-asr-flash",
        language: str = "zh",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not self._api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY 未设置（环境变量或 stt.api_key 配置项），"
                "无法调用 DashScope ASR"
            )
        self._model = model
        self._language = language
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._gen_kwargs = gen_kwargs or {}
        # 无本地模型，无需 warmup；但上游 BaseSTTHandler.__init__ 会调 self.warmup()，
        # 这里给个空操作即可
        logger.info("Qwen3-ASR-Flash handler 就绪: model=%s url=%s", self._model, self._url)

    def warmup(self) -> None:
        """云端 API 无需预热。"""
        pass

    def _transcribe(self, audio: np.ndarray) -> str:
        """发送音频到 DashScope，返回转写文本。网络异常时抛 RuntimeError。"""
        data_url = _float32_to_wav_base64(audio, TARGET_SAMPLE_RATE)
        body = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": data_url},
                        }
                    ],
                }
            ],
            "asr_options": {
                "language": self._language,
                "enable_itn": True,
            },
            **self._gen_kwargs,
        }
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(
                f"DashScope ASR HTTP {e.code}: {body_text[:500]}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"DashScope ASR 网络错误: {e.reason}") from e

        try:
            choice = result["choices"][0]
            text = choice["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                f"DashScope ASR 响应解析失败: {json.dumps(result, ensure_ascii=False)[:500]}"
            ) from e

        # 情绪侧信道（DashScope 返回 emotion annotation）
        try:
            annotations = choice["message"].get("annotations") or []
            emotion = annotations[0].get("emotion", "NEUTRAL") if annotations else "NEUTRAL"
        except (IndexError, KeyError, TypeError):
            emotion = "NEUTRAL"
        _write_emotion_sidecar(str(emotion).upper())

        return text

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        logger.debug("infering qwen3-asr-flash (DashScope)...")

        audio = np.asarray(vad_audio.audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        try:
            pred_text = self._transcribe(audio)
        except RuntimeError as e:
            logger.error("DashScope ASR 失败: %s", e)
            pred_text = ""

        logger.debug("finished qwen3-asr-flash inference")

        if getattr(vad_audio, "mode", None) == "progressive":
            # 说话过程中的中间块：只作实时预览，不进 LLM
            yield PartialTranscription(
                text=pred_text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
            )
            return

        console.print(f"[yellow]USER: {pred_text}")

        yield Transcription(
            text=pred_text,
            language_code=None,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )
