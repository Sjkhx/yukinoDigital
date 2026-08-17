"""Orchestrator：浏览器唯一入口，编排语音管线（s2s）与数字人服务（avatar）。

架构：
    浏览器 ←→ 本进程（aiohttp，:8000）
              ├→ s2s realtime ws（:8765，voxemw.pipeline.launch 起的语音管线）
              └→ avatar ws（:8767，voxemw.avatar.service 起的数字人服务，可缺席）

职责：
- 下行：s2s 的 TTS 音频 delta 双写 → 浏览器（播放）+ avatar（驱动口型）
- 上行：浏览器麦克风音频/控制消息 → 转发 s2s
- persona：浏览器发 {"type": "vox.persona", "id": ...} 切换人设，
  本进程把人设正文/音色/肖像注入三路（s2s instructions、TTS voice、avatar 肖像）
- 打断：s2s 报 speech_started → 通知 avatar 丢弃未消费音频、运动上下文归位
- 对话状态下发：由 s2s 事件推导 speech_active（说话期间 avatar 禁 idle 生成，
  防句间停顿插入 idle 帧卡画面）与 idle_mode（listening/thinking/calm，
  决定待机驱动音频），见 avatar_state_transition
- listen 双流：用户说话段的麦克风音频 tee 给 avatar 做 active listening
- 记忆：会话开始把 persona 记忆注入 instructions；response.done 后异步写入
- 垫音（filler，默认关）：转写完成即播预渲染口头禅盖 LLM 首句空白
- 降级：avatar 缺席时纯语音模式，前端显示静态肖像
- 2dlive：avatar.backend=2dlive（yukino2d 前端 WebGL 渲染）时无 avatar ws、
  无二进制帧下发——口型由浏览器本地 TTS 播放链 AnalyserNode 驱动
- 单用户单会话：新浏览器连接顶掉旧会话（s2s 只有 1 个管线槽位，
  换网络产生的僵尸会话被新连接立即踢掉，无需等超时/刷新两次）

浏览器侧协议（/ws）：
  文本帧（JSON）：
    → {"type": "vox.persona", "id": "<persona_id>"}   切换人设
    → {"type": "vox.drained"}                          播放排空信号（帧合流时序用）
    → OpenAI Realtime 事件原样透传（input_audio_buffer.append / response.cancel 等）
    ← OpenAI Realtime 事件原样透传（transcription / response.done 等）
    ← {"type": "vox.status", "avatar": "on"|"off", "persona": "<id>", ...}
  二进制帧：
    ← 0x01 + tag(1B) + JPEG：数字人视频帧（tag 0x00=idle 直画 / 0x01=speech 进队列）
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json

import numpy as np
import logging
import os
import random
import sys
import tempfile
import urllib.request
import wave
from pathlib import Path

import websockets

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

FRAME_TYPE_JPEG = 0x01
SAMPLE_RATE_16K = 16000  # 管线全程 16kHz（垫音/音频 delta 均为 int16 mono）

# s2s 事件 → 编排动作（纯函数分类，便于单测）
AUDIO_DELTA_EVENTS = {"response.output_audio.delta", "response.audio.delta"}  # GA / beta 名都收
AVATAR_RESET_EVENTS = {
    "input_audio_buffer.speech_started",  # 用户开口（打断）：avatar 停嘴
}


def build_session_update(persona_id: str, persona_text: str) -> dict:
    """注入人设的 session.update：instructions = 人设正文，voice = persona id
    （TTS voices 表 key，见 voxemw.pipeline.args.tts_setup_kwargs）。"""
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": persona_text,
            "audio": {
                "input": {
                    "turn_detection": {
                        "type": "server_vad",
                        "interrupt_response": True,
                    }
                },
                "output": {"voice": persona_id},
            },
        },
    }


def resolve_avatar_routing(avatar_cfg: dict, personas: dict) -> tuple[str | None, str]:
    """(avatar_url, avatar_backend)。

    backend="2dlive"（yukino2d 前端 WebGL 渲染）时不连 avatar ws、不启动
    服务（:8767 空闲），但 avatar_backend 仍下发 "2dlive"、状态报 on——
    前端本地渲染也算有数字人。其余 backend 走原有逻辑：enabled 且任一
    persona 有 ref_image 才建 ws 连接，否则后端降级 "off"。"""
    enabled = bool(avatar_cfg.get("enabled", True))
    backend = str(avatar_cfg.get("backend", "avtr1"))
    if backend == "2dlive":
        return None, "2dlive" if enabled else "off"
    available = enabled and any(p.get("ref_image") for p in personas.values())
    url = (
        f"ws://{avatar_cfg.get('host', '127.0.0.1')}:{avatar_cfg.get('port', 8767)}"
        if available else None
    )
    return url, backend if available else "off"


def classify_s2s_event(event: dict) -> tuple[bool, bool, bytes | None]:
    """分类 s2s 下行事件。返回 (relay_to_browser, reset_avatar, audio_pcm|None)。"""
    etype = event.get("type", "")
    pcm = None
    if etype in AUDIO_DELTA_EVENTS:
        delta = event.get("delta")
        if delta:
            pcm = base64.b64decode(delta)
    return True, etype in AVATAR_RESET_EVENTS, pcm


def avatar_state_transition(event: dict, speaking: bool) -> tuple[bool, list[dict]]:
    """s2s 事件 → avatar 状态控制消息（纯函数，便于单测）。

    返回 (new_speaking, 控制消息列表)。两个状态：
    - speech_active：首个音频 delta 开、response.done/打断关。说话期间 avatar
      禁 idle 生成——句间停顿 pending 排空时插入 idle 帧会被前端直画，卡画面
    - idle_mode：listening（用户开口）/ thinking（用户说完）/ calm（助手说完），
      决定待机驱动音频（persona 嘟囔循环或纯静音）
    """
    etype = event.get("type", "")
    msgs: list[dict] = []
    if etype in AUDIO_DELTA_EVENTS and event.get("delta"):
        if not speaking:
            speaking = True
            msgs.append({"type": "speech_active", "on": True})
    elif etype == "response.done":
        if speaking:
            speaking = False
            msgs.append({"type": "speech_active", "on": False})
        msgs.append({"type": "idle_mode", "mode": "calm"})
    elif etype == "input_audio_buffer.speech_started":
        if speaking:
            speaking = False
            msgs.append({"type": "speech_active", "on": False})
        msgs.append({"type": "idle_mode", "mode": "listening"})
    elif etype == "input_audio_buffer.speech_stopped":
        if not speaking:
            msgs.append({"type": "idle_mode", "mode": "thinking"})
    return speaking, msgs


FILLER_GROUPS = ("positive", "negative", "neutral")

# SenseVoice 情绪标签 → 垫音分组（stt_sensevoice 写 /tmp 侧信道，零额外推理成本）
EMOTION_TO_GROUP = {
    "HAPPY": "positive",
    "SURPRISED": "positive",
    "SAD": "negative",
    "ANGRY": "negative",
    "FEARFUL": "negative",
    "DISGUSTED": "negative",
}

EMOTION_SIDECAR_PATH = os.path.join(tempfile.gettempdir(), "voxemw_stt_emotion")


def yukino_task_done_reply(project: str, title: str, summary: str, config: dict) -> tuple[str, str]:
    """把 DSH 任务完成信息交给雪乃人设，生成【日语】+【译文】两段回复。"""
    import re

    try:
        from voxemw.config import resolve_api_key
        personas = config.get("personas") or {}
        resolved = personas.get("resolved") or {}
        default_id = personas.get("default") or "yukino"
        persona = resolved.get(default_id) or {}
        persona_text = (persona.get("text") or "").strip()
        llm = config.get("llm") or {}
        model = str(llm.get("model_name", "deepseek-v4-flash"))
        base_url = str(llm.get("base_url", "https://api.deepseek.com/v1"))
        api_key = resolve_api_key(llm)
    except Exception as e:
        logger.warning("task-done LLM 配置读取失败: %s", e)
        return "", ""

    user_prompt = (
        "DSH 刚完成了一个任务，请像平常对话一样回我。\n"
        f"项目：{project}\n"
        f"会话：{title}\n"
        f"最新总结：{summary or '（无详细结果）'}\n"
        "记住：用你的固定格式，先日语后中文译文。"
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": persona_text or "你是雪之下雪乃。"},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            result = json.loads(r.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning("task-done LLM 总结失败: %s", e)
        return "", ""

    # 按原项目格式拆出日语正文和中文译文
    ja = ""
    zh = ""
    m = re.search(r"【日语】([\s\S]*?)【译文】", content)
    if m:
        ja = m.group(1).strip()
        zh = content[m.end():].strip()
    else:
        m = re.search(r"【日语】([\s\S]*)", content)
        if m:
            ja = m.group(1).strip()
        else:
            ja = content
    return ja, zh


def synthesize_task_done_audio(text: str, server_url: str) -> bytes | None:
    """调用 GPT-SoVITS V2ProPlus 常驻服务合成音频，返回 16kHz int16 PCM bytes。

    失败返回 None（由调用方降级为纯文本显示）。服务端返回的采样率在
    X-Sample-Rate 头里（通常 32000），这里统一重采样到 16000。
    """
    import io

    try:
        import scipy.signal as sps  # noqa: F401
    except Exception:
        sps = None

    req = urllib.request.Request(
        f"{server_url.rstrip('/')}/tts",
        data=json.dumps({"text": text, "speed": 1.0}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            raw = r.read()
            sr = int(r.headers.get("X-Sample-Rate", "16000"))
    except Exception as e:
        logger.warning("task-done TTS 合成失败: %s", e)
        return None

    try:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if sr != 16000 and audio.size > 0:
            if sps is not None and sr % 16000 == 0:
                audio = sps.decimate(audio, sr // 16000, ftype="iir").astype(np.float32)
            elif sps is not None:
                from math import gcd
                g = gcd(sr, 16000)
                audio = sps.resample_poly(audio, up=16000 // g, down=sr // g).astype(np.float32)
            else:
                # 极简线性插值兜底：把原采样点按比例映射到 16000
                n = int(round(len(audio) * 16000 / sr))
                idx = np.linspace(0, len(audio) - 1, n)
                audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
        pcm = np.clip(audio, -32768, 32767).astype(np.int16)
        return pcm.tobytes()
    except Exception as e:
        logger.warning("task-done TTS 重采样失败: %s", e)
        return None


def read_emotion_sidecar(path: str = EMOTION_SIDECAR_PATH) -> str:
    """读 STT 写的情绪侧信道（缺失/异常回退 NEUTRAL）。"""
    try:
        return Path(path).read_text().strip() or "NEUTRAL"
    except OSError:
        return "NEUTRAL"


def _read_filler_wav(wav_path: Path) -> bytes | None:
    try:
        with wave.open(str(wav_path), "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
                logger.warning("垫音格式不符（需 16k mono s16），跳过: %s", wav_path)
                return None
            return w.readframes(w.getnframes())
    except (wave.Error, OSError) as e:
        logger.warning("垫音读取失败 %s: %s", wav_path, e)
        return None


def load_fillers(persona: dict) -> dict[str, list[tuple[bytes, str]]]:
    """persona 素材目录 fillers/<group>/*.wav → 分组 (PCM, 台词) 列表。
    根目录散落的 wav 归入 neutral；台词读 fillers/texts.json（相对路径→文本），
    缺条目台词为空串（跳过历史注入）。台词供注入 LLM 历史，让模型知道自己「说」过。"""
    groups: dict[str, list[tuple[bytes, str]]] = {g: [] for g in FILLER_GROUPS}
    image = persona.get("ref_image")
    if not image:
        return groups
    fdir = Path(image).parent / "fillers"
    texts: dict = {}
    texts_path = fdir / "texts.json"
    if texts_path.is_file():
        try:
            texts = json.loads(texts_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("垫音台词表读取失败 %s: %s", texts_path, e)
    for wav_path in sorted(fdir.rglob("*.wav")):
        pcm = _read_filler_wav(wav_path)
        if pcm is None:
            continue
        group = wav_path.parent.name if wav_path.parent != fdir else "neutral"
        if group not in groups:
            group = "neutral"
        text = texts.get(str(wav_path.relative_to(fdir)), "")
        groups[group].append((pcm, text))
    return groups


def build_filler_history_item(text: str) -> dict:
    """垫音台词 → 注入 LLM 历史的 assistant 消息（只入历史，不触发响应）。
    让模型知道自己刚「说」过这句垫音，后续轮次保持连贯。"""
    return {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        },
    }


def pick_filler_index(n: int, last: int) -> int:
    """随机选垫音下标，避免与上一条重复（纯函数，便于单测）。"""
    idx = random.randrange(n)
    if n > 1 and idx == last:
        idx = (idx + 1) % n
    return idx


class Session:
    """一个浏览器连接 ↔ 一路 s2s + 一路 avatar 的编排。"""

    def __init__(self, browser_ws, s2s_url: str, avatar_url: str | None,
                 personas: dict, default_persona: str,
                 filler_enabled: bool = True, avatar_backend: str = "avtr1",
                 memory_store=None, history_store=None, weibo_cfg: dict | None = None):
        import datetime
        import uuid

        self.browser = browser_ws
        self.s2s_url = s2s_url
        self.avatar_url = avatar_url
        self.avatar_backend = avatar_backend  # 前端据此选口型延迟默认值（adelay）
        self.personas = personas
        self.persona_id = default_persona
        self.memory = memory_store  # 记忆积木（None = 未启用/降级）
        self.history = history_store  # SQLite 对话历史（None = 未启用/降级）
        self.session_id = (
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        )
        weibo_cfg = weibo_cfg or {}
        self.weibo_db = weibo_cfg.get("db_path") if weibo_cfg.get("enabled") else None
        self.weibo_top_n = int(weibo_cfg.get("top_n", 8))
        self._turn_user_text = ""       # 本轮用户转写（记忆/历史写入用）
        self._turn_assistant_text = ""  # 本轮助手回复（记忆/历史写入用）
        self._turn_translation = ""     # 本轮译文（vox.translation.delta 累积）
        self._user_speaking = False  # server VAD 判定的用户说话段（listen 轨转发门控）
        self.s2s = None
        self.avatar = None
        self._avatar_speaking = False  # 是否已向 avatar 下发 speech_active=on
        self._filler_enabled = filler_enabled
        self._fillers = load_fillers(personas[default_persona])
        self._filler_last: dict[str, int] = {}  # 每个分组各自记上一条，避免连续重复
        self._filler_task: asyncio.Task | None = None

    async def run(self) -> None:
        import websockets

        # s2s 连接被拒（pipeline slot 被占/释放窗口，1008）时重试——直接抛会把
        # 浏览器会话崩成 aiohttp 500（历史 26 次 Error handling request）
        s2s = None
        for attempt in range(3):
            try:
                s2s = await websockets.connect(self.s2s_url, max_size=16 * 1024 * 1024)
                break
            except (websockets.ConnectionClosed, OSError) as e:
                logger.warning("s2s 连接失败（第 %d/3 次）: %s", attempt + 1, e)
                if attempt == 2:
                    return  # 优雅结束本会话，浏览器侧自动重连
                await asyncio.sleep(2)

        async with s2s:
            self.s2s = s2s
            if self.avatar_url:
                try:
                    self.avatar = await websockets.connect(
                        self.avatar_url, max_size=16 * 1024 * 1024
                    )
                except OSError as e:
                    logger.warning("avatar 服务不可达，降级纯语音: %s", e)
                    self.avatar = None
            try:
                await self._apply_persona(self.persona_id)
            except Exception as e:
                # persona 下发时 s2s 已断 → 本会话无意义，优雅退出（勿冒泡崩溃）
                logger.warning("persona 下发失败，结束本会话: %s", e)
                return
            await self._send_status()
            tasks = [
                asyncio.create_task(self._browser_to_s2s()),
                asyncio.create_task(self._s2s_to_browser()),
            ]
            if self.avatar is not None:
                tasks.append(asyncio.create_task(self._avatar_to_browser()))
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            self._cancel_filler()
            for task in pending:
                task.cancel()
            for task in done:
                logger.info("session 退出: task=%s exc=%r",
                            task.get_coro().__qualname__, task.exception())

    async def close(self) -> None:
        """关闭本会话（新浏览器连接顶掉旧连接时调用）。
        断开 s2s 释放管线槽位；转发协程随连接关闭自行退出。"""
        logger.info("close() 被调用: s2s=%s", self.s2s)
        if self.s2s is not None:
            try:
                await self.s2s.close()
            except Exception:
                pass

    async def _send_status(self) -> None:
        avatar_on = self.avatar is not None or self.avatar_backend == "2dlive"
        await self.browser.send_str(json.dumps({
            "type": "vox.status",
            "avatar": "on" if avatar_on else "off",
            "avatar_backend": self.avatar_backend,
            "persona": self.persona_id,
        }))

    async def _apply_persona(self, persona_id: str) -> None:
        persona = self.personas[persona_id]
        self.persona_id = persona_id
        self._fillers = load_fillers(persona)
        instructions = persona["text"]
        # 绝对时间锚点（最优先）：LLM 无时钟，不注入就只能靠对话里的相对时间
        # 猜——"周日是不是明天"这类问题必然错（2026-08-11 用户反馈）
        import datetime

        now = datetime.datetime.now()
        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        time_block = (
            "# 当前时间\n"
            f"今天是 {now.year}年{now.month}月{now.day}日 {weekdays[now.weekday()]}。"
            "回答涉及日期/时间的问题时以此为准，例如「今天」即上述日期。\n"
        )
        instructions = time_block + instructions
        if self.memory is not None:
            try:
                from voxemw.memory import build_core_block, build_memory_block

                # 注入顺序：核心资料（精确，最优先）→ 人设正文 → 向量召回记忆
                core = await asyncio.to_thread(self.memory.load_core, persona_id)
                core_block = build_core_block(core)
                if core_block:
                    instructions = core_block + "\n\n" + instructions
                    logger.info("核心资料注入（%d 字符）", len(core))
                memories = await asyncio.to_thread(self.memory.search, persona_id)
                block = build_memory_block(memories)
                if block:
                    instructions = instructions + "\n\n" + block
                    logger.info("记忆注入 %d 条", len(memories))
            except Exception as e:
                logger.warning("记忆召回失败（跳过）: %s", e)
        if self.weibo_db:
            from voxemw.weibo import build_posts_block, get_recent_posts

            posts = await asyncio.to_thread(
                get_recent_posts, self.weibo_db, self.weibo_top_n
            )
            block = build_posts_block(posts)
            if block:
                instructions = instructions + "\n\n" + block
                logger.info("动态注入 %d 条", len(posts))
        try:
            await self.s2s.send(json.dumps(build_session_update(persona_id, instructions)))
        except websockets.ConnectionClosed:
            # pipeline slot 被占/断开时 send 抛 1008——必须兜住，
            # 否则异常冒泡把整个 session.run 带崩，浏览器被莫名断开
            #（历史 26 次崩溃，2026-08-12 定位：pipeline pool size 1 的拒绝窗口）
            logger.warning("s2s 连接不可用（pipeline slot 占用或已断开），persona 下发失败")
            raise
        if self.avatar is not None:
            image = persona.get("ref_image")
            if image:
                await self.avatar.send(json.dumps({"type": "set_image", "path": image}))
            await self.avatar.send(json.dumps({"type": "reset"}))

    # ── 三条转发协程 ──

    async def _browser_to_s2s(self) -> None:
        append_count = 0
        async for message in self.browser:
            if message.type.name != "TEXT":
                continue  # 二进制帧（历史截帧协议）已废弃，直接忽略
            try:
                event = json.loads(message.data)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "vox.new_session":
                # 前端「新会话」按钮：生成新 session_id，后续历史写入归入新会话；
                # 本轮未写盘的残稿一并清空，避免串到新会话。
                import datetime as _dt
                import uuid as _uuid

                self.session_id = (
                    _dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + _uuid.uuid4().hex[:6]
                )
                self._turn_user_text = ""
                self._turn_assistant_text = ""
                self._turn_translation = ""
                logger.info("新会话: %s", self.session_id)
                continue
            if event.get("type") == "vox.persona":
                pid = event.get("id")
                if pid in self.personas:
                    await self._apply_persona(pid)
                    await self._send_status()
                continue
            if event.get("type") == "vox.drained":
                continue  # 帧合流后该信号仅作时序参考，无需动作
            # listen 轨 tee：用户说话段（server VAD 门控，防环境噪音/回声引起多余反应）
            # 的麦克风音频转发给 avatar 做 active listening（官方 listen 轨常开，
            # 这里按段转发是 deliberate 的门控收敛）
            if (event.get("type") == "input_audio_buffer.append"
                    and self.avatar is not None and self._user_speaking):
                await self.avatar.send(json.dumps({
                    "type": "listen", "pcm": event.get("audio", "")}))
            # 观测日志：浏览器→s2s 音频转发计数 + RMS/峰值（排查"说话无响应"：
            # RMS 跳动=音频有内容、接近 0=麦克风采集到静音）
            if event.get("type") == "input_audio_buffer.append":
                append_count += 1
                if append_count % 50 == 1:
                    rms, peak, n = -1.0, -1, 0
                    try:
                        import base64 as _b64

                        pcm = _b64.b64decode(event.get("audio", ""))
                        n = len(pcm) // 2
                        if n:
                            arr = np.frombuffer(pcm, dtype=np.int16)
                            rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
                            peak = int(np.abs(arr).max())
                    except Exception as e:
                        logger.warning("append 音频解析失败: %s", e)
                    logger.info(
                        "append 第 %d 条: rms=%.1f peak=%d len=%d",
                        append_count, rms, peak, n,
                    )
            try:
                await self.s2s.send(message.data)
            except Exception as e:
                logger.error("转发到 s2s 失败（session 结束）: %s", e)
                raise

    # ── 垫音：转写完成 → 立即播一条预渲染口头禅，填 LLM 首句的 ~1.4s 空白 ──

    def _cancel_filler(self) -> None:
        if self._filler_task is not None and not self._filler_task.done():
            self._filler_task.cancel()
        self._filler_task = None

    def _start_filler(self) -> None:
        self._cancel_filler()
        if not self._filler_enabled:
            return
        # 按用户情绪选垫音分组（SenseVoice 同次推理副产品，侧信道读取零成本）
        emotion = read_emotion_sidecar()
        group = EMOTION_TO_GROUP.get(emotion, "neutral")
        clips = self._fillers.get(group) or self._fillers.get("neutral") or []
        if not clips:
            return
        last = self._filler_last.get(group, -1)
        idx = pick_filler_index(len(clips), last)
        self._filler_last[group] = idx
        logger.info("垫音: 情绪=%s 分组=%s 第%d/%d条", emotion, group, idx + 1, len(clips))
        pcm, text = clips[idx]
        self._filler_task = asyncio.create_task(self._play_filler(pcm, text))

    async def _play_filler(self, clip: bytes, text: str = "") -> None:
        """把垫音 PCM 伪造成 response.output_audio.delta 推给浏览器+avatar，
        台词同步注入 LLM 历史（assistant 消息，只入历史不触发响应）。
        与真实回复构成「连续两段回复」（同截帧垫场→打分结构），前端唇同步原生支持。
        播完后把 avatar 切回待机（speech_active=off + idle_mode=thinking）：
        垫音帧与真回复帧之间的 LLM 等待空隙由待机微动桥接，画面不定格。"""
        try:
            if text:
                await self.s2s.send(json.dumps(build_filler_history_item(text), ensure_ascii=False))
            if self.avatar is not None and not self._avatar_speaking:
                self._avatar_speaking = True
                await self.avatar.send(json.dumps({"type": "speech_active", "on": True}))
            chunk = SAMPLE_RATE_16K * 2 * 2 // 5  # 0.4s int16 一块
            for i in range(0, len(clip), chunk):
                b64 = base64.b64encode(clip[i:i + chunk]).decode()
                await self.browser.send_str(
                    json.dumps({"type": "response.output_audio.delta", "delta": b64}))
                if self.avatar is not None:
                    await self.avatar.send(json.dumps({"type": "audio", "pcm": b64}))
            # 伪造 response.done 关闭垫音「回复」：前端下一个 delta（真回复）会重锚
            # 视频基准并清掉垫音的零填充闭嘴尾帧——否则 ~0.96s 尾帧占着帧序号，
            # 整条回复口型落后音频 ~1s（音频结束嘴还在动）
            await self.browser.send_str(json.dumps({"type": "response.done"}))
            if self.avatar is not None and self._avatar_speaking:
                self._avatar_speaking = False
                await self.avatar.send(json.dumps({"type": "speech_active", "on": False}))
                await self.avatar.send(json.dumps({"type": "idle_mode", "mode": "thinking"}))
        except asyncio.CancelledError:
            raise  # 用户插话取消：前端已被 speech_started flush，直接退出
        except Exception as e:
            logger.info("垫音播放中断（连接关闭？）: %r", e)

    def _maybe_write_memory(self) -> None:
        """response.done → 异步写入本轮对话到记忆（Mem0 抽取，不占语音延迟）。"""
        user_text, assistant_text = self._turn_user_text, self._turn_assistant_text
        self._turn_user_text = ""
        self._turn_assistant_text = ""
        if self.memory is None or not user_text:
            return

        async def _write():
            try:
                # 核心资料文档（精确）与 Mem0 向量记忆（笼统）并行异步写入
                await asyncio.to_thread(
                    self.memory.update_core, self.persona_id, user_text, assistant_text
                )
                await asyncio.to_thread(
                    self.memory.add_turn, user_text, assistant_text, self.persona_id
                )
            except Exception as e:
                logger.info("记忆写入失败（忽略）: %s", e)

        asyncio.create_task(_write())

    async def _s2s_to_browser(self) -> None:
        async for raw in self.s2s:
            if not isinstance(raw, str):
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                await self.browser.send_str(raw)
                continue
            self._track_dialog_state(event)
            etype = event.get("type", "")
            # 译文增量：vox.translation.delta 按流式下发，累积到本轮翻译
            if etype == "vox.translation.delta":
                self._turn_translation += (event.get("delta") or "")
            # 记忆/历史：跟踪本轮转写文本（写入发生在 response.done）
            if etype == "conversation.item.input_audio_transcription.completed":
                self._turn_user_text = (event.get("transcript") or "").strip()
                self._write_history_user()
            elif etype == "response.output_audio_transcript.done":
                self._turn_assistant_text += (event.get("transcript") or "")
            elif etype == "response.done":
                self._maybe_write_memory()
                self._write_history_assistant()
            # 转写完成 → 立即垫音；用户再开口（打断）→ 取消垫音
            if etype == "conversation.item.input_audio_transcription.completed":
                if (event.get("transcript") or "").strip():
                    self._start_filler()
            elif etype == "input_audio_buffer.speech_started":
                self._cancel_filler()
            relay, reset_avatar, pcm = classify_s2s_event(event)
            if pcm is not None:
                self._cancel_filler()  # 真音频抢先到达：停发垫音余量，防结尾误切待机
            if self.avatar is not None:
                self._avatar_speaking, ctrl_msgs = avatar_state_transition(
                    event, self._avatar_speaking
                )
                for msg in ctrl_msgs:
                    await self.avatar.send(json.dumps(msg))
            if pcm is not None and self.avatar is not None:
                await self.avatar.send(json.dumps({
                    "type": "audio",
                    "pcm": base64.b64encode(pcm).decode(),
                }))
            if reset_avatar and self.avatar is not None:
                await self.avatar.send(json.dumps({"type": "reset"}))
            if relay:
                await self.browser.send_str(raw)

    def _track_dialog_state(self, event: dict) -> None:
        etype = event.get("type", "")
        if etype == "input_audio_buffer.speech_started":
            self._user_speaking = True
        elif etype == "input_audio_buffer.speech_stopped":
            self._user_speaking = False

    def _write_history_user(self) -> None:
        if self.history is None or not self._turn_user_text:
            return
        try:
            self.history.add_message(
                self.session_id, "user", self._turn_user_text, persona=self.persona_id
            )
        except Exception as e:
            logger.info("历史写入用户消息失败（忽略）: %s", e)

    def _write_history_assistant(self) -> None:
        if self.history is None:
            return
        text = (self._turn_assistant_text or "").strip()
        if not text:
            return
        try:
            self.history.add_message(
                self.session_id, "assistant", text,
                translation=(self._turn_translation or "").strip(),
                persona=self.persona_id,
            )
        except Exception as e:
            logger.info("历史写入助手消息失败（忽略）: %s", e)
        finally:
            self._turn_user_text = ""
            self._turn_assistant_text = ""
            self._turn_translation = ""

    async def _avatar_to_browser(self) -> None:
        # 中转队列 + 独立发送任务：浏览器/隧道抖动时丢最旧帧，
        # 而不是 await 阻塞 avatar 读取、把背压传回数字人服务
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=25)

        async def sender() -> None:
            while True:
                await self.browser.send_bytes(await q.get())

        task = asyncio.create_task(sender())
        try:
            async for raw in self.avatar:
                if isinstance(raw, bytes):
                    if q.full():
                        try:
                            q.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    q.put_nowait(bytes([FRAME_TYPE_JPEG]) + raw)
        finally:
            task.cancel()


def create_app(config: dict):
    from aiohttp import web

    @web.middleware
    async def cors_middleware(request, handler):
        if request.method == "OPTIONS":
            return web.Response(
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, GET, OPTIONS, DELETE",
                    "Access-Control-Allow-Headers": "Content-Type",
                }
            )
        try:
            resp = await handler(request)
        except web.HTTPException as e:
            resp = e
        resp.headers.setdefault("Access-Control-Allow-Origin", "*")
        resp.headers.setdefault("Access-Control-Allow-Methods", "POST, GET, OPTIONS, DELETE")
        resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type")
        return resp

    server = config.get("server") or {}
    avatar_cfg = config.get("avatar") or {}
    personas = config["personas"]["resolved"]
    default_persona = config["personas"]["default"]

    s2s_url = f"ws://{server.get('s2s_host', '127.0.0.1')}:{server.get('s2s_port', 8765)}/v1/realtime"
    avatar_url, avatar_backend = resolve_avatar_routing(avatar_cfg, personas)
    filler_enabled = bool((config.get("filler") or {}).get("enabled", True))

    from voxemw.memory import create_memory_store
    from voxemw.chat_history import create_chat_history_store

    memory_store = create_memory_store(config)
    history_store = create_chat_history_store(config)

    async def index(_request):
        return web.FileResponse(REPO_ROOT / "web" / "index.html")

    async def api_personas(_request):
        return web.json_response({
            "default": default_persona,
            "avatar": "on" if avatar_backend != "off" else "off",
            "avatar_backend": avatar_backend,
            "list": [
                {
                    "id": pid,
                    "name": p["name"],
                    "label": p.get("label") or p["name"],
                    "has_image": bool(p.get("ref_image")),
                }
                for pid, p in personas.items()
            ],
        })

    async def api_memory(_request):
        """查看记忆知识库：核心资料文档 + 对话摘要列表（美化页面数据源）。"""
        if memory_store is None:
            return web.json_response({"enabled": False})
        core = await asyncio.to_thread(memory_store.load_core, default_persona)
        summaries = await asyncio.to_thread(memory_store.list_summaries, default_persona)
        return web.json_response({
            "enabled": True,
            "persona": default_persona,
            "core": core,
            "summary_count": len(summaries),
            "summaries": summaries,
        })

    async def api_memory_summary(request):
        """修改/删除单条摘要。DELETE = 删除；POST body {"summary": ...} = 修改（重新向量化）。"""
        if memory_store is None:
            return web.json_response({"ok": False, "error": "记忆未启用"})
        point_id = request.match_info["pid"]
        if request.method == "DELETE":
            ok = await asyncio.to_thread(memory_store.delete_summary, default_persona, point_id)
            return web.json_response({"ok": ok})
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "无效 JSON"}, status=400)
        text = (body or {}).get("summary", "")
        ok = await asyncio.to_thread(memory_store.update_summary, default_persona, point_id, text)
        return web.json_response({"ok": ok, "error": "" if ok else "修改失败（内容为空或不存在）"})

    async def api_memory_core(request):
        """手动保存核心资料文档。POST body {"core": ...}。"""
        if memory_store is None:
            return web.json_response({"ok": False, "error": "记忆未启用"})
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "无效 JSON"}, status=400)
        text = (body or {}).get("core", "")
        ok = await asyncio.to_thread(memory_store.save_core, default_persona, text)
        return web.json_response({"ok": ok, "error": "" if ok else "保存失败（首行须为 # 用户核心资料）"})

    async def api_persona_image(request):
        pid = request.match_info["pid"]
        persona = personas.get(pid)
        image = (persona or {}).get("ref_image")
        if not image:
            return web.Response(status=404)
        # 肖像可能被用户换图,禁缓存避免浏览器一直显示旧照片
        return web.FileResponse(image, headers={"Cache-Control": "no-cache, must-revalidate"})

    # ---- SQLite 对话历史 API ----
    async def history_page(_request):
        return web.FileResponse(REPO_ROOT / "web" / "history.html")

    async def api_history(_request):
        if history_store is None:
            return web.json_response({"enabled": False})
        conversations = await asyncio.to_thread(history_store.list_conversations, 100)
        return web.json_response({"enabled": True, "conversations": conversations})

    async def api_history_detail(request):
        if history_store is None:
            return web.json_response({"enabled": False})
        sid = request.match_info["sid"]
        messages = await asyncio.to_thread(history_store.get_messages, sid)
        return web.json_response({"session_id": sid, "messages": messages})

    async def api_history_delete(request):
        if history_store is None:
            return web.json_response({"ok": False, "error": "历史未启用"})
        sid = request.match_info["sid"]
        ok = await asyncio.to_thread(history_store.delete_conversation, sid)
        return web.json_response({"ok": ok, "error": "" if ok else "会话不存在"})

    async def api_yukino_task_done(request):
        """DSH 插件回调：把 DSH 任务完成总结写入雪乃当前会话。"
        请求：{"text": "xx项目，上一轮做xx的任务完成了。总结..."}
        动作：写入 SQLite 当前会话（角色 user），并实时推给浏览器显示。
        """
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "无效 JSON"}, status=400)

        body = body or {}
        raw_text = (body.get("text") or "").strip()
        if raw_text:
            project = Path(raw_text.split("项目，上一轮做")[0]).name if "项目，上一轮做" in raw_text else "默认项目"
            title = "DSH 任务"
            summary = raw_text
        else:
            project = str(body.get("project") or "默认项目").strip()
            project = Path(project).name or "默认项目"
            title = str(body.get("title") or "DSH 任务").strip()
            summary = (body.get("summary") or "").strip()
            raw_text = f"{project}项目，上一轮做「{title}」的任务完成了。"
            if summary:
                raw_text += f" {summary}"
        if not raw_text:
            return web.json_response({"ok": False, "error": "缺少 text/project"}, status=400)

        # 交给雪乃人设：生成【日语】+【译文】两段，和原项目 agent 格式一致
        ja_text, zh_text = await asyncio.to_thread(
            yukino_task_done_reply, project, title, summary, config
        )
        if not ja_text:
            ja_text = raw_text
            zh_text = ""

        tts_url = str((config.get("tts") or {}).get("server_url", "http://127.0.0.1:8899"))
        session = current_session.get("session")
        if session is not None and session.browser is not None:
            try:
                await session.browser.send_str(json.dumps({
                    "type": "vox.task-done",
                    "text": ja_text,
                    "translation": zh_text,
                }))
            except Exception as e:
                logger.info("推送 task-done 到浏览器失败（忽略）: %s", e)

            # 用真实雪乃音色合成日语正文并推送音频（前端只负责播放，口型照常驱动）
            try:
                pcm = await asyncio.to_thread(synthesize_task_done_audio, ja_text, tts_url)
                if pcm:
                    await session.browser.send_str(json.dumps({
                        "type": "vox.task-done-audio",
                        "pcm": base64.b64encode(pcm).decode(),
                    }))
            except Exception as e:
                logger.info("task-done 音频推送失败（忽略）: %s", e)

        if history_store is not None and session is not None:
            try:
                history_store.add_message(
                    session.session_id, "user", raw_text,
                    persona=session.persona_id,
                )
                history_store.add_message(
                    session.session_id, "assistant", ja_text,
                    translation=zh_text,
                    persona=session.persona_id,
                )
            except Exception as e:
                logger.info("写入 task-done 历史失败（忽略）: %s", e)

        return web.json_response({"ok": True, "session_id": session.session_id if session else None})

    # 单用户产品：新浏览器连接顶掉旧会话（换网络/僵尸会话不再需要刷新两次）
    current_session: dict = {"session": None}

    async def ws_handler(request):
        ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
        await ws.prepare(request)
        old = current_session["session"]
        if old is not None:
            logger.info("新连接到达，顶掉旧会话（释放管线槽位）")
            await old.close()
        session = Session(ws, s2s_url, avatar_url, personas, default_persona,
                          filler_enabled=filler_enabled,
                          avatar_backend=avatar_backend, memory_store=memory_store,
                          history_store=history_store,
                          weibo_cfg=config.get("weibo"))
        current_session["session"] = session
        try:
            await session.run()
        finally:
            if current_session["session"] is session:
                current_session["session"] = None
            if session.avatar is not None:
                await session.avatar.close()
        return ws

    app = web.Application(middlewares=[cors_middleware])
    app.router.add_get("/", index)
    app.router.add_get("/api/personas", api_personas)
    app.router.add_get("/api/personas/{pid}/image", api_persona_image)
    app.router.add_get("/api/memory", api_memory)
    app.router.add_route("*", "/api/memory/summary/{pid}", api_memory_summary)
    app.router.add_post("/api/memory/core", api_memory_core)
    app.router.add_get("/history", history_page)
    app.router.add_get("/api/history", api_history)
    app.router.add_get("/api/history/{sid}", api_history_detail)
    app.router.add_delete("/api/history/{sid}", api_history_delete)
    app.router.add_post("/api/yukino/task-done", api_yukino_task_done)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/static", REPO_ROOT / "web")
    # yukino2d 立绘/素材同源托管（2dlive 前端引擎 WebGL 纹理必须同源，
    # 直接指原目录，不复制文件；show_index 默认 False 不列目录）
    app.router.add_static("/yukino2d", REPO_ROOT / "yukino2d")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxEMW 编排入口（浏览器 ↔ s2s + avatar）")
    parser.add_argument("--config", default=os.environ.get("VOXEMW_CONFIG", "configs/assistant.yaml"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    from aiohttp import web

    from voxemw.config import load_config, load_dotenv

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    load_dotenv(REPO_ROOT / ".env.local")
    config = load_config(config_path)

    server = config.get("server") or {}
    host = str(server.get("host", "0.0.0.0"))
    port = int(server.get("port", 8000))
    logger.info("orchestrator 就绪: http://%s:%d", host, port)
    web.run_app(create_app(config), host=host, port=port, print=None)


if __name__ == "__main__":
    main()
