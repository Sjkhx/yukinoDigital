"""Viseme 口型贴片合成引擎（动漫立绘说话头，纯 CPU/轻 GPU，零模型）。

接口与 AVTR1Engine / THA3Engine 完全同构（service.py ws 协议层无感知）：
feed_audio/feed_listen/reset/set_image/set_speech_active/set_idle_mode/close +
run_inference_loop(on_frames)/warmup(on_frames)，帧统一输出 1280×720 RGB uint8。

实现（方案 A，文档《雪之下雪乃动漫口型方案研究》）：
- 立绘窗口（VIEW_RECT，16:9 脸特写）预缩放成底图 sheet（外扩 8px 供头动）
- 嘴部：8 级开合贴片（口腔/牙齿/下唇压缩，颜色从立绘采样），
  音量包络（RMS→EMA）驱动开合度逐帧 crossfade——嘴就是立绘的嘴，
  从机制上不会像生成式模型那样画风崩坏/黑块
- 眼：12 级眨眼贴片（上眼睑覆盖），随机间隔 3-7s 快闭缓开
- 头动/呼吸：sheet 窗口正弦平移（±3px）；倾听（idle_mode=listening）
  用户音量 → 点头 + 注视
- 音频账本/门控与 THA3 引擎逐条一致（消费窗 3200 零前瞻、句尾补零
  排空、欠载停帧、idle 0.2s 节流、reset 保留运动上下文）

锚点来源：scripts/prepare_tha3_image.py 无参数模式生成网格参考图 →
用户报格子 → 写入 configs/assistant.yaml avatar.viseme（默认值即雪乃）。
"""

from __future__ import annotations

import logging
import math
import os
import threading
from pathlib import Path

import numpy as np

from voxemw.avatar.viseme_core import (
    SAMPLE_RATE,
    FPS,
    FRAMES_PER_CHUNK,
    CHUNK_STEP,
    CHUNK_SECONDS,
    OUT_W,
    OUT_H,
    VIEW_RECT,
    MOUTH_RECT,
    EYE_LEFT_RECT,
    EYE_RIGHT_RECT,
    OpennessTracker,
    BlinkMachine,
    generate_mouth_patches,
    generate_blink_patches,
    blend_patches,
    view_to_canvas_scale,
    rect_to_canvas,
)

logger = logging.getLogger(__name__)

WARMUP_CHUNKS = 2
LISTEN_CAP = SAMPLE_RATE * 8
PAD = 8                        # sheet 外扩（画布 px，头动幅度上限）
CANVAS_BG = (24, 26, 32)       # 画布左右填充色（立绘窗口铺满高度时无外露）


class VisemeEngine:
    """口型贴片合成引擎：pipeline 调用序列化在 inference 线程。"""

    def __init__(self, image_path: str, avatar_cfg: dict | None = None):
        avatar_cfg = avatar_cfg or {}
        v = avatar_cfg.get("viseme") or {}
        # 锚点：配置 > 模块默认（雪乃网格校准值）
        self._view_rect = tuple(v.get("view_rect", VIEW_RECT))
        self._mouth_rect = tuple(v.get("mouth_rect", MOUTH_RECT))
        self._eye_left = tuple(v.get("eye_left_rect", EYE_LEFT_RECT))
        self._eye_right = tuple(v.get("eye_right_rect", EYE_RIGHT_RECT))
        self._bg = tuple(v.get("bg_color", CANVAS_BG))

        self._openness = OpennessTracker(
            rms_floor=float(v.get("rms_floor", 0.003)),
            rms_ceil=float(v.get("rms_ceil", 0.25)),
            curve_power=float(v.get("curve_power", 1.3)),
            open_min=float(v.get("mouth_floor", 0.05)),
            open_max=float(v.get("mouth_peak", 0.95)),
            ema_attack=float(v.get("ema_attack", 0.45)),
            ema_release=float(v.get("ema_release", 0.18)),
        )
        blink_i = v.get("blink_interval", [3, 7])
        self._blink = BlinkMachine(
            min_interval=float(blink_i[0]), max_interval=float(blink_i[1]),
            duration=float(v.get("blink_duration", 0.4)))
        self._open_max_px = float(v.get("open_max_px", 30))
        self._sway_amp_x = float(v.get("sway_amp_x", 3.0))
        self._sway_amp_y = float(v.get("sway_amp_y", 2.0))
        self._nod_amp = float(v.get("nod_amp", 3.0))
        self.idle_motion = bool(avatar_cfg.get("idle_motion", True))

        self.on_frames = None
        self._cond = threading.Condition()
        self._closed = False
        self._pending_image = None
        self._speech_active = False
        self._idle_mode = "calm"
        # 音频账本（与 AVTR-1/THA3 同构）
        self._buf = np.empty(0, dtype=np.float32)
        self._pos = 0
        self._real_len = 0
        self._listen = np.empty(0, dtype=np.float32)
        self._listen_gate = 0.0
        self._t = 0.0            # 会话秒数（头动相位/眨眼基准）

        self._build_assets(image_path)

    # ── 内部 ──

    def _build_assets(self, image_path: str) -> None:
        """立绘 → sheet + 嘴/眼贴片（全部预生成，毫秒级）。"""
        from PIL import Image

        logger.info("构建 viseme 资产: %s", image_path)
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        vx0, vy0, vx1, vy1 = self._view_rect
        # 外扩 PAD（画布 px）→ 立绘 px
        scale = view_to_canvas_scale(self._view_rect)
        pad_px = int(math.ceil(PAD / scale))
        crop = img.crop((max(0, vx0 - pad_px), max(0, vy0 - pad_px),
                         min(w, vx1 + pad_px), min(h, vy1 + pad_px)))
        # 目标 = (OUT_W + 2*PAD) × (OUT_H + 2*PAD)
        self._sheet = np.asarray(
            crop.resize((OUT_W + 2 * PAD, OUT_H + 2 * PAD), Image.LANCZOS),
            dtype=np.uint8)
        # 贴片画布坐标（含 PAD 边框）
        self._m_canvas = rect_to_canvas(self._mouth_rect, self._view_rect, scale)
        self._e_canvas = (rect_to_canvas(self._eye_left, self._view_rect, scale),
                          rect_to_canvas(self._eye_right, self._view_rect, scale))
        # 嘴贴片 / 眼贴片在 sheet 坐标（+PAD）
        off = PAD
        m = (self._m_canvas[0] + off, self._m_canvas[1] + off,
             self._m_canvas[2] + off, self._m_canvas[3] + off)
        self._mouth_patches = generate_mouth_patches(self._sheet, m,
                                                     self._open_max_px * scale)
        self._blink_patches = [
            generate_blink_patches(self._sheet, (e[0] + off, e[1] + off,
                                                 e[2] + off, e[3] + off))
            for e in self._e_canvas
        ]
        self._image_path = image_path

    def _render_frame(self, audio_640: np.ndarray, t: float) -> np.ndarray:
        """40ms 音频 + 会话时刻 → 1280×720 RGB uint8。"""
        # 头动：正弦平移（sheet 内裁窗）+ 倾听点头
        dx = PAD + int(round(self._sway_amp_x * math.sin(2 * math.pi * t / 6.0)))
        dy = PAD + int(round(self._sway_amp_y * math.sin(2 * math.pi * t / 4.0)))
        if self._listen_gate > 0:
            dy += int(round(-self._nod_amp * self._listen_gate *
                            math.sin(2 * math.pi * 0.8 * t)))
        frame = self._sheet[dy:dy + OUT_H, dx:dx + OUT_W].copy()

        # 嘴：开合度 → crossfade 贴片 → 粘贴（贴片外缘与 sheet 无缝）
        openness = self._openness.step(audio_640)
        patch = blend_patches(self._mouth_patches, openness)
        mx0, my0 = self._m_canvas[0] + PAD - dx, self._m_canvas[1] + PAD - dy
        frame[my0:my0 + patch.shape[0], mx0:mx0 + patch.shape[1]] = patch[..., :3]

        # 眼：眨眼贴片粘贴
        wink = self._blink.update(t)
        if wink > 0:
            for (e0, e1, e2, e3), patches in zip(self._e_canvas, self._blink_patches):
                idx = min(int(wink * (len(patches) - 1)), len(patches) - 1)
                bp = patches[idx]
                px0, py0 = e0 + PAD - dx, e1 + PAD - dy
                frame[py0:py0 + bp.shape[0], px0:px0 + bp.shape[1]] = bp[..., :3]
        return frame

    def _listen_window(self):
        if self._idle_mode != "listening" or len(self._listen) == 0:
            return np.zeros(CHUNK_STEP, dtype=np.float32)
        tail = self._listen[-CHUNK_STEP:]
        if len(tail) < CHUNK_STEP:
            tail = np.pad(tail, (CHUNK_STEP - len(tail), 0))
        return tail

    # ── 生产侧（ws 线程调用）──

    def feed_audio(self, pcm_f32) -> None:
        with self._cond:
            self._buf = np.concatenate([self._buf[: self._real_len], pcm_f32])
            self._real_len += len(pcm_f32)
            self._cond.notify()

    def reset(self) -> None:
        """打断：只清音频缓冲（运动上下文保留——开合度/眨眼状态不重置）。"""
        with self._cond:
            self._buf = np.empty(0, dtype=np.float32)
            self._pos = 0
            self._real_len = 0
            self._cond.notify()

    def set_image(self, image_path: str) -> None:
        with self._cond:
            self._pending_image = image_path
            self._cond.notify()

    def set_speech_active(self, on: bool) -> None:
        with self._cond:
            self._speech_active = on
            self._cond.notify()

    def feed_listen(self, pcm_f32) -> None:
        with self._cond:
            self._listen = np.concatenate([self._listen, pcm_f32])[-LISTEN_CAP:]
            self._cond.notify()

    def set_idle_mode(self, mode: str) -> None:
        with self._cond:
            self._idle_mode = mode
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify()

    # ── 消费侧（inference 线程）──

    def run_inference_loop(self, on_frames) -> None:
        """阻塞循环（与 THA3 引擎同构，消费窗 3200 零前瞻）。"""
        import time as _time

        last_idle_at = 0.0
        while True:
            with self._cond:
                while not self._closed:
                    unconsumed = self._real_len - self._pos
                    buffered = len(self._buf) - self._pos
                    if buffered >= CHUNK_STEP:
                        break
                    if 0 < unconsumed < CHUNK_STEP and not self._speech_active:
                        self._buf = np.concatenate([
                            self._buf,
                            np.zeros(CHUNK_STEP - buffered, dtype=np.float32)])
                        continue
                    if unconsumed == 0 and self.idle_motion and not self._speech_active:
                        wait = last_idle_at + CHUNK_SECONDS - _time.monotonic()
                        if wait > 0:
                            self._cond.wait(timeout=wait)
                            continue
                        break
                    self._cond.wait(timeout=0.5)
                if self._closed:
                    return
                if self._pending_image:
                    self._build_assets(self._pending_image)
                    self._pending_image = None
                is_idle = (self._real_len - self._pos) == 0
                if is_idle:
                    audio = np.zeros(CHUNK_STEP, dtype=np.float32)
                    last_idle_at = _time.monotonic()
                else:
                    audio = self._buf[self._pos:self._pos + CHUNK_STEP]
                    self._pos += CHUNK_STEP
                    if self._pos > 0:
                        self._buf = self._buf[self._pos:]
                        self._real_len = max(0, self._real_len - self._pos)
                        self._pos = 0
            # 倾听包络（listen 轨音量 → 点头幅度）
            if self._idle_mode == "listening":
                lrms = float(np.sqrt(np.mean(self._listen_window() ** 2)))
                self._listen_gate = 0.15 * (1.0 if lrms > 0.01 else 0.0) + \
                    0.85 * self._listen_gate
            else:
                self._listen_gate = 0.0
            frames = np.stack([
                self._render_frame(audio[i * 640:(i + 1) * 640],
                                   self._t + i / FPS)
                for i in range(FRAMES_PER_CHUNK)
            ])
            self._t += CHUNK_SECONDS
            on_frames(frames, is_idle)

    def warmup(self, on_frames) -> None:
        """静音跑 WARMUP_CHUNKS+1 chunk（贴片合成无模型预热，纯走账本对齐）。"""
        import time as _time

        logger.info("Viseme 预热（静音 2 chunk）...")
        self.feed_audio(np.zeros(CHUNK_STEP * (WARMUP_CHUNKS + 1), dtype=np.float32))
        while True:
            with self._cond:
                done = self._real_len - self._pos <= 0
            if done:
                break
            _time.sleep(0.1)
        self.reset()
        logger.info("Viseme 预热完成")
