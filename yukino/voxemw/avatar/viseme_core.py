"""Viseme 口型贴片合成纯逻辑：画布布局 / 嘴部开合贴片 / 眨眼贴片 /
音频→开合度 / 眨眼调度（无 I/O、无线程，可单测）。

方案 A（《雪之下雪乃动漫口型方案研究.docx》）：不生成嘴部纹理——
从立绘裁剪嘴部区域，程序生成 8 级开合变体（口腔/牙齿/下唇压缩，
颜色全部从立绘采样），音量包络驱动开合度逐帧 crossfade；眼区生成
眨眼变体；头动/呼吸用剪切窗口偏移。画风 100% 保真（嘴就是立绘的嘴）。

所有坐标都在「画布坐标」：1280×720。立绘窗口 VIEW_RECT（原图坐标）
预缩放成 1280×720 底图（sheet），嘴/眼矩形换算到画布坐标（缩放后）。
"""

from __future__ import annotations

import math

import numpy as np

SAMPLE_RATE = 16000
FPS = 25
FRAMES_PER_CHUNK = 5
CHUNK_STEP = 3200                     # 0.2s（与 AVTR-1/THA3 对齐）
CHUNK_SECONDS = CHUNK_STEP / SAMPLE_RATE
OUT_W, OUT_H = 1280, 720

# 默认锚点（立绘原图坐标；prepare/calibrate 工具或 config 可覆盖）。
# 雪乃立绘 assets/yukino/ref.png（1279×2177）2026-08-09 用户网格校准：
#   嘴 (5,3) 格下边中间 → (711,840)；唇色检测 → (632,847)-(788,879)
#   左眼 (4,3) → (591,696)；右眼 (6,3) → (815,696)
VIEW_RECT = (259, 500, 1148, 1000)    # 16:9 特写窗口（脸居中、眼嘴完整）
# 嘴矩形下扩到下巴皮肤：开口腔体需要下带（下唇+皮肤）空间，纯唇色矩形
# （高 32px）放不下最大开口（原图 30px）
MOUTH_RECT = (632, 840, 788, 905)
EYE_LEFT_RECT = (541, 666, 640, 725)
EYE_RIGHT_RECT = (782, 666, 881, 725)

# 开合贴片
OPEN_MAX_PX = 30          # 最大开口（立绘原图 px）；口型幅度主旋钮
PATCH_COUNT = 8           # 开合阶梯级数（0..1 共 8 张）
LIP_THICK = 3             # 口腔带上下唇线厚度（px）
TEETH_FRAC = 0.22         # 口腔上部牙齿带占比
TAPER_POW = 4.0           # 口角收拢掩码幂次（中央全开、角上闭合）
CAVITY_DARKEN = 0.5       # 口腔底色 = 唇色 × (1-darken) + 暗棕
FEATHER_PX = 2            # 口腔带纵向羽化

# 音频 → 开合度
RMS_FLOOR = 0.003         # ~-50 dBFS
RMS_CEIL = 0.25           # ~-12 dBFS
CURVE_POWER = 1.3         # 弱音不张嘴、强音不开满
OPEN_MIN, OPEN_MAX = 0.05, 0.95
EMA_ATTACK = 0.45         # 开口速度（每帧）
EMA_RELEASE = 0.18        # 闭嘴惯性（每帧）

# 眨眼
BLINK_MIN, BLINK_MAX = 3.0, 7.0   # 随机间隔（秒）
BLINK_DURATION = 0.4              # 单次眨眼（快闭缓开）
BLINK_LEVELS = 12                 # 预生成闭眼级数

# 头动 / 呼吸 / 倾听
SWAY_AMP_X, SWAY_AMP_Y = 3.0, 2.0   # 摆头幅度（画布 px）
SWAY_PERIOD = 6.0
BREATH_PERIOD = 4.0
NOD_AMP = 3.0                       # 倾听点头（画布 px）
NOD_PERIOD = 0.6


# ── 坐标换算 ──

def view_to_canvas_scale(view_rect: tuple) -> float:
    """立绘窗口 → 画布的等比缩放系数（高度对齐 720）。"""
    _, _, _, y1 = view_rect
    return OUT_H / (y1 - view_rect[1])


def rect_to_canvas(rect: tuple, view_rect: tuple, scale: float) -> tuple[int, int, int, int]:
    """立绘矩形 → 画布坐标（整数化）。"""
    vx0, vy0, _, _ = view_rect
    x0, y0, x1, y1 = rect
    return (int(round((x0 - vx0) * scale)), int(round((y0 - vy0) * scale)),
            int(round((x1 - vx0) * scale)), int(round((y1 - vy0) * scale)))


# ── 颜色采样 ──

def sample_color(portrait_rgb: np.ndarray, rect: tuple, mask_fn=None) -> np.ndarray:
    """从立绘矩形区域取主色（中位数，mask_fn 可选过滤）。"""
    x0, y0, x1, y1 = rect
    region = portrait_rgb[y0:y1, x0:x1]
    if mask_fn is not None:
        region = region[mask_fn(region)]
        if len(region) == 0:
            region = portrait_rgb[y0:y1, x0:x1]
    return np.median(region.reshape(-1, 3), axis=0)


def is_lip_color(px: np.ndarray) -> np.ndarray:
    """唇色判定：r > g > b 且饱和度适中（动漫唇色）。"""
    r, g, b = px[..., 0].astype(int), px[..., 1].astype(int), px[..., 2].astype(int)
    return (r > g) & (g > b) & ((r - b) > 20)


# ── 嘴部开合贴片 ──

def generate_mouth_patches(sheet_rgb: np.ndarray, mouth_rect: tuple,
                           open_max_px: float = OPEN_MAX_PX,
                           count: int = PATCH_COUNT) -> list[np.ndarray]:
    """从底图生成 count 级开合贴片（画布坐标，uint8 RGB + alpha 4 通道）。

    patch[0] = 立绘原嘴（闭合）；patch[i] = 开口 o=i/(count-1)：
      上带（上唇）原样；口腔带（牙齿→口腔色渐变 + 口角收拢）；下带
      （下唇+下巴皮肤纵向压缩）。贴片外缘像素与底图逐字节一致（无缝），
      只有口腔带内部有内容差异，口角 taper 收拢成 alpha 渐变。
    """
    x0, y0, x1, y1 = mouth_rect
    w, h = x1 - x0, y1 - y0
    closed = sheet_rgb[y0:y1, x0:x1].astype(np.float64)

    # 唇色 / 口腔色 / 牙齿色采样（从闭口贴片）
    lip_mask = is_lip_color(closed)
    lip = closed[lip_mask] if lip_mask.sum() > 5 else closed.reshape(-1, 3)
    lip_color = np.median(lip, axis=0)
    skin_region = sheet_rgb[max(0, y0 - h):y0, x0:x1]
    skin = np.median(skin_region.reshape(-1, 3), axis=0) if len(skin_region) else lip_color
    cavity = lip_color * CAVITY_DARKEN + np.array([40, 24, 20]) * (1 - CAVITY_DARKEN)
    teeth = skin * 0.92 + np.array([255, 255, 255]) * 0.08

    # 唇线位置：唇色像素的垂直中心（上唇底部）
    if lip_mask.sum() > 5:
        ys = np.where(lip_mask.any(axis=1))[0]
        lip_line = int(np.median(ys))
    else:
        lip_line = h // 2

    # 口角收拢掩码：中央全开、两侧闭合（taper）
    xx = np.linspace(-1, 1, w)
    taper = (1 - np.abs(xx) ** TAPER_POW)[None, :]

    patches = [closed.astype(np.uint8).copy()]
    # 腔体高度上限：下带（下唇+皮肤）的 90%，防溢出贴片
    max_cavity = max(1, int((h - lip_line) * 0.9))
    for i in range(1, count):
        open_px = min(open_max_px, max_cavity) * i / (count - 1)
        cavity_h = max(1, int(round(open_px)))
        patches.append(_build_open_patch(closed, lip_line, cavity_h, cavity, teeth,
                                         taper, h, w, skin))
    # 统一转为 RGBA（alpha=255，口角 taper 不作用于贴片本身——由粘贴时的
    # 羽化 mask 处理；口腔带内部已是渐变内容）
    out = []
    for p in patches:
        alpha = np.full((p.shape[0], p.shape[1], 1), 255, dtype=np.uint8)
        out.append(np.concatenate([p, alpha], axis=2))
    return out


def _build_open_patch(closed, lip_line, cavity_h, cavity, teeth, taper, h, w, skin):
    """单级开合贴片：上带 + 口腔带 + 下带。"""
    upper = closed[:lip_line].copy()                 # 上唇固定
    # 下带：原图 [lip_line, h) 压缩到 (h - lip_line - cavity_h) 行
    lower_src = closed[lip_line:h]
    lower_h = max(1, h - lip_line - cavity_h)
    lower = _resize_rows(lower_src, lower_h)

    # 口腔带：上牙齿（TEETH_FRAC 比例）→ 口腔色，垂直羽化渐变
    teeth_h = max(1, int(cavity_h * TEETH_FRAC))
    body_h = max(1, cavity_h - teeth_h)
    cavity_rows = np.concatenate([
        np.tile(teeth[None, None, :], (teeth_h, 1, 1)),
        np.tile(cavity[None, None, :], (body_h, 1, 1)),
    ])
    # 垂直羽化：口腔带上下边缘柔化（FEATHER_PX 行余弦）
    cavity_rows = _feather_rows(cavity_rows, FEATHER_PX)
    # 口角收拢：两侧向口腔色渐隐（内容上收敛——边缘行保留原图嘴线）
    cavity_rows = cavity_rows * taper[:, :, None] + \
        np.tile(closed[lip_line][None, :, :], (cavity_h, 1, 1)) * (1 - taper[:, :, None])
    # 下唇线：口腔带底缘 2px 唇色线
    if lower_h > 0:
        line = np.tile(closed[max(0, lip_line - 1):lip_line], (min(2, lower_h), 1, 1))
        lower[:min(2, lower_h)] = line[:min(2, lower_h)]

    patch = np.concatenate([upper, cavity_rows, lower], axis=0)
    return np.clip(patch, 0, 255).astype(np.uint8)


def _resize_rows(src: np.ndarray, target_h: int) -> np.ndarray:
    """行方向 resize（PIL BILINEAR 语义，纯 numpy 实现）。"""
    if target_h == src.shape[0]:
        return src.copy()
    idx = np.linspace(0, src.shape[0] - 1, target_h)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, src.shape[0] - 1)
    frac = (idx - i0)[:, None, None]
    return (src[i0] * (1 - frac) + src[i1] * frac).astype(np.float64)


def _feather_rows(rows: np.ndarray, feather: int) -> np.ndarray:
    """纵向羽化：顶底 feather 行余弦渐变到 0。"""
    if feather <= 0 or rows.shape[0] <= feather * 2:
        return rows
    ramp = np.sin(np.linspace(0, np.pi / 2, feather))
    out = rows.copy().astype(np.float64)
    for i in range(feather):
        out[i] = rows[i] * ramp[i]
        out[-1 - i] = rows[-1 - i] * ramp[i]
    return out


def blend_patches(patches: list[np.ndarray], openness: float) -> np.ndarray:
    """开合度 → 相邻贴片加权混合（crossfade，RGBA uint8）。"""
    o = max(0.0, min(1.0, openness))
    pos = o * (len(patches) - 1)
    i0 = min(int(pos), len(patches) - 1)
    i1 = min(i0 + 1, len(patches) - 1)
    frac = pos - i0
    a = patches[i0].astype(np.float64)
    b = patches[i1].astype(np.float64)
    return np.clip(a * (1 - frac) + b * frac, 0, 255).astype(np.uint8)


# ── 眨眼贴片 ──

def generate_blink_patches(sheet_rgb: np.ndarray, eye_rect: tuple,
                           levels: int = BLINK_LEVELS) -> list[np.ndarray]:
    """眼区闭眼变体：垂直压缩 + 上眼睑肤色覆盖 + 眼睑线（RGBA uint8）。

    patches[0] = 睁眼原样；patches[k] = 闭眼度 p=k/(levels-1)。
    动漫式眨眼 = 上眼睑向下覆盖（眼区上部肤色带 + 下缘深色线）。
    """
    x0, y0, x1, y1 = eye_rect
    w, h = x1 - x0, y1 - y0
    open_eye = sheet_rgb[y0:y1, x0:x1].astype(np.float64)

    # 上眼睑肤色：眼区上方 60% 高度采样
    lid_region = sheet_rgb[max(0, y0 - h // 2):y0 + int(h * 0.4), x0:x1]
    lid_color = np.median(lid_region.reshape(-1, 3), axis=0) if len(lid_region) else \
        np.median(open_eye.reshape(-1, 3), axis=0)
    # 眼睑线颜色：眼区最暗像素
    flat = open_eye.reshape(-1, 3)
    line_color = flat[flat.sum(axis=1).argmin()]

    patches = [np.concatenate([open_eye.astype(np.uint8),
                               np.full((h, w, 1), 255, dtype=np.uint8)], axis=2)]
    for k in range(1, levels):
        p = k / (levels - 1)
        cover_h = int(round(h * 0.55 * p))       # 上眼睑下移
        squash = 1 - 0.5 * p                      # 剩余眼区压缩
        visible_h = max(1, int(round(h * squash)))
        # 下眼睑（剩余眼区）压缩到下方
        bottom = _resize_rows(open_eye, max(1, h - cover_h))[:visible_h]
        frame = np.full((h, w, 3), 255, dtype=np.float64)
        # 上眼睑覆盖带（肤色 + 底部眼睑线）
        lid = np.tile(lid_color[None, None, :], (cover_h, 1, 1))
        if cover_h > 0:
            lid[cover_h - 1] = line_color          # 闭合线
            lid[max(0, cover_h - 3):cover_h] *= 0.85
        frame[:cover_h] = lid
        frame[cover_h:cover_h + visible_h] = bottom
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        patches.append(np.concatenate([np.clip(frame, 0, 255).astype(np.uint8), alpha], axis=2))
    return patches


# ── 音频 → 开合度 ──

class OpennessTracker:
    """跨 chunk 的 EMA 开合度（reset 不清——静音自然回落闭口）。"""

    def __init__(self, rms_floor: float = RMS_FLOOR, rms_ceil: float = RMS_CEIL,
                 curve_power: float = CURVE_POWER, open_min: float = OPEN_MIN,
                 open_max: float = OPEN_MAX, ema_attack: float = EMA_ATTACK,
                 ema_release: float = EMA_RELEASE):
        self.rms_floor = rms_floor
        self.rms_ceil = rms_ceil
        self.curve_power = curve_power
        self.open_min = open_min
        self.open_max = open_max
        self.ema_attack = ema_attack
        self.ema_release = ema_release
        self.value = open_min

    def step(self, frame_audio: np.ndarray) -> float:
        """输入 640 采样（40ms）音频，返回开合度 0..1。"""
        rms = float(np.sqrt(np.mean(frame_audio ** 2)))
        if rms <= self.rms_floor:
            level = 0.0
        else:
            level = (math.log10(rms / self.rms_floor) /
                     math.log10(self.rms_ceil / self.rms_floor))
            level = max(0.0, min(1.0, level)) ** self.curve_power
        target = self.open_min + (self.open_max - self.open_min) * level
        alpha = self.ema_attack if target > self.value else self.ema_release
        self.value = alpha * target + (1 - alpha) * self.value
        return self.value


# ── 眨眼调度 ──

class BlinkMachine:
    """随机眨眼：间隔 [min,max] 秒，单次 BLINK_DURATION 快闭缓开。"""

    def __init__(self, rng=None, min_interval: float = BLINK_MIN,
                 max_interval: float = BLINK_MAX, duration: float = BLINK_DURATION):
        import random

        self._rng = rng or random.Random()
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.duration = duration
        self._next_at = self._rng.uniform(min_interval, max_interval)
        self._start = -1.0

    def update(self, t: float) -> float:
        """会话秒数 t → 闭眼度 0..1（0=睁眼）。"""
        if self._start < 0 and t >= self._next_at:
            self._start = t
        if self._start < 0:
            return 0.0
        elapsed = t - self._start
        if elapsed >= self.duration:
            self._start = -1.0
            self._next_at = t + self._rng.uniform(self.min_interval, self.max_interval)
            return 0.0
        close_frac = 0.35
        p = elapsed / self.duration
        if p < close_frac:
            return p / close_frac
        return 1.0 - (p - close_frac) / (1.0 - close_frac)
