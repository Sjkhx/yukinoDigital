"""THA3 数字人引擎（Talking Head Anime 3，动漫参数化说话头，PyTorch）。

接口与 AVTR1Engine 完全同构（service.py ws 协议层无感知）：
feed_audio/feed_listen/reset/set_image/set_speech_active/set_idle_mode/close +
run_inference_loop(on_frames)/warmup(on_frames)，帧统一输出 1280×720 RGB uint8
（THA3 输出 512×512 RGBA → 等比 720×720 → 深色画布 16:9 中央）。

设计参考《雪之下雪乃动漫口型方案研究.docx》方案 B：动漫立绘 + 45 维姿态参数
（头部旋转/五元音口型/眨眼/眉毛/虹膜/呼吸）→ 全脸动画帧。音频 → 音量包络 →
mouth_aaa/mouth_delta（五元音分类留二期，vowel_detect 配置位）；
随机眨眼 + 呼吸正弦 + 低频头动替代 AVTR-1 的静音驱动微动；
listen 轨（idle_mode=="listening"）驱动 iris 注视 + 点头微表情。

流式语义对齐 AVTR-1（2026-08-09）：
- chunk 步进 0.2s（3200 采样）产 5 帧，输入窗口 3200（THA3 零模型前瞻——
  AVTR-1 的 6480 前瞻窗不需要），稳态供帧天然落后音频 ~0（+GPU 渲染 ~10ms），
  前端 AVATAR_AUDIO_DELAY 用 0.2s（orchestrator 下发 avatar_backend 选择）。
- reset() 只清音频缓冲、保留运动上下文（开合度 EMA/眨眼计时/微动相位）：
  打断后静音 chunk 让嘴自然闭合，无跳变。
- 句中欠载（speech_active=true 而缓冲不足一个窗口）：停帧等待，不补零。
- 句尾（speech_active 转 false）仍有未消费真音频：立即右补零排空（帧标
  speech），嘴型自然闭合。THA3 无模型输入，不需要 AVTR-1 的 TAIL_FADE
  （淡出是给 TRT 模型防急回中性位用的；音量包络自身 EMA 平滑）。
- 无活动语音段：静音 chunk 持续渲染（0.2s 实时节流），帧标 idle。

运行环境：项目 .venv（torch cu128，WSL2），
本地 RTX 5080（sm_120）实时 ~10-30ms/帧。权重来源（按序）：
配置 tha3_storage > 项目 data/models/<variant> > tha3 pip 包内置
（34j/tha3 wheel 自带 separable_float，国内 pip 镜像装包即得，无需外网）；
scripts/fetch_tha3_models.py --from-package 一键复制到位。
variant 对应 tha3.poser.modes.* 的 create_poser，module_file_names 显式传路径。
"""

from __future__ import annotations

import logging
import math
import os
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FPS = 25
FRAMES_PER_CHUNK = 5
CHUNK_STEP = 3200                                # 0.2s（5 帧 × 640）
CHUNK_SECONDS = CHUNK_STEP / SAMPLE_RATE         # 0.2
OUT_H, OUT_W = 720, 1280
THA3_SIZE = 512                                  # 官方模型输入/输出分辨率
CANVAS_SIDE = 720                                # THA3 输出等比放大后的贴图边长
CANVAS_X = (OUT_W - CANVAS_SIDE) // 2            # 1280×720 画布中央贴图起点 x
MODEL_FILES = [
    "editor.pt",
    "eyebrow_decomposer.pt",
    "eyebrow_morphing_combiner.pt",
    "face_morpher.pt",
    "two_algo_face_body_rotator.pt",
]

WARMUP_CHUNKS = 2
LISTEN_CAP = SAMPLE_RATE * 8    # listen 环形缓冲上限（最近 8s 用户语音）

# variant → create_poser 所在模块（官方 demo 与 34j pip 包结构一致）
POSER_MODULES = {
    "standard_float": "tha3.poser.modes.standard_float",
    "standard_half": "tha3.poser.modes.standard_half",
    "separable_float": "tha3.poser.modes.separable_float",
    "separable_half": "tha3.poser.modes.separable_half",
}

# ── 45 维参数布局（tha3.poser.modes.standard_float.get_pose_parameters，官方顺序）──
# 每组 (name, index)；双眼参数列 L,R 两个 index。
PARAM = {
    # 眉毛（0..11，范围 0..1）
    "eyebrow_troubled": (0, 1), "eyebrow_angry": (2, 3), "eyebrow_lowered": (4, 5),
    "eyebrow_raised": (6, 7), "eyebrow_happy": (8, 9), "eyebrow_serious": (10, 11),
    # 眼（12..23，0..1）
    "eye_wink": (12, 13), "eye_happy_wink": (14, 15), "eye_surprised": (16, 17),
    "eye_relaxed": (18, 19), "eye_unimpressed": (20, 21),
    "eye_raised_lower_eyelid": (22, 23),
    # 虹膜 morph（24..25，0..1）
    "iris_small": (24, 25),
    # 口（26..36；aaa 默认 1.0，即中立口型）
    "mouth_aaa": 26, "mouth_iii": 27, "mouth_uuu": 28,
    "mouth_eee": 29, "mouth_ooo": 30, "mouth_delta": 31,
    "mouth_lowered_corner": (32, 33), "mouth_raised_corner": (34, 35),
    "mouth_smirk": 36,
    # 虹膜旋转（37..38，±1）
    "iris_rotation_x": 37, "iris_rotation_y": 38,
    # 头/身/呼吸（39..44；39-43 为 ±1，44 为 0..1）
    "head_x": 39, "head_y": 40, "neck_z": 41, "body_y": 42, "body_z": 43,
    "breathing": 44,
}


# ── 纯函数：音频 → 45 维参数（无 torch 依赖，可单测）──

def volume_to_mouth(rms: float, rms_floor: float = 0.003,
                    rms_peak: float = 0.25, k: float = 0.9,
                    curve_power: float = 1.3) -> float:
    """RMS → 开合度 [0,1]。dB 线性化 + 幂曲线：弱音不张嘴、强音不开满。

    rms_floor/rms_peak 是 dB 映射的端点（-50dBFS ~ -12dBFS 默认）；
    k 是总体缩放（防开满）；curve_power>1 压低弱音、抬升强音。"""
    if rms <= rms_floor:
        return 0.0
    level = math.log10(rms / rms_floor) / math.log10(rms_peak / rms_floor)
    return max(0.0, min(1.0, level)) ** curve_power * k


def build_pose_vector(mouth_open: float = 0.0, mouth_scale: float = 0.8,
                      wink: float = 0.0, breathing: float = 0.5,
                      head_x: float = 0.0, head_y: float = 0.0,
                      neck_z: float = 0.0, iris_x: float = 0.0,
                      iris_y: float = 0.0) -> np.ndarray:
    """组装 45 维参数向量（顺序与官方 get_pose_parameters 一致）。

    mouth_open 0..1 → mouth_aaa（开口）；mouth_scale 0..1 → mouth_delta
    （嘴整体大小，默认 0.8 略收）；wink 0..1 → eye_wink 双眼同步；
    breathing 0..1；head_x/head_y/neck_z/iris_x/iris_y 按官方 ±1 范围直接传。"""
    pose = np.zeros(45, dtype=np.float32)
    pose[PARAM["mouth_aaa"]] = float(np.clip(mouth_open, 0.0, 1.0))
    pose[PARAM["mouth_delta"]] = float(np.clip(mouth_scale, 0.0, 1.0))
    l, r = PARAM["eye_wink"]
    pose[l] = pose[r] = float(np.clip(wink, 0.0, 1.0))
    pose[PARAM["breathing"]] = float(np.clip(breathing, 0.0, 1.0))
    pose[PARAM["head_x"]] = float(head_x)
    pose[PARAM["head_y"]] = float(head_y)
    pose[PARAM["neck_z"]] = float(neck_z)
    pose[PARAM["iris_rotation_x"]] = float(iris_x)
    pose[PARAM["iris_rotation_y"]] = float(iris_y)
    return pose


def blink_state(elapsed_since_start: float, duration: float = 0.4) -> float:
    """眨眼相位：快闭（前 35% 时长 0→1）缓开（后 65% 1→0），返回闭眼度 0..1。

    用时长为 0 或负（未在眨眼）返回 0。"""
    if elapsed_since_start < 0 or elapsed_since_start >= duration:
        return 0.0
    close_frac = 0.35  # 前 35% 闭合
    p = elapsed_since_start / duration
    if p < close_frac:
        return p / close_frac
    return 1.0 - (p - close_frac) / (1.0 - close_frac)


# ── 引擎 ──

class THA3Engine:
    """THA3 推理引擎：所有 pipeline 调用序列化在 inference 线程。"""

    def __init__(self, image_path: str, avatar_cfg: dict | None = None):
        import torch

        avatar_cfg = avatar_cfg or {}
        tha3 = avatar_cfg.get("tha3") or {}
        variant = str(tha3.get("variant", "separable_float"))
        if variant not in POSER_MODULES:
            raise RuntimeError(f"未知 THA3 variant={variant!r}，可选: {list(POSER_MODULES)}")

        # 模型目录：tha3_storage 配置 > 项目 data/models > tha3 pip 包内置
        # （34j/tha3 wheel 自带 separable_float 权重——国内 pip 镜像装包即得模型）
        storage = avatar_cfg.get("tha3_storage")
        model_dir = None
        if storage:
            candidate = Path(os.path.expanduser(str(storage))) / variant
            if all((candidate / f).is_file() for f in MODEL_FILES):
                model_dir = candidate
        if model_dir is None:
            proj_dir = Path(__file__).resolve().parent.parent.parent / "data" / "models" / variant
            if all((proj_dir / f).is_file() for f in MODEL_FILES):
                model_dir = proj_dir
        if model_dir is None:
            import importlib.util

            spec = importlib.util.find_spec("tha3")
            if spec is not None and spec.origin:
                # spec.origin 是 tha3/__init__.py → 包根 = parent
                pkg_dir = Path(spec.origin).parent / "data" / "models" / variant
                if all((pkg_dir / f).is_file() for f in MODEL_FILES):
                    model_dir = pkg_dir
        if model_dir is None:
            raise RuntimeError(
                f"THA3 模型缺失（variant={variant}）。最快路径：pip install tha3"
                f"（wheel 自带 {variant} 权重）后运行"
                f" scripts/fetch_tha3_models.py --from-package；或下载 5 个 .pt"
                f" 到 data/models/{variant}/（scripts/fetch_tha3_models.py）。")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("加载 THA3 poser（%s, %s, %s）: %s",
                    device, torch.__version__, variant, model_dir)
        import importlib

        poser_mod = importlib.import_module(POSER_MODULES[variant])
        self._poser = poser_mod.create_poser(
            device,
            module_file_names={Path(f).stem: str(model_dir / f) for f in MODEL_FILES},
        )
        self._device = torch.device(device)

        # 口型/微动参数（全部可配，见 configs/assistant.yaml avatar.tha3）
        self._cfg = {
            "rms_floor": float(tha3.get("rms_floor", 0.003)),
            "rms_peak": float(tha3.get("rms_peak", 0.25)),
            "mouth_k": float(tha3.get("mouth_k", 0.9)),
            "curve_power": float(tha3.get("curve_power", 1.3)),
            "mouth_floor": float(tha3.get("mouth_floor", 0.05)),   # 音量→嘴开合度下限
            "mouth_peak": float(tha3.get("mouth_peak", 0.95)),     # 上限
            "smooth": float(tha3.get("smooth", 0.3)),              # 开合度 EMA（每 chunk）
            "blink_min": float(tha3.get("blink_interval", [3, 7])[0]),
            "blink_max": float(tha3.get("blink_interval", [3, 7])[1]),
            "blink_duration": float(tha3.get("blink_duration", 0.4)),
            "head_motion": bool(tha3.get("head_motion", True)),
            "bg_color": tuple(tha3.get("bg_color", [24, 26, 32])),
            "vowel_detect": bool(tha3.get("vowel_detect", False)),  # 二期预留
        }

        self.idle_motion = bool(avatar_cfg.get("idle_motion", True))
        self.on_frames = None

        # 音频流账本（与 avtr1_engine 同构：_buf 未裁剪采样流，_pos 已消费帧边界，
        # _real_len 真实音频长度——补零不算；feed 时丢弃未消费补零段防漂移）
        self._buf = np.empty(0, dtype=np.float32)
        self._pos = 0
        self._real_len = 0
        self._listen = np.empty(0, dtype=np.float32)  # 用户麦克风环形缓冲

        # 运动上下文（reset 保留；仅 set_image 换肖像时保留——参数与肖像无关，
        # 但与 AVTR-1 的"换图冷启动"不同，这里不重置，嘴自然过渡）
        self._openness = 0.0          # 开合度 EMA（跨 chunk 持久）
        self._t = 0.0                 # 会话秒数（微动/眨眼相位基准）
        self._next_blink_at = self._t + self._rand(3.0, 7.0)
        self._blink_start = -1.0
        self._listen_gate = 0.0       # 倾听包络 EMA（0..1）

        self._cond = threading.Condition()
        self._closed = False
        self._pending_image = None
        self._speech_active = False
        self._idle_mode = "calm"      # listening 时 listen 轨生效，其余静音

        # 立绘：avatar.tha3.image 优先（512 胸像，prepare_tha3_image.py 生成），
        # fallback 到 service 传入的 persona ref_image（全身立绘也能跑，构图略差）
        image_cfg = tha3.get("image")
        if image_cfg:
            p = Path(os.path.expanduser(str(image_cfg)))
            if not p.is_absolute():
                p = Path(__file__).resolve().parent.parent.parent / p
            if p.is_file():
                image_path = str(p)
        self._load_image(image_path)

    # ── 内部 ──

    def _rand(self, lo: float, hi: float) -> float:
        import random
        return random.uniform(lo, hi)

    def _load_image(self, image_path: str) -> None:
        """立绘 → (1,4,512,512) RGBA [-1,1] tensor（THA3 输入）。

        官方 extract_pytorch_image_from_PIL_image 是 scale=2/offset=-1
        （[0,1]→[-1,1]，alpha 通道同样映射：透明 0→-1、不透明 1→+1）。
        之前喂 [0,1] 导致输出泛白/低对比（2026-08-09 修复）。"""
        from PIL import Image
        import torch

        logger.info("加载 THA3 立绘: %s", image_path)
        img = Image.open(image_path).convert("RGBA").resize(
            (THA3_SIZE, THA3_SIZE), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0 * 2.0 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        self._image = tensor.to(self._device)
        self._image_path = image_path

    # ── 生产侧（ws 线程调用）──

    def feed_audio(self, pcm_f32) -> None:
        with self._cond:
            # 句尾补零是投机性的：新真音频到达时丢弃未消费补零段再追加
            self._buf = np.concatenate([self._buf[: self._real_len], pcm_f32])
            self._real_len += len(pcm_f32)
            self._cond.notify()

    def reset(self) -> None:
        """打断：只清音频缓冲（运动上下文保留——静音 chunk 让嘴自然闭合）。"""
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
        """用户麦克风音频（listen 轨）。环形保留最近 LISTEN_CAP 采样。"""
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

    def _listen_window(self):
        """当前 chunk 的 listen 轨：idle_mode=="listening" 时取环形缓冲末尾
        CHUNK_STEP（不足左补零），否则纯静音。"""
        if self._idle_mode != "listening" or len(self._listen) == 0:
            return np.zeros(CHUNK_STEP, dtype=np.float32)
        tail = self._listen[-CHUNK_STEP:]
        if len(tail) < CHUNK_STEP:
            tail = np.pad(tail, (CHUNK_STEP - len(tail), 0))
        return tail

    def _update_motion(self, audio: np.ndarray, t: float) -> np.ndarray:
        """音频 chunk + 会话时刻 → 45 维参数向量（每 chunk 一次，帧间不插值：
        25fps 下 chunk 内 5 帧共享参数即可——嘴型随 0.2s 包络节奏走）。

        - 音量包络：chunk RMS → volume_to_mouth → EMA 平滑（attack/release 同
          alpha，开口快闭口慢由 mouth_floor 下限托底）
        - 眨眼：随机间隔 3-7s，400ms 快闭缓开（blink_state）
        - 呼吸/头动/注视：低频正弦（idle 幅度，说话减半）
        - 倾听（idle_mode=listening）：listen RMS → iris 注视 + 点头"""
        cfg = self._cfg

        # 音量 → 开合度（目标值 + EMA）
        rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
        target = volume_to_mouth(rms, cfg["rms_floor"], cfg["rms_peak"],
                                 cfg["mouth_k"], cfg["curve_power"])
        target = cfg["mouth_floor"] + (cfg["mouth_peak"] - cfg["mouth_floor"]) * target
        alpha = cfg["smooth"]
        self._openness = alpha * target + (1 - alpha) * self._openness
        openness = self._openness

        # 眨眼计时。注意完成判断用 elapsed >= duration 而非 wink <= 0：
        # 触发当 chunk elapsed=0 → blink_state=0，若用 0 判完成会立刻重置，
        # 眨眼永远闪不出来。
        if self._blink_start < 0 and t >= self._next_blink_at:
            self._blink_start = t
        wink = 0.0
        if self._blink_start >= 0:
            elapsed = t - self._blink_start
            wink = blink_state(elapsed, cfg["blink_duration"])
            if elapsed >= cfg["blink_duration"]:
                self._blink_start = -1.0
                self._next_blink_at = t + self._rand(cfg["blink_min"], cfg["blink_max"])

        # 倾听包络（listen 轨 RMS，EMA 平滑）
        if self._idle_mode == "listening":
            lrms = float(np.sqrt(np.mean(self._listen_window() ** 2)))
            gate = 1.0 if lrms > 0.01 else 0.0
            self._listen_gate = 0.15 * gate + 0.85 * self._listen_gate
        else:
            self._listen_gate = 0.0
        lg = self._listen_gate

        # 微动幅度：说话时减半（嘴动为主，头动过多抢注意力）
        amp = 1.0 if self._idle_mode == "calm" or lg > 0 else 0.5
        if not cfg["head_motion"]:
            amp = 0.0
        head_x = 0.06 * amp * math.sin(2 * math.pi * t / 7.3)
        head_y = 0.05 * amp * math.sin(2 * math.pi * t / 5.1) + 0.04 * lg * math.sin(
            2 * math.pi * 0.8 * t)          # 倾听点头叠加
        neck_z = 0.03 * amp * math.sin(2 * math.pi * t / 8.7)
        breath = 0.5 + 0.3 * math.sin(2 * math.pi * 0.22 * t)
        iris_x = 0.06 * amp * math.sin(2 * math.pi * t / 11.0)
        iris_y = 0.3 * lg + 0.05 * amp * math.sin(2 * math.pi * t / 9.0)

        return build_pose_vector(
            mouth_open=openness,
            mouth_scale=0.8 + 0.2 * openness,
            wink=wink, breathing=breath,
            head_x=head_x, head_y=head_y, neck_z=neck_z,
            iris_x=iris_x, iris_y=iris_y,
        )

    def _render_frame(self, pose_vec: np.ndarray) -> np.ndarray:
        """pose → THA3 推理 → 1280×720 RGB uint8。

        输出 [-1,1] → (x+1)/2 即可（A/B 实测：输入 [-1,1] 时恒等 pose 重建
        与原图无色差；官方 app 额外做的 linear→sRGB 在 separable_float 上
        反而会洗白画面，2026-08-09 实测弃用）。alpha 同样 [-1,1]，按 alpha
        合成到背景色（低 alpha 区域不产生黑块）。"""
        import cv2
        import torch

        pose = torch.from_numpy(pose_vec).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            out = self._poser.pose(self._image, pose)[0]
        arr = (out.cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0  # [-1,1] → [0,1]
        rgb = np.clip(arr[..., :3], 0.0, 1.0)
        alpha = np.clip(arr[..., 3:4], 0.0, 1.0)
        # alpha 合成到背景色（低 alpha 区域不产生黑块）
        bg = np.array(self._cfg["bg_color"], dtype=np.float64) / 255.0
        comp = rgb * alpha + bg * (1.0 - alpha)
        comp = np.clip(comp, 0.0, 1.0)
        rgb = (comp * 255.0).astype(np.uint8)

        # 512 → 720 等比放大，贴 1280×720 画布中央（左右填背景色）
        canvas = np.full((OUT_H, OUT_W, 3), self._cfg["bg_color"], dtype=np.uint8)
        face = cv2.resize(rgb, (CANVAS_SIDE, CANVAS_SIDE),
                          interpolation=cv2.INTER_LINEAR)
        canvas[:, CANVAS_X:CANVAS_X + CANVAS_SIDE] = face
        return canvas

    def run_inference_loop(self, on_frames) -> None:
        """阻塞循环（与 avtr1_engine 同构，消费窗口 = CHUNK_STEP 无前瞻）：
        - 真实音频攒满 3200 即生成（标 speech）；
        - 段结束（speech_active=false）仍有真音频尾巴：立即右补零排空；
        - 句中欠载（speech_active=true 缓冲不足）：停帧等新音频，不补零；
        - 无活动段（缓冲全空且 speech_active=false）：静音 idle chunk 按
          0.2s 实时节流（标 idle）。
        on_frames(frames_uint8: np.ndarray (5,720,1280,3), is_idle: bool)。"""
        import time as _time

        last_idle_at = 0.0
        while True:
            with self._cond:
                while not self._closed:
                    unconsumed = self._real_len - self._pos      # 真音频余量
                    buffered = len(self._buf) - self._pos        # 含补零
                    if buffered >= CHUNK_STEP:
                        break  # 可生成
                    if 0 < unconsumed < CHUNK_STEP and not self._speech_active:
                        # 段结束尾巴：立即右补零排空（无淡出——音量包络自带平滑）
                        self._buf = np.concatenate([
                            self._buf,
                            np.zeros(CHUNK_STEP - buffered, dtype=np.float32)])
                        continue
                    if unconsumed == 0 and self.idle_motion and not self._speech_active:
                        wait = last_idle_at + CHUNK_SECONDS - _time.monotonic()
                        if wait > 0:
                            self._cond.wait(timeout=wait)
                            continue  # 唤醒/超时后重查（真音频优先）
                        break
                    # 句中欠载 / 说话期间禁 idle：等状态变化（0.5s 兜底自醒）
                    self._cond.wait(timeout=0.5)
                if self._closed:
                    return
                if self._pending_image:
                    self._load_image(self._pending_image)
                    self._pending_image = None
                is_idle = (self._real_len - self._pos) == 0
                if is_idle:
                    audio = np.zeros(CHUNK_STEP, dtype=np.float32)
                    last_idle_at = _time.monotonic()
                else:
                    audio = self._buf[self._pos : self._pos + CHUNK_STEP]
                    self._pos += CHUNK_STEP
                    if self._pos > 0:
                        self._buf = self._buf[self._pos :]
                        self._real_len = max(0, self._real_len - self._pos)
                        self._pos = 0
            pose_vec = self._update_motion(audio, self._t)
            frames = np.stack([self._render_frame(pose_vec)
                               for _ in range(FRAMES_PER_CHUNK)])
            self._t += CHUNK_SECONDS
            on_frames(frames, is_idle)

    def warmup(self, on_frames) -> None:
        """静音跑 WARMUP_CHUNKS+1 chunk：GPU 首次推理初始化 + 运动参数预填。"""
        import time as _time

        logger.info("THA3 预热（GPU 首个 chunk 较慢）...")
        self.feed_audio(np.zeros(CHUNK_STEP * (WARMUP_CHUNKS + 1), dtype=np.float32))
        while True:
            with self._cond:
                done = self._real_len - self._pos <= 0
            if done:
                break
            _time.sleep(0.1)
        self.reset()  # 清账本；运动上下文保留（开合度≈0 闭嘴态）
        logger.info("THA3 预热完成")
