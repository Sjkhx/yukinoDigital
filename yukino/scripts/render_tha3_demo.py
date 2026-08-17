#!/usr/bin/env python3
"""THA3 引擎本地冒烟：正弦音频 → 参数驱动 → 渲染帧序列 → GIF + 统计。

无 GPU 也能跑（CPU 慢但可用）；RTX 5080 上 25fps 实时。
验证三件事：
  1. 嘴随音量包络张合（帧间嘴部区域差异 > 阈值）
  2. 随机眨眼出现（wink 参数序列有脉冲）
  3. 立绘正常渲染（无崩溃/黑屏，输出 GIF 人工确认画风）

用法：
    python scripts/render_tha3_demo.py                    # 3s，out/tha3_demo.gif
    python scripts/render_tha3_demo.py --seconds 2 --frames 20 --out /tmp/demo.gif
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402


def tone_audio(duration_s: float, sr: int = 16000) -> np.ndarray:
    """220Hz 正弦 + 3Hz 振幅调制（模拟说话包络），float32。"""
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    return (0.4 * envelope * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="THA3 引擎渲染冒烟")
    parser.add_argument("--seconds", type=float, default=3.0, help="渲染时长（秒）")
    parser.add_argument("--frames", type=int, default=0, help="限帧（默认不限）")
    parser.add_argument("--out", default=str(REPO_ROOT / "out" / "tha3_demo.gif"))
    parser.add_argument("--image", default=str(REPO_ROOT / "assets" / "yukino" / "tha3_input.png"),
                        help="THA3 胸像输入（scripts/prepare_tha3_image.py 生成）")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "assistant.yaml"))
    args = parser.parse_args()

    from voxemw.avatar.tha3_engine import THA3Engine, CHUNK_STEP, FRAMES_PER_CHUNK

    image = Path(args.image)
    if not image.is_file():
        sys.exit(f"胸像不存在: {image}\n先运行 scripts/prepare_tha3_image.py --rect ... 生成")

    import yaml

    avatar_cfg = (yaml.safe_load(open(args.config, encoding="utf-8")) or {}).get("avatar") or {}
    if avatar_cfg.get("tha3", {}).get("image"):
        avatar_cfg["tha3"]["image"] = str(image)  # 用显式传入的胸像

    print(f"加载 THA3 引擎（{image}）...")
    engine = THA3Engine(str(image), avatar_cfg)
    print(f"  设备: {engine._device}")

    audio = tone_audio(args.seconds)
    n_chunks = min(len(audio) // CHUNK_STEP, args.frames // FRAMES_PER_CHUNK
                   if args.frames else len(audio) // CHUNK_STEP)
    if n_chunks == 0:
        sys.exit("音频太短，至少 0.2s")

    frames: list[np.ndarray] = []
    winks: list[float] = []
    mouths: list[float] = []
    print(f"渲染 {n_chunks} chunk（{n_chunks * FRAMES_PER_CHUNK} 帧）...")
    for ci in range(n_chunks):
        chunk = audio[ci * CHUNK_STEP:(ci + 1) * CHUNK_STEP]
        pose = engine._update_motion(chunk, ci * 0.2)
        winks.append(float(pose[12]))            # eye_wink_left
        mouths.append(float(pose[26]))           # mouth_aaa
        for _ in range(FRAMES_PER_CHUNK):
            frames.append(engine._render_frame(pose))

    print(f"  渲染完成: {len(frames)} 帧")

    # 1) 嘴部动检：全帧 diff（头动也会贡献，但嘴动是主峰）
    diffs = [float(np.abs(frames[i].astype(int) - frames[i - 1].astype(int)).mean())
             for i in range(1, len(frames), 2)]
    print(f"  帧间平均像素差（偶数帧间隔）: {np.mean(diffs):.2f} "
          f"(静音应≈0.3-1，说话应有 2+ 的峰)")
    # 2) 眨眼：wink 序列峰值
    print(f"  mouth_aaa 范围: [{min(mouths):.2f}, {max(mouths):.2f}]"
          f"  峰值差异: {max(mouths) - min(mouths):.2f}（应 >0.3 = 嘴在张合）")
    print(f"  eye_wink 峰值: {max(winks):.2f}（应 >0.5 至少一次 = 有眨眼；"
          f"3s 内可能无——重跑更长）")

    # GIF（缩小到 480×270 控制体积）
    from PIL import Image

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(f[::2, ::2]) for f in frames]
    pil_frames[0].save(out_path, save_all=True, append_images=pil_frames[1:],
                       duration=40, loop=0)
    print(f"GIF 已保存: {out_path}（{len(pil_frames)} 帧）")


if __name__ == "__main__":
    main()
