"""VoxCPM2 音色保持调优扫描：固定 cfg=2.0（音色贴参考不动），
只对比 timesteps(10/15) × atempo(有/无)。

用户反馈（2026-08-11）：参数扫描改变音色不可接受，只想去电音感。
电音嫌疑：①atempo 0.886 频域变速（音色零改变可验证）；②timesteps=10 扩散
步数低（质感粗糙）。

输出 out/param_scan2/（48kHz）：
  10t_raw       当前音色基准（无变速）
  10t_atempo    当前链路等价（10t + atempo0.886）——"现状"对照
  15t_raw       timesteps 微升（音色应基本不变）
  15t_atempo    15t + atempo（若最终方案仍要变速）

听法：先听 10t_raw vs 10t_atempo（atempo 是否电音源）；再听 15t_raw vs 10t_raw
（质感改善是否值得、音色变化是否可接受）。
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
SR = 48000

TEXT = "こんにちは、主人。今日もいい天気ですね。一緒に散歩でもどうですか？"


def synth(model, cache, text, timesteps, out):
    chunks = []
    for wav, _, _ in model.tts_model._generate_with_prompt_cache(
        target_text=text,
        prompt_cache=cache,
        min_len=2,
        max_len=2000,
        inference_timesteps=timesteps,
        cfg_value=2.0,  # 音色主旋钮，保持不动
        retry_badcase=False,
        streaming=True,
    ):
        chunks.append(np.asarray(wav.squeeze(0).cpu().numpy(), dtype=np.float32))
    audio = np.concatenate(chunks) if chunks else np.empty(0)
    print(f"[{out.name}] steps={timesteps} -> {len(audio) / SR:.2f}s", flush=True)
    import wave

    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(audio * 32767, -32767, 32767).astype(np.int16)).tobytes())
    return out


def atempo(src, dst):
    import subprocess

    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-af", "atempo=0.886", str(dst)],
        check=True,
    )
    print(f"[{dst.name}] atempo 变速完成", flush=True)


def main():
    import os

    os.environ["PATH"] = f"{os.path.expanduser('~')}/.local/bin:" + os.environ.get("PATH", "")

    from voxcpm import VoxCPM

    print("Loading VoxCPM2 ...", flush=True)
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False, device="cuda")
    ref_wav = str(REPO / "assets" / "yukino" / "ref.wav")
    ref_text = (REPO / "assets" / "yukino" / "ref.txt").read_text(encoding="utf-8").strip()
    cache = model.tts_model.build_prompt_cache(
        prompt_text=ref_text, prompt_wav_path=ref_wav, reference_wav_path=ref_wav
    )

    out_dir = REPO / "out" / "param_scan2"
    out_dir.mkdir(parents=True, exist_ok=True)

    for steps in (10, 15):
        raw = synth(model, cache, TEXT, steps, out_dir / f"{steps}t_raw.wav")
        atempo(raw, out_dir / f"{steps}t_atempo.wav")

    print("全部完成，去 out/param_scan2/ 试听对比", flush=True)


if __name__ == "__main__":
    main()
