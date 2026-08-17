"""VoxCPM2 日语合成实验：验证多句/带【】标记文本的合成是否完整。

对比三组文本的输出音频时长与能量分布：
  1. 纯日语两句话
  2. 带【日语】标记的两句话（与管线实际输入一致）
  3. 【日语】+【译文】完整格式
输出 /tmp/jp_synth_*.wav 供试听，打印每 0.5s 能量（静音段=模型没生成内容）。
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SAMPLE_RATE = 48000  # VoxCPM2 AudioVAE 输出率

CASES = {
    "pure_ja": "こんにちは。今日はいい天気ですね。",
    "marked_ja": "【日语】こんにちは。今日はいい天気ですね。",
    "full": "【日语】こんにちは。今日はいい天気ですね。【译文】你好。今天天气不错。",
}


def energy_profile(audio: np.ndarray, win: float = 0.5) -> list[float]:
    """每 win 秒窗口的 RMS，静音段 ≈ 0。"""
    n = int(win * SAMPLE_RATE)
    out = []
    for i in range(0, len(audio), n):
        seg = audio[i:i + n]
        out.append(float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0)
    return out


def main() -> None:
    from voxcpm import VoxCPM

    print("Loading VoxCPM2 ...", flush=True)
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False, device="cuda")
    ref_wav = str(REPO / "assets" / "yukino" / "ref.wav")
    ref_text = (REPO / "assets" / "yukino" / "ref.txt").read_text(encoding="utf-8").strip()
    cache = model.tts_model.build_prompt_cache(
        prompt_text=ref_text, prompt_wav_path=ref_wav, reference_wav_path=ref_wav
    )

    import wave

    for name, text in CASES.items():
        print(f"\n=== {name}: {text!r} ===", flush=True)
        chunks = []
        for wav, _, _ in model.tts_model._generate_with_prompt_cache(
            target_text=text,
            prompt_cache=cache,
            min_len=2,
            max_len=2000,
            inference_timesteps=10,
            cfg_value=2.0,
            retry_badcase=False,
            streaming=True,
        ):
            chunks.append(np.asarray(wav.squeeze(0).cpu().numpy(), dtype=np.float32))
        audio = np.concatenate(chunks) if chunks else np.empty(0)
        dur = len(audio) / SAMPLE_RATE
        print(f"duration: {dur:.2f}s ({len(audio)} samples)", flush=True)
        prof = energy_profile(audio)
        print(f"energy/0.5s: {[round(e, 3) for e in prof]}", flush=True)
        out = REPO / "out" / f"jp_synth_{name}.wav"
        out.parent.mkdir(exist_ok=True)
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes((np.clip(audio * 32767, -32767, 32767).astype(np.int16)).tobytes())
        print(f"saved: {out}", flush=True)


if __name__ == "__main__":
    main()
