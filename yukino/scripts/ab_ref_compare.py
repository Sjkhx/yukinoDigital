"""48k vs 16k 参考音频对照合成：排查电音/崩坏来源。

假设（2026-08-11）：48k 拼接参考与 VoxCPM2 prompt 编码不匹配 → 崩坏；
跨场景拼接 → 续写混乱。对照：
  A: 48k 拼接 ref（当前线上）
  B: 16k 转码 ref（对齐旧格式）
  C: 16k 单片段 ref（YUK014 单独,不拼接）
合成同一句日语,输出 wav + 每 0.5s 能量(崩坏=能量异常尖峰/断续)。
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TEXT = "こんにちは、主人。今日もいい天気ですね。"

CASES = {
    "A_48k_concat": "/tmp/ref_48k.wav",
    "B_16k_concat": "/tmp/ref_16k.wav",
    "C_16k_single": "/tmp/ref_16k_single.wav",
}


def energy_profile(audio: np.ndarray, sr: int, win: float = 0.5) -> list[float]:
    n = int(win * sr)
    return [float(np.sqrt(np.mean(audio[i:i + n] ** 2))) if audio[i:i + n].size else 0.0
            for i in range(0, len(audio), n)]


def main() -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "/tmp/ref_48k.wav",
                    "-ar", "16000", "-ac", "1", "/tmp/ref_16k.wav"], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i",
                    str(REPO / "tmp_ref_clips" / "A010ESS0_YUK014.ogg"),
                    "-ar", "16000", "-ac", "1", "/tmp/ref_16k_single.wav"], check=True)

    from voxcpm import VoxCPM

    print("Loading VoxCPM2 ...", flush=True)
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False, device="cuda")

    out_dir = REPO / "out" / "ref_ab"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, ref in CASES.items():
        ref_text = (REPO / "assets" / "yukino" / "ref.txt").read_text(encoding="utf-8").strip()
        cache = model.tts_model.build_prompt_cache(
            prompt_text=ref_text, prompt_wav_path=ref, reference_wav_path=ref
        )
        chunks = []
        for wav, _, _ in model.tts_model._generate_with_prompt_cache(
            target_text=TEXT, prompt_cache=cache, min_len=2, max_len=2000,
            inference_timesteps=10, cfg_value=2.0, retry_badcase=False, streaming=True,
        ):
            chunks.append(np.asarray(wav.squeeze(0).cpu().numpy(), dtype=np.float32))
        audio = np.concatenate(chunks) if chunks else np.empty(0)
        prof = energy_profile(audio, 48000)
        print(f"\n[{name}] ref={ref} -> {len(audio)/48000:.2f}s", flush=True)
        print(f"  energy/0.5s: {[round(e, 3) for e in prof]}", flush=True)
        import wave

        with wave.open(str(out_dir / f"{name}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000)
            w.writeframes((np.clip(audio * 32767, -32767, 32767).astype(np.int16)).tobytes())
    print("\n完成,听 out/ref_ab/ 对比", flush=True)


if __name__ == "__main__":
    main()
