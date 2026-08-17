"""模拟管线 TTSInput 流:LLM 输出分多个 chunk,每个 TTSInput 独立 feed+合成。

对照实际日志(19:54:02 只合成了 3.38s),验证:
  1. 整段一次 feed 合成时长(应 ≈ 完整)
  2. 多 chunk 逐段 feed 合成时长(管线实际行为)
打印每段时长,确认是"多段拼接正常"还是"只有首段合成"。
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
SR = 48000


def synth(model, cache, text, label):
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
    print(f"[{label}] text={text!r} -> {len(audio) / SR:.2f}s", flush=True)
    return audio


def main():
    from voxcpm import VoxCPM

    from voxemw.pipeline.tts_voxcpm import BilingualFlow

    print("Loading VoxCPM2 ...", flush=True)
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False, device="cuda")
    ref_wav = str(REPO / "assets" / "yukino" / "ref.wav")
    ref_text = (REPO / "assets" / "yukino" / "ref.txt").read_text(encoding="utf-8").strip()
    cache = model.tts_model.build_prompt_cache(
        prompt_text=ref_text, prompt_wav_path=ref_wav, reference_wav_path=ref_wav
    )

    # 模拟管线输入(与日志 19:54:02 同文本)
    full_llm = "【日语】ふん、私の声が聞こえない？なら、ちゃんと耳を澄まして聞きなさい。何度も言わせないで。【译文】哼,听不到我的声音？那就好好竖起耳朵听清楚,别让我说第二遍。"

    # 1) 整段一次 feed → 一次合成
    f1 = BilingualFlow()
    ja_d, _ = f1.feed(full_llm)
    print(f"整段 feed ja_d = {ja_d!r}", flush=True)
    synth(model, cache, ja_d, "whole")

    # 2) 管线实际:LLM 按句切分多 chunk,每 chunk 独立 TTSInput
    f2 = BilingualFlow()
    chunk1 = "【日语】ふん、私の声が聞こえない？"
    chunk2 = "なら、ちゃんと耳を澄まして聞きなさい。何度も言わせないで。"
    chunk3 = "【译文】哼,听不到我的声音？那就好好竖起耳朵听清楚,别让我说第二遍。"
    total = 0.0
    for i, c in enumerate([chunk1, chunk2, chunk3], 1):
        ja, zh = f2.feed(c)
        print(f"chunk{i}: text={c!r} -> ja_d={ja!r} zh_d={zh!r}", flush=True)
        if ja.strip():
            a = synth(model, cache, ja, f"chunk{i}")
            total += len(a) / SR
    print(f"多 chunk 累计: {total:.2f}s", flush=True)


if __name__ == "__main__":
    main()
