"""GPT-SoVITS v4 常驻推理服务（WSL2，GPTSoVITS conda 环境运行）。

- 模型常驻 GPU（Yukino Strong/Weak 可配），每句只做推理，避免冷启动
- POST /tts {"text": str, "speed": float} → 16kHz int16 mono PCM（bytes）
- 由 voxemw/pipeline/tts_gptsovits.py 的 handler 调用

启动（WSL2）：
    cd "$GPT_SOVITS_ROOT"    # GPT-SoVITS 代码根（默认 ~/GPT-SoVITS，可环境变量覆盖）
    TMPDIR=/tmp nohup ~/miniconda3/envs/GPTSoVITS/bin/python \
        <yukino 仓库根>/yukino/scripts/gptsovits_server.py \
        --port 8899 --voice strong > ~/gptsovits_server.log 2>&1 &

环境要求：GPTSoVITS env（torch 2.9.0+cu128）；必须从仓库根 cwd 运行（BERT 相对路径）。
"""
import argparse
import io
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

GSV_ROOT = Path(os.environ.get("GPT_SOVITS_ROOT") or os.path.expanduser("~/GPT-SoVITS"))
sys.path.insert(0, str(GSV_ROOT))
sys.path.insert(0, str(GSV_ROOT / "GPT_SoVITS"))

import numpy as np
import torch
import torchaudio
import soundfile as sf

# torchcodec 无法加载 → soundfile 顶替 torchaudio.load
def _load_so(path, *a, **k):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim == 1:
        data = data[None, :]
    else:
        data = data.T
    return torch.from_numpy(np.ascontiguousarray(data)), sr

torchaudio.load = _load_so

from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
from tools.i18n.i18n import I18nAuto

BASE = str(GSV_ROOT / "GPT_SoVITS" / "pretrained_models" / "yukino")
VOICES = {
    "strong": ("Yukino_Strong-e25.ckpt", "Yukino_Strong.pth"),
    "weak": ("Yukino_Weak-e25.ckpt", "Yukino_Weak_01.pth"),
}
VOICE = "weak"  # 模块级，main() 解析 --voice 后更新；/tts 每请求 reload 用
REF_AUDIO = f"{BASE}/001.あなたいきなり姉さんのこと言い当てるから驚いたわ.wav"
REF_TEXT = "あなたいきなり姉さんのこと言い当てるから驚いたわ"

app = FastAPI()
i18n = I18nAuto()
LANG_JA = i18n("日文")
# 试听 batch（inference_cli）成功参数=「不切」+pause 0.3+infer_panel。
# 「按标点符号切」会把句子切太碎导致断句机械（2026-08-12 调试结论）——回退试听同款。
# 采样参数：1.0/1.0 是试听（短句）音色，但 LLM 长文本下 AR 发散/重复率高
#（实测 2/9 次重复）；0.8/0.9 + top_k=5 显著降低发散，配合 /tts 的崩坏检测
# 自动重试兜底（2026-08-12 参数扫描结论）。
HOW_TO_CUT = i18n("不切")


class TTSRequest(BaseModel):
    text: str
    speed: float = 1.0
    temperature: float = 0.8  # 0.8/0.9 平衡音色与 AR 发散（试听 1.0/1.0 长句易重复）
    top_p: float = 0.9
    sample_steps: int = 8
    pause_second: float = 0.3  # 句间停顿（试听同款）
    ref_audio: str = ""  # 空=默认 001；可传 002/003（文件名关键词）


REFS = {
    "001": ("001.あなたいきなり姉さんのこと言い当てるから驚いたわ.wav", "あなたいきなり姉さんのこと言い当てるから驚いたわ"),
    "002": ("002.自分の不器用さ無様さ愚かしさの遠因を他人に求めるなんて.wav", "自分の不器用さ無様さ愚かしさの遠因を他人に求めるなんて"),
    "003": ("003.一緒に過ごす時間が居心地いいって思えて　嬉しかった.wav", "一緒に過ごす時間が居心地いいって思えて　嬉しかった"),
}


@app.post("/tts")
async def tts(req: TTSRequest):  # async → 主线程（event loop）执行，CUDA Graph 捕获需要主线程
    # 每次请求重新加载权重：重置 infer_panel 全局状态——服务进程内多次合成
    # 会状态累积导致质量崩坏（CLI 独立进程无此问题，2026-08-12 对照实验确认）
    gpt, sovits = VOICES[VOICE]
    change_gpt_weights(gpt_path=f"{BASE}/{gpt}")
    change_sovits_weights(sovits_path=f"{BASE}/{sovits}")

    audio_parts = []
    sr_out = 16000
    ref_file, ref_text = REFS.get(req.ref_audio, REFS["001"])
    ref_audio_path = f"{BASE}/{ref_file}"
    # 崩坏检测：AR 发散（EOS 未触发）会生成数倍时长的重复/噪声音频——
    # 实测 20 字句子崩坏 21.6s（1.08s/字）vs 正常 0.2-0.3s/字（2026-08-12）。
    # 按文本长度估算正常时长上限，超限判定崩坏 → 重新合成（AR 采样随机，重试
    # 大概率恢复），最多 3 次；全部崩坏返回最短结果（宁可短不可炸）。
    # 0.4s/字：正常 0.2-0.3s/字（含停顿），68 字段拖沓 33.9s（0.5s/字）漏网
    # 后收紧（2026-08-12）；0.4s/字对正常长句（95 字 ≈ 25s）不误伤。
    expected = max(len(req.text) * 0.4, 10.0)  # 0.4s/字，下限 10s
    best: np.ndarray | None = None
    for attempt in range(3):
        audio_parts = []
        sr = 32000
        for sr, audio in get_tts_wav(
            ref_wav_path=ref_audio_path,
            prompt_text=ref_text,
            prompt_language=LANG_JA,
            text=req.text,
            text_language=LANG_JA,
            how_to_cut=HOW_TO_CUT,
            speed=req.speed,
            top_k=5,  # 官方 CLI 默认（20 太激进，AR 采样发散→重复/噪声）
            top_p=req.top_p,
            temperature=req.temperature,
            sample_steps=req.sample_steps,
            pause_second=req.pause_second,
            # CLI 成功路径=infer_panel（不传 use_cuda_graph）；CUDA Graph 对动态长度
            # 文本状态错乱 → 「重复三次不同语气」（2026-08-12 根因）
            use_cuda_graph=False,
        ):
            # 不重采样：返回原始采样率 PCM，采样率放响应头，由 handler 侧
            # decimate（IIR 8 阶）高质量降采样（resample_poly 混叠致电音）
            audio_parts.append(audio.astype(np.int16))
        if not audio_parts:
            return Response(content=b"", media_type="application/octet-stream")
        out = np.concatenate(audio_parts)
        dur = len(out) / sr
        if dur <= expected:
            break
        logger.warning(
            "合成疑似崩坏（%.1fs > 期望 %.1fs，%d 字），重试 %d/2",
            dur, expected, len(req.text), attempt + 1,
        )
        if best is None or len(out) < len(best):
            best = out
    else:
        out = best
        logger.warning("重试后仍崩坏，返回最短结果（%.1fs）", len(out) / sr)
    return Response(
        content=out.tobytes(),
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(sr)},
    )


@app.get("/health")
def health():
    return {"ok": True}


def main():
    global VOICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--voice", choices=list(VOICES), default="strong")
    args = ap.parse_args()
    VOICE = args.voice

    gpt, sovits = VOICES[args.voice]
    print(f"loading {args.voice}: {gpt} + {sovits}", flush=True)
    change_gpt_weights(gpt_path=f"{BASE}/{gpt}")
    change_sovits_weights(sovits_path=f"{BASE}/{sovits}")
    print("models loaded", flush=True)

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
