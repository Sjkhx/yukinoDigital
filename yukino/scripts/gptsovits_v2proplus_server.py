"""GPT-SoVITS V2ProPlus 常驻推理服务（供 VoxEMW tts_gptsovits handler 调用）。

使用 jdc4429/GPT-SoVITS-V2ProPlus-Windows 代码库 + 雪之下雪乃 V2ProPlus 权重：
  - GPT 权重:  yukino-e15.ckpt
  - SoVITS:    yukino_e8_s1744.pth

协议与 scripts/gptsovits_server.py 保持一致：
  GET  /health -> {"ok": true}
  POST /tts    {"text": str, "speed": float, "ref_audio": "001"|"002"|"003"}
               -> 16-bit mono PCM bytes，响应头 X-Sample-Rate 表示采样率

启动（必须在 myenv / 任意含 torch 2.9.1+cu130 的 conda 环境）：
    cd ~/GPT-SoVITS
    ~/miniconda3/envs/myenv/bin/python \
        ~/VoxEMW/scripts/gptsovits_v2proplus_server.py --port 8899
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import uvicorn

GSV_ROOT = Path(os.environ.get("GPT_SOVITS_ROOT") or os.path.expanduser("~/GPT-SoVITS"))
os.chdir(GSV_ROOT)
sys.path.insert(0, str(GSV_ROOT))
sys.path.insert(0, str(GSV_ROOT / "GPT_SoVITS"))

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

CONFIG_PATH = str(GSV_ROOT / "GPT_SoVITS" / "configs" / "tts_infer.yaml")
YUKINO_DIR = GSV_ROOT / "GPT_SoVITS" / "pretrained_models" / "yukino"
REF_DIR = YUKINO_DIR / "reference_audios" / "日语" / "emotions"

# 参考音频三件套（文件名即参考文本，中文前缀只是情绪标签）
REFS = {
    "001": (REF_DIR / "【浅喜】あなたいきなり姉さんのこと言い当てるから驚いたわ.wav",
            "あなたいきなり姉さんのこと言い当てるから驚いたわ"),
    "002": (REF_DIR / "【冷淡】自分の不器用さ無様さ愚かしさの遠因を他人に求めるなんて.wav",
            "自分の不器用さ無様さ愚かしさの遠因を他人に求めるなんて"),
    "003": (REF_DIR / "【动容】一緒に過ごす時間が居心地いいって思えて　嬉しかった.wav",
            "一緒に過ごす時間が居心地いいって思えて　嬉しかった"),
}

app = FastAPI()
PIPELINE: TTS | None = None


class TTSRequest(BaseModel):
    text: str
    speed: float = 1.0
    ref_audio: str = "001"
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 5
    sample_steps: int = 32
    text_split_method: str = "cut5"


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/tts")
def tts(req: TTSRequest):
    if PIPELINE is None:
        return JSONResponse({"error": "models not loaded"}, status_code=503)
    if req.ref_audio not in REFS:
        return JSONResponse({"error": f"unknown ref_audio: {req.ref_audio}"}, status_code=400)
    ref_audio_path, ref_text = REFS[req.ref_audio]
    if not Path(ref_audio_path).is_file():
        return JSONResponse({"error": f"reference audio missing: {ref_audio_path}"}, status_code=500)

    try:
        # 注意：TTS.run 是生成器函数，必须 next() 取出第一段结果后关闭，
        # 直接解包会得到 “not enough values to unpack (expected 2, got 1)”。
        gen = PIPELINE.run({
            "text": req.text,
            "text_lang": "ja",
            "ref_audio_path": str(ref_audio_path),
            "prompt_text": ref_text,
            "prompt_lang": "ja",
            "top_k": req.top_k,
            "top_p": req.top_p,
            "temperature": req.temperature,
            "text_split_method": req.text_split_method,
            "batch_size": 1,
            "batch_threshold": 0.75,
            "split_bucket": True,
            "speed_factor": req.speed,
            "fragment_interval": 0.3,
            "seed": -1,
            "parallel_infer": True,
            "repetition_penalty": 1.35,
            "sample_steps": req.sample_steps,
            "super_sampling": False,
            "streaming_mode": False,
            "overlap_length": 2,
            "min_chunk_length": 16,
        })
        try:
            sr, audio = next(gen)
        finally:
            gen.close()
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)

    pcm = np.asarray(audio, dtype=np.int16)
    return Response(
        content=pcm.tobytes(),
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(sr)},
    )


def main() -> None:
    global PIPELINE

    ap = argparse.ArgumentParser(description="GPT-SoVITS V2ProPlus inference server for VoxEMW")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--gsv-root", default=str(GSV_ROOT))
    args = ap.parse_args()

    root = Path(args.gsv_root)
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "GPT_SoVITS"))

    print(f"loading TTS config: {args.config}", flush=True)
    cfg = TTS_Config(args.config)
    print("loading yukino V2ProPlus models...", flush=True)
    PIPELINE = TTS(cfg)
    print("models loaded", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
