"""转写雪乃 ogg 参考片段候选 → 台词文本（DashScope qwen3-asr-flash, 日语）。

用法：.venv/bin/python scripts/transcribe_ref_candidates.py
候选文件从 Windows 路径复制到 /tmp/yukino_ogg/ 后运行。
"""
import base64
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from voxemw.config import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env.local")
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
if not API_KEY:
    sys.exit("缺 DASHSCOPE_API_KEY")

SRC = Path("/tmp/yukino_ogg")
CANDIDATES = [
    "A010ESS0_YUK040.ogg",
    "A010ESS0_YUK014.ogg",
    "A010ESS0_YUK007.ogg",
    "A010ESS0_YUK026.ogg",
    "A010ESS0_YUK018.ogg",
]


def to_wav_b64(ogg: Path) -> str:
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(ogg), "-f", "wav", "-ac", "1", "-ar", "16000", "pipe:1"],
        capture_output=True,
    )
    return base64.b64encode(r.stdout).decode()


def transcribe(ogg: Path) -> str:
    data_url = "data:audio/wav;base64," + to_wav_b64(ogg)
    body = {
        "model": "qwen3-asr-flash",
        "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": data_url}}]}],
        "asr_options": {"language": "ja", "enable_itn": True},
    }
    req = urllib.request.Request(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode())
    return result["choices"][0]["message"]["content"]


def main() -> None:
    os.environ["PATH"] = f"{os.path.expanduser('~')}/.local/bin:" + os.environ.get("PATH", "")
    if not SRC.is_dir():
        sys.exit(f"缺目录 {SRC}（先复制 ogg 过去）")
    for name in CANDIDATES:
        p = SRC / name
        if not p.is_file():
            print(f"{name}: 缺失")
            continue
        try:
            print(f"{name}: {transcribe(p)}", flush=True)
        except Exception as e:
            print(f"{name}: ERROR {e}", flush=True)


if __name__ == "__main__":
    main()
