#!/usr/bin/env bash
# VoxEMW WSL2 一键环境安装（幂等，可重复执行）
#
# 做三件事：
#   1) 创建 .venv + 装依赖（torch cu128 → requirements → silero-vad / tha3）
#   2) 生成 .env.local（缺失 key 提示）
#   3) 检查 GPT-SoVITS 三件套（conda env / 代码 / 权重），缺失给指引
#
# GPT-SoVITS 的 conda 环境与权重不可自动化（论坛权重无稳定 URL），
# 按 docs/SETUP_WSL.md 第 3 步手动装。
#
# 用法（WSL2 内，仓库根）：
#   bash scripts/setup_wsl.sh           # 安装/补齐（幂等）
#   bash scripts/setup_wsl.sh check     # 只检查，不安装
#
# 可覆盖环境变量：
#   SETUP_BASE_PY     建 venv 用的 python（默认 python3，需 >=3.10）
#   PIP_INDEX_URL     pip 镜像（默认阿里云）
#   TORCH_INDEX       torch 源（默认 cu128：RTX 50 系 sm_120 必需）
#   GPT_SOVITS_PY     GPT-SoVITS 环境 python（默认 ~/miniconda3/envs/myenv/bin/python）
#   GPT_SOVITS_ROOT   GPT-SoVITS 代码根（默认 ~/GPT-SoVITS）
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

BASE_PY="${SETUP_BASE_PY:-python3}"
PY="$ROOT/.venv/bin/python"
PIP_MIRROR="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
TORCH_VER="2.8.0"
TTS_PY="${GPT_SOVITS_PY:-$HOME/miniconda3/envs/myenv/bin/python}"
TTS_ROOT="${GPT_SOVITS_ROOT:-$HOME/GPT-SoVITS}"
WEIGHT_DIR="$TTS_ROOT/GPT_SoVITS/pretrained_models/yukino"
# 坑 10：ffmpeg 是 ~/.local/bin 软链（pixi env 自带），默认 PATH 没有，check 前补上
export PATH="$HOME/.local/bin:$PATH"

log() { echo "==> $*"; }
warn_key() {  # $1=变量名 $2=说明
    local v
    v=$(grep "^$1=" "$ROOT/.env.local" 2>/dev/null | head -1 | cut -d= -f2-) || true
    case "$v" in
        "" | sk-*x*) echo "  !! 缺 $1：$2" ;;
    esac
}

if [ "${1:-}" = "check" ]; then
    echo "==> 检查环境（仓库根：$ROOT）"
    if [ -x "$PY" ]; then
        echo "  OK  .venv: $("$PY" --version 2>&1)"
        "$PY" -c "import torch,silero_vad,aiohttp" >/dev/null 2>&1 \
            && echo "  OK  核心依赖: torch $("$PY" -c 'import torch;print(torch.__version__)' 2>/dev/null) + silero_vad + aiohttp" \
            || echo "  !! 核心依赖不全：重跑 bash scripts/setup_wsl.sh"
        "$PY" -c "import tha3" >/dev/null 2>&1 \
            && echo "  OK  tha3（可选，avatar.backend=tha3 用）" \
            || echo "  !! tha3 未装（当前 2dlive 不需要；切 tha3 引擎时需装）"
    else
        echo "  !! 缺 .venv：先跑 bash scripts/setup_wsl.sh"
    fi
    [ -f "$ROOT/.env.local" ] && echo "  OK  .env.local 存在" || echo "  !! 缺 .env.local：跑 setup 会从 .env.example 生成"
    echo "  -- GPT-SoVITS（缺失按 docs/SETUP_WSL.md 第 3 步手动装）--"
    [ -x "$TTS_PY" ] && echo "  OK  GPT-SoVITS python: $TTS_PY" || echo "  !! 缺 $TTS_PY（conda env myenv + torch cu130）"
    [ -d "$TTS_ROOT" ] && echo "  OK  GPT-SoVITS 代码: $TTS_ROOT" || echo "  !! 缺 $TTS_ROOT（jdc4429/GPT-SoVITS-V2ProPlus-Windows）"
    [ -d "$WEIGHT_DIR" ] && echo "  OK  雪乃 V2ProPlus 权重" || echo "  !! 缺雪乃权重（docs/MODEL_DOWNLOAD.md）"
    command -v ffmpeg >/dev/null 2>&1 \
        && echo "  OK  ffmpeg" \
        || echo "  !! 未找到 ffmpeg（当前 tts.rate=1.0 不需要；调语速需装，见坑 10）"
    exit 0
fi

# ── 0. 系统 python 可用性 ──
if ! "$BASE_PY" -c "import sys; assert sys.version_info >= (3,10)" >/dev/null 2>&1; then
    echo "ERROR: $BASE_PY 不可用或 <3.10（Ubuntu 需先 sudo apt install python3-venv python3-pip）" >&2
    exit 1
fi

# ── 1. .venv ──
if [ ! -x "$PY" ]; then
    log "创建 .venv（$BASE_PY -m venv）..."
    "$BASE_PY" -m venv .venv
fi
log "升级 pip..."
"$PY" -m pip install -U pip -q

# ── 2. torch cu128（requirements.txt 顶部注释：torch 需先单独装）──
if "$PY" -c "import torch, torchaudio; assert torch.__version__.startswith('$TORCH_VER') and torchaudio.__version__.startswith('$TORCH_VER')" >/dev/null 2>&1; then
    log "torch 已安装: $("$PY" -c 'import torch; print(torch.__version__)')"
else
    log "安装 torch/torchaudio $TORCH_VER（$TORCH_INDEX，约 2GB，较久）..."
    "$PY" -m pip install --no-cache-dir "torch==$TORCH_VER" "torchaudio==$TORCH_VER" --index-url "$TORCH_INDEX"
fi

# ── 3. 依赖（剔除 tha3 单独 --no-deps 装；speech-to-speech 国内回退 0.2.11）──
REQ_TMP=$(mktemp)
# 坑 4：speech-to-speech 是 git+ 地址，GitHub 被墙时 pip 直连失败 → 回退镜像上的 0.2.11
if curl -fsSL --max-time 6 -o /dev/null https://github.com >/dev/null 2>&1; then
    log "GitHub 可达：保留 speech-to-speech 的 git+ 版本"
    grep -v '^tha3$' requirements.txt > "$REQ_TMP"
else
    log "GitHub 不可达（国内网络）：speech-to-speech 回退 pip 版 0.2.11（坑 4 回滚锚点）"
    grep -v '^tha3$\|^speech-to-speech' requirements.txt > "$REQ_TMP"
    echo "speech-to-speech==0.2.11" >> "$REQ_TMP"
fi
log "安装依赖（镜像 $PIP_MIRROR，可能较久）..."
"$PY" -m pip install -r "$REQ_TMP" -i "$PIP_MIRROR"
rm -f "$REQ_TMP"
log "安装 tha3（--no-deps，坑 5：wxPython 无 Linux wheel，勿带依赖装）..."
"$PY" -m pip install --no-deps -i "$PIP_MIRROR" tha3
log "安装 silero-vad（坑 3：上游 hub 加载隐式依赖，重装环境后需手动补）..."
"$PY" -m pip install -i "$PIP_MIRROR" silero-vad

# ── 4. .env.local ──
if [ ! -f .env.local ]; then
    cp .env.example .env.local
    log "已从 .env.example 生成 .env.local，请填写 API key"
fi
echo "  -- API key 检查 --"
warn_key DEEPSEEK_API_KEY "必填（LLM，platform.deepseek.com 申请）"
warn_key DASHSCOPE_API_KEY "必填（ASR，bailian.console.aliyun.com 申请）"
warn_key MEMORY_EMBEDDER_API_KEY "memory 用；不填可关 configs/assistant.yaml 的 memory.enabled"

# ── 5. GPT-SoVITS 环境检查（不阻塞，缺失给指引）──
echo "  -- GPT-SoVITS（缺失按 docs/SETUP_WSL.md 第 3 步手动装）--"
[ -x "$TTS_PY" ] && echo "  OK  GPT-SoVITS python: $TTS_PY" || echo "  !! 缺 $TTS_PY（conda env myenv + torch cu130）"
[ -d "$TTS_ROOT" ] && echo "  OK  GPT-SoVITS 代码: $TTS_ROOT" || echo "  !! 缺 $TTS_ROOT（jdc4429/GPT-SoVITS-V2ProPlus-Windows）"
[ -d "$WEIGHT_DIR" ] && echo "  OK  雪乃 V2ProPlus 权重" || echo "  !! 缺雪乃权重（docs/MODEL_DOWNLOAD.md）"
command -v ffmpeg >/dev/null 2>&1 || echo "  !! 未找到 ffmpeg（当前 tts.rate=1.0 不需要；调语速需装，见坑 10）"

echo ""
echo "==> 完成。下一步："
echo "  1) 填 API key：编辑 .env.local"
echo "  2) GPT-SoVITS 有缺失的按 docs/SETUP_WSL.md 第 3 步补齐"
echo "  3) bash scripts/start_assistant_wsl.sh    启动三服务（:8899 :8765 :8000）"
echo "  4) 浏览器打开 http://localhost:8000"
