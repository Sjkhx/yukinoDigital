#!/usr/bin/env bash
# VoxEMW 本地 WSL2 一键启停
#   gptsovits-v2proplus（:8899）→ pipeline（:8765）→ orchestrator（:8000）
#
# 用法（在 WSL2 内执行）：
#   bash scripts/start_assistant_wsl.sh           # 启动全部
#   bash scripts/start_assistant_wsl.sh stop      # 停止全部
#   bash scripts/start_assistant_wsl.sh status    # 查看状态
#
# 环境约定：
#   .venv                        VoxEMW 管线/编排依赖
#   ~/miniconda3/envs/myenv      GPT-SoVITS V2ProPlus（torch 2.9.1+cu130）
#   ~/GPT-SoVITS                 GPT-SoVITS 代码与模型
#   avatar.backend=2dlive        数字人前端渲染，不需要 avatar 服务（:8767）
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

PY="$ROOT/.venv/bin/python"
VOXEMW_CONFIG="${VOXEMW_CONFIG:-$ROOT/configs/assistant.yaml}"
PIP_MIRROR="https://mirrors.aliyun.com/pypi/simple"

# GPT-SoVITS V2ProPlus 服务（可用环境变量覆盖）
TTS_PY="${GPT_SOVITS_PY:-$HOME/miniconda3/envs/myenv/bin/python}"
TTS_ROOT="${GPT_SOVITS_ROOT:-$HOME/GPT-SoVITS}"
TTS_SERVER="$ROOT/scripts/gptsovits_v2proplus_server.py"
TTS_PORT=8899

# 本地适配：限线程防空转占满核、离线加载模型缓存
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PATH="$HOME/.local/bin:$PATH"

log() { echo "==> $*"; }

status() {
    local p o t
    p=$(pgrep -f "voxemw.pipeline.launch" | head -1 || true)
    o=$(pgrep -f "voxemw.avatar.orchestrator" | head -1 || true)
    t=$(pgrep -f "gptsovits_v2proplus_server.py" | head -1 || true)
    if [ -n "$t" ]; then echo "gptsovits    : RUNNING (PID $t)"; else echo "gptsovits    : stopped"; fi
    if [ -n "$p" ]; then echo "pipeline     : RUNNING (PID $p)"; else echo "pipeline     : stopped"; fi
    if [ -n "$o" ]; then echo "orchestrator : RUNNING (PID $o)"; else echo "orchestrator : stopped"; fi
    ss -tln 2>/dev/null | grep -E ":(8899|8765|8000) " | sed 's/^/  port /' || true
}

stop() {
    log "停止服务..."
    pkill -f "voxemw.pipeline.launch" 2>/dev/null || true
    pkill -f "voxemw.avatar.orchestrator" 2>/dev/null || true
    pkill -f "gptsovits_v2proplus_server.py" 2>/dev/null || true
    # 等旧进程完全退出释放 qdrant 锁（LocalMode 文件锁，sleep 1 不够，
    # 新进程抢不到锁 → 记忆静默降级，2026-08-11 事故）
    sleep 5
    status
    echo "已停止"
}

if [ "${1:-}" = "stop" ]; then stop; exit 0; fi
if [ "${1:-}" = "status" ]; then status; exit 0; fi

# 只启动 GPT-SoVITS（pipeline/orchestrator 改在 Windows 跑时用：
#   bash scripts/start_assistant_wsl.sh gptsovits
#   powershell -File scripts/start_assistant_win.ps1）
start_gptsovits() {
    log "启动 GPT-SoVITS V2ProPlus（myenv，:8899，模型加载约 30s）..."
    (
        cd "$TTS_ROOT"
        setsid nohup "$TTS_PY" "$TTS_SERVER" --port "$TTS_PORT" \
            >> "$ROOT/logs/gptsovits_v2proplus.log" 2>&1 < /dev/null &
        echo $! > "$ROOT/logs/gptsovits_v2proplus.pid"
        disown
    )
    TTS_PID=$(cat logs/gptsovits_v2proplus.pid)
    echo "    PID=$TTS_PID，日志 logs/gptsovits_v2proplus.log"

    log "等待 GPT-SoVITS（:8899，最多 180s）..."
    TTS_READY=0
    for i in $(seq 1 90); do
        if curl -fsS --max-time 2 "http://127.0.0.1:$TTS_PORT/health" >/dev/null 2>&1; then
            echo "    gptsovits 就绪（${i}0s 内）"
            TTS_READY=1
            break
        fi
        if ! kill -0 "$TTS_PID" 2>/dev/null; then
            echo "    错误：gptsovits 进程已退出！日志尾部："
            tail -30 logs/gptsovits_v2proplus.log
            exit 1
        fi
        sleep 2
    done
    if [ "$TTS_READY" != "1" ]; then
        echo "    错误：gptsovits 等待超时，日志尾部："
        tail -30 logs/gptsovits_v2proplus.log
        exit 1
    fi
}
if [ "${1:-}" = "gptsovits" ]; then
    if ss -tln 2>/dev/null | grep -q ":8899 "; then echo ":8899 已在线（gptsovits 已在跑）"; exit 0; fi
    start_gptsovits
    exit 0
fi

# ── 依赖自检 ──
if ! "$PY" -c "import silero_vad" 2>/dev/null; then
    log "silero-vad 缺失，自动补装（$PIP_MIRROR）..."
    "$PY" -m pip install -q -i "$PIP_MIRROR" silero-vad
fi

if ! [ -x "$PY" ]; then
    echo "ERROR: 缺少 VoxEMW venv：$PY" >&2
    exit 1
fi
if ! [ -x "$TTS_PY" ]; then
    echo "ERROR: 缺少 GPT-SoVITS 环境：$TTS_PY" >&2
    exit 1
fi
if ! [ -d "$TTS_ROOT/GPT_SoVITS" ]; then
    echo "ERROR: 缺少 GPT-SoVITS 代码目录：$TTS_ROOT" >&2
    exit 1
fi
if ! [ -f "$TTS_SERVER" ]; then
    echo "ERROR: 缺少 TTS 服务脚本：$TTS_SERVER" >&2
    exit 1
fi

# ── 停止旧进程，再启动 ──
stop

mkdir -p logs

# avatar：当前 2dlive 为前端本地渲染，不需要后端服务
BACKEND=$("$PY" -c "
import yaml, sys
c = yaml.safe_load(open('$VOXEMW_CONFIG', encoding='utf-8'))
print((c.get('avatar') or {}).get('backend', 'avtr1'))
" 2>/dev/null || echo avtr1)
if [ "$BACKEND" != "2dlive" ]; then
    log "警告：avatar.backend=$BACKEND 需要后端 avatar 服务，本脚本不启动它；"
    log "      数字人会降级为纯语音。"
fi

# ── 1. GPT-SoVITS V2ProPlus ──
start_gptsovits

# ── 2. 语音管线 ──
log "启动语音管线（.venv，:8765）..."
setsid nohup "$PY" -m voxemw.pipeline.launch --config "$VOXEMW_CONFIG" \
    >> logs/pipeline.log 2>&1 < /dev/null & disown
PIPELINE_PID=$!
echo "    PID=$PIPELINE_PID，日志 logs/pipeline.log"

# ── 3. orchestrator ──
log "启动 orchestrator（.venv，:8000 对外）..."
setsid nohup "$PY" -m voxemw.avatar.orchestrator --config "$VOXEMW_CONFIG" \
    >> logs/orchestrator.log 2>&1 < /dev/null & disown
echo "    PID=$!，日志 logs/orchestrator.log"

# ── 等待就绪 ──
log "等待 orchestrator（:8000）..."
for _ in $(seq 1 30); do
    if ss -tln 2>/dev/null | grep -q ":8000 "; then echo "    orchestrator 就绪"; break; fi
    sleep 1
done

log "等待语音管线（:8765，最多 5 分钟）..."
for i in $(seq 1 60); do
    if ss -tln 2>/dev/null | grep -q ":8765 "; then
        echo "    pipeline 就绪（约 $((i * 5))s）"
        break
    fi
    if ! kill -0 "$PIPELINE_PID" 2>/dev/null; then
        echo "    警告：pipeline 进程已退出！日志尾部："
        tail -15 logs/pipeline.log
        exit 1
    fi
    sleep 5
done

echo ""
echo "全部就绪。浏览器打开 http://localhost:8000"
echo "排障：tail -f logs/{gptsovits_v2proplus,pipeline,orchestrator}.log"
