#!/usr/bin/env bash
# VoxEMW 数字人语音助手 —— 一键启停（三进程同卡：avatar → pipeline → orchestrator）
#
# 用法：bash scripts/start_assistant.sh [stop]
# 顺序：数字人服务（GPU）→ s2s 语音管线（GPU）→ orchestrator（CPU，:8000 对外）。
set -euo pipefail
cd "$(dirname "$0")/.."

VOXEMW_CONFIG="${VOXEMW_CONFIG:-configs/assistant.yaml}"

# 本地适配：ffmpeg 软链在 ~/.local/bin（pixi env 自带），确保进 PATH
export PATH="$HOME/.local/bin:$PATH"
# torch/OpenMP 线程默认按核数起且忙等自旋:三个 torch 进程同时推理时空转占满核、
# load 爆 40+、实时流抖动卡顿。限 4 线程 + 被动等待
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
# 模型已全部预下载到数据盘,离线加载:避免每次启动都向 hf-mirror 发校验请求
# (网络抖动时 AutoProcessor.from_pretrained 会直接挂);
# HF_HOME 必须显式指到数据盘——非 setup 上下文启动时默认 ~/.cache/huggingface 是空的
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
[ -d /root/autodl-tmp ] && export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
# nltk 等启动期检查会连 GitHub raw（直连被墙会卡死管线启动），开学术加速兜底
source /etc/network_turbo >/dev/null 2>&1 || true

if [ "${1:-}" = "stop" ]; then
    pkill -f "voxemw.avatar.service" 2>/dev/null || true
    pkill -f "voxemw.pipeline.launch" 2>/dev/null || true
    pkill -f "voxemw.avatar.orchestrator" 2>/dev/null || true
    echo "已停止全部服务"
    exit 0
fi

mkdir -p logs
# 先停旧进程（避免显存/端口占用冲突）
pkill -f "voxemw.avatar.service" 2>/dev/null || true
pkill -f "voxemw.pipeline.launch" 2>/dev/null || true
pkill -f "voxemw.avatar.orchestrator" 2>/dev/null || true
sleep 2

# 按 avatar.backend 选引擎：tha3（动漫参数化，项目 .venv 直启）/ avtr1（pixi env）
BACKEND=$(python3 -c "
import yaml, sys
c = yaml.safe_load(open('$VOXEMW_CONFIG', encoding='utf-8'))
print((c.get('avatar') or {}).get('backend', 'avtr1'))
" 2>/dev/null || echo avtr1)

if [ "$BACKEND" = "2dlive" ]; then
    # 2DLive：数字人纯前端渲染（浏览器本地 WebGL），不启动 avatar 服务进程
    echo "==> avatar.backend=2dlive：数字人前端本地渲染，跳过 avatar 服务（:8767）"
    SKIP_AVATAR=1
elif [ "$BACKEND" = "tha3" ]; then
    # THA3：项目 .venv（torch cu128，WSL2 本地 5080 直跑），无 pixi/TRT 依赖
    export PATH="$HOME/.local/bin:$PATH"
    AVATAR_PY="${AVATAR_PY:-.venv/bin/python}"
    echo "==> 启动数字人服务（THA3 动漫说话头，.venv，:8767）"
else
    # AVTR-1：pixi env python 直调（勿 pixi run——会按 lock 重同步 env，
    # 把 pip 降级的 onnxruntime-gpu 1.22 还原成 1.28）
    # 本地适配：/root/autodl-tmp → $HOME（AutoDL 数据盘路径）
    AVTR_ENV="${AVTR_ENV:-$HOME/avtr-1/.pixi/envs/renderer}"
    # 本地适配：pixi env 的 bin 提供 nvcc（torch.compile/inductor 需要），并进 PATH。
    # 注意：必须重建干净 PATH——WSL 默认附加的 Windows PATH 里有 mingw gcc.exe /
    # CUDA nvcc.exe（PE 格式），subprocess 执行会 Permission denied。
    export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$AVTR_ENV/bin:$HOME/.local/bin"
    # 本地适配：系统无 build-essential，triton 编译用 pixi env 的 conda 编译器（真实 ELF）
    export CC="${CC:-$AVTR_ENV/bin/x86_64-conda-linux-gnu-gcc}"
    export CXX="${CXX:-$AVTR_ENV/bin/x86_64-conda-linux-gnu-g++}"
    SP=$AVTR_ENV/lib/python3.12/site-packages
    export LD_LIBRARY_PATH="$(echo $SP/nvidia/*/lib | tr " " ":"):${LD_LIBRARY_PATH:-}"
    export AVTR1_LOCAL_STORAGE="${AVTR1_LOCAL_STORAGE:-$HOME/avtr1_storage}"
    AVATAR_PY="$AVTR_ENV/bin/python"
    echo "==> 启动数字人服务（AVTR-1，pixi env，:8767）"
fi
if [ "${SKIP_AVATAR:-0}" != "1" ]; then
    nohup "$AVATAR_PY" -m voxemw.avatar.service --config "$VOXEMW_CONFIG" \
        > logs/avatar.log 2>&1 &
    echo "    PID=$!，日志 logs/avatar.log（THA3 预热秒级；AVTR-1 要数分钟，耐心等）"
fi

echo "==> 启动语音管线（.venv，:8765）"
nohup .venv/bin/python -m voxemw.pipeline.launch --config "$VOXEMW_CONFIG" \
    > logs/pipeline.log 2>&1 &
echo "    PID=$!，日志 logs/pipeline.log（TTS torch.compile 也要一两分钟）"

echo "==> 等待语音管线 ws 就绪..."
for i in $(seq 1 120); do
    if grep -q "Uvicorn running" logs/pipeline.log 2>/dev/null; then
        break
    fi
    sleep 5
done

echo "==> 启动 orchestrator（.venv，:8000 对外）"
nohup .venv/bin/python -m voxemw.avatar.orchestrator --config "$VOXEMW_CONFIG" \
    > logs/orchestrator.log 2>&1 &
echo "    PID=$!，日志 logs/orchestrator.log"

echo ""
if [ "${SKIP_AVATAR:-0}" = "1" ]; then
    echo "全部启动（2dlive 模式：无 avatar.log，数字人由浏览器本地渲染）。排障：tail -f logs/{pipeline,orchestrator}.log"
else
    echo "全部启动。排障：tail -f logs/{avatar,pipeline,orchestrator}.log"
fi
