#!/usr/bin/env bash
# 开发副本（Windows 桌面 git 仓库）→ WSL2 运行副本同步。
#
# 教训（2026-08-09）：rsync --delete 会把 WSL2 里桌面没有的目录整个清掉——
# .venv（torch cu128 环境）就是这么被删光重装的。本脚本显式排除：
#   .venv / .git / data（运行数据+模型） / logs / .env.local（密钥）
#   __pycache__ / .pytest_cache / *.docx / *.jpg
#
# 用法：bash scripts/sync_to_wsl.sh            # 桌面 → WSL2
#       bash scripts/sync_to_wsl.sh --reverse  # WSL2 → 桌面（备份用）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 仓库根 → WSL2 运行副本（.venv/.git/data/.env.local 已排除防误删）
SRC="$ROOT/"
DST="${YUKINO_WSL_ROOT:-$HOME/VoxEMW}/"
if [ "${1:-}" = "--reverse" ]; then
    SRC="${YUKINO_WSL_ROOT:-$HOME/VoxEMW}/"
    DST="$ROOT/"
fi

EXCLUDES=(
    --exclude '.git' --exclude '.venv' --exclude '.pytest_cache'
    --exclude '__pycache__' --exclude 'logs' --exclude 'data'
    --exclude '.env.local' --exclude '.env.example'
    --exclude '*.docx' --exclude '*.jpg' --exclude '~*' --exclude '.claude'
    --exclude 'out' --exclude 'tmp_ref_clips'   # 实验产物/临时素材，防 --delete 误删
)

echo "==> ${SRC} → ${DST}"
rsync -av --delete "${EXCLUDES[@]}" "$SRC" "$DST"
echo "==> 同步完成（.venv/.git/data/.env.local 已排除）"
