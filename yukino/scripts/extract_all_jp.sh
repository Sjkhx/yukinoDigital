#!/bin/bash
# 批量提取全部日文 PDF（跳过 [2-page] 变体）→ novel/Japanese/txt/
# 输入目录用 YUKINO_NOVEL_SRC 指定（默认仓库内 novel/Japanese，未存在则报错）
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${YUKINO_NOVEL_SRC:-$ROOT/novel/Japanese}"
DST="$SRC/txt"
SCRIPT="$ROOT/scripts/extract_jp_pdf.py"
PY="${YUKINO_PY:-$ROOT/.venv/bin/python}"
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

if [ ! -d "$SRC" ]; then
  echo "ERROR: 未找到日文 PDF 目录：$SRC（可设置 YUKINO_NOVEL_SRC 指向你的素材目录）" >&2
  exit 1
fi

for pdf in "$SRC"/*.pdf; do
  base=$(basename "$pdf" .pdf)
  case "$base" in
    *"\[2-page\]"*|*"\[2- page\]"*) echo "SKIP $base (2-page variant)"; continue ;;
  esac
  out="$DST/${base##*。}.txt"
  if [ -s "$out" ]; then
    echo "EXISTS $out"
  else
    "$PY" "$SCRIPT" "$pdf" "$out" 2>&1 | tail -1
  fi
done
echo "ALL DONE"
