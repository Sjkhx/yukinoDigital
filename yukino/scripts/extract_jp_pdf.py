"""日文竖排轻小说 PDF → 干净 txt。

处理要点（基于第01巻实测布局）：
- 页眉区（y < HEADER_BOTTOM）整行丢弃：页码 + 章节名（① 图标等）
- 振り仮名过滤：span 字号 < 正文中位数 × FURIGANA_RATIO 丢弃（实测正文 25.8pt / 振り仮名 13.3pt）
- 竖排重构：每字一 span，按 x 聚列（列间距 ~42pt，列内 jitter <12pt），
  列按 x 降序（右→左=阅读序），列内按 y 升序（上→下）
- 页级质量过滤：CJK 占比 < 40% 的页跳过（封面/目录/人物介绍页的 CID 字体本身无 ToUnicode，提取必乱）
- 插图/空白页自动跳过
"""
import re
import sys
from pathlib import Path

import pymupdf as fitz

HEADER_BOTTOM = 120.0  # 页眉区（页码/章节名）所在 y 阈值
FURIGANA_RATIO = 0.6  # 字号 < 中位数×此比例 → 视为振り仮名
COLUMN_GAP = 12.0  # 相邻列 x 间距 > 此值 → 新列（列内 jitter 通常 <12）
CJK_MIN_RATIO = 0.4  # 页级 CJK 占比低于此 → 整页丢弃

_CJK = re.compile(r"[　-〿぀-ゟ㐀-鿿＀-￯]")


def _page_text(page) -> list[str]:
    spans = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                t = span["text"]
                if not t.strip():
                    continue
                x, y = span["origin"]
                spans.append((x, y, span["size"], t))

    # 1) 滤页眉区
    body = [(x, y, sz, t) for x, y, sz, t in spans if y >= HEADER_BOTTOM]
    if not body:
        return []
    # 2) 页级质量过滤（CID 字体坏页）
    joined = "".join(t for _, _, _, t in body)
    if len(_CJK.findall(joined)) / max(len(joined), 1) < CJK_MIN_RATIO:
        return []
    # 3) 滤振り仮名
    sizes = sorted(sz for _, _, sz, _ in body)
    med = sizes[len(sizes) // 2]
    body = [(x, y, sz, t) for x, y, sz, t in body if sz >= med * FURIGANA_RATIO]
    if not body:
        return []

    # 4) 按 x 聚列：排序后以间距 > COLUMN_GAP 切分
    columns: list[list[tuple[float, float, str]]] = []
    prev_x = None
    for x, y, _sz, t in sorted(body, key=lambda s: s[0]):
        if prev_x is None or x - prev_x > COLUMN_GAP:
            columns.append([])
        columns[-1].append((x, y, t))
        prev_x = x

    # 5) 列内按 y 排序；列按 x 降序（右→左）
    lines = ["".join(ch for _, _, ch in sorted(col, key=lambda s: s[1])) for col in columns]
    return list(reversed(lines))


def extract_volume(path: str) -> str:
    doc = fitz.open(path)
    pages_out: list[str] = []
    for pno in range(len(doc)):
        lines = _page_text(doc[pno])
        if lines:
            pages_out.append("\n".join(lines))
    doc.close()
    return "\n\n".join(pages_out)


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    text = extract_volume(src)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(text, encoding="utf-8")
    print(f"OK {Path(src).name}: {len(text)} chars -> {dst}")


if __name__ == "__main__":
    main()
