"""检查日文 PDF 的文本层质量：整本扫描，统计乱码页占比。"""
import re
import sys

from pypdf import PdfReader

f = sys.argv[1]
r = PdfReader(f)
cjk = re.compile(r"[　-〿぀-ヿ一-鿿＀-￯]")


def is_garbled(t):
    if not t:
        return True
    n = len(t)
    if n < 20:
        return True
    k = len(cjk.findall(t))
    return k / n < 0.4


bad = 0
total = 0
samples = {}
for i in range(len(r.pages)):
    try:
        t = r.pages[i].extract_text() or ""
    except Exception:
        t = ""
    total += 1
    if is_garbled(t):
        bad += 1
        if len(samples) < 6:
            samples[i] = t[:80]

print(f"pages: {total}, garbled/low-text: {bad} ({bad*100//total}%), clean: {total-bad}")
for i, t in samples.items():
    print(f"  bad page {i}: {t.replace(chr(10), ' | ')[:70]}")

for i in [11, 16, 40, 90, 200, 280]:
    t = (r.pages[i].extract_text() or "").replace(chr(10), "")
    print(f"--- page {i}: {t[:120]}")
