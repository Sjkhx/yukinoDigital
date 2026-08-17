#!/usr/bin/env python3
"""Revert index.html from the new image back to ref.png"""
import base64, os
from pathlib import Path

PROJ = str(Path(__file__).resolve().parent.parent)

# 1. Generate ref.png base64
with open(os.path.join(PROJ, 'ref.png'), 'rb') as f:
    b64 = base64.b64encode(f.read()).decode('ascii')
print(f'[1/4] Generated base64: {len(b64)} chars')

# 2. Read current index.html
idx_path = os.path.join(PROJ, 'index.html')
with open(idx_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
print(f'[2/4] Read {len(lines)} lines')

# 3. Replace IMG_EMBED (line 119, 0-indexed)
new_embed = "const IMG_EMBED = 'data:image/png;base64," + b64 + "';\n"
print(f'  IMG_EMBED new length: {len(new_embed)} chars')
lines[119] = new_embed

# 4. Replace CONFIG block (lines 121-153, 0-indexed, 33 lines)
new_config = [
    "const CONFIG = {\n",
    "  img: 'ref.png',\n",
    "  /* 画布尺寸(px)，使用素材原始尺寸以确保纹理坐标精确 */\n",
    "  size: { w: 1279, h: 2177 },\n",
    "  grid: 12,                           // 网格单元大小(px)，越小越精细\n",
    "  /* ---- 特征定位（均为素材像素坐标）----\n",
    "     请根据实际图片调整！建议用图像编辑器取关键点坐标。\n",
    "     脸框：大致包围面部（不含头发）\n",
    "     眼框：包围眼睛（含眼眶，比眼睛本身略大）\n",
    "     嘴框：包围嘴唇区域（口裂线附近） */\n",
    "  face:   { cx: 696, cy: 793, w: 320, h: 300 },  // 脸\n",
    "  eyeL:   { cx: 605, cy: 722, w: 60,  h: 25 },   // 左眼\n",
    "  eyeR:   { cx: 784, cy: 716, w: 70,  h: 25 },   // 右眼\n",
    "  mouth:  { cx: 700, cy: 810, w: 150, h: 140 },  // 嘴\n",
    "  browY:  690,                       // 眉线（以上视为头发区）\n",
    "  neckY:  905,                       // 颈部线（以下视为身体区）\n",
    "  /* --- 幅度 --- */\n",
    "  amp: {\n",
    "    headTurn: 14,    // 头部转向位移 px\n",
    "    headNod:  6,     // 头部点头位移 px\n",
    "    jawDrop:  43,    // 嘴开合幅度 px\n",
    "    breathe:  0.0006,// 呼吸整体缩放\n",
    "    hair:     3.2,   // 头发摆动 px\n",
    "    hairLag:  9,     // 头发对头部转向的延迟跟随 px\n",
    "  },\n",
    "  /* 平滑速度(1/秒)，越大越快 */\n",
    "  speed: { mouth:16, eye:26, head:6, hair:2.8, breathe:4 },\n",
    "};\n",
]
# Replace lines[121:154] (inclusive of 121, exclusive of 154) with new_config
lines[121:154] = new_config
print(f'[3/4] Replaced CONFIG ({len(new_config)} lines)')

# 5. Fix mouth displacement formula (line 420 changed -> need to find new index)
# After replacing 33 lines with 29, the line index shifts by (29 - 33) = -4
# Original line 420 will now be at line 420 + (-4) = 416
# But wait, line 119 (IMG_EMBED) was replaced 1:1 so no shift there.
# Actually, lines[121:154] is 33 lines replaced with 29 → -4 shift for everything after line 153.
# So:
# - Line 420 became line 416
# - Line 448 became line 444
# Let's verify by searching
def find_line(text):
    for i, line in enumerate(lines):
        if text in line:
            return i, line.rstrip()
    return None, None

idx_mouth, txt_mouth = find_line('wMouth[k]*(P.mouth')
print(f'  Mouth formula: line {idx_mouth}: {txt_mouth}')
idx_open, txt_open = find_line('CONFIG.mouthOpen')
print(f'  MouthOpen guard: line {idx_open}: {txt_open}')

if idx_mouth is not None:
    old = lines[idx_mouth]
    lines[idx_mouth] = old.replace(
        "+ wMouth[k]*(P.mouth - CONFIG.mouthNeutral)*CONFIG.mouthScale",
        "+ wMouth[k]*P.mouth"
    )
    print(f'  Fixed mouth formula on line {idx_mouth}')

if idx_open is not None:
    # Remove the mouthOpen guard line
    indent = lines[idx_open][:len(lines[idx_open]) - len(lines[idx_open].lstrip())]
    lines[idx_open] = ""
    print(f'  Removed mouthOpen guard on line {idx_open}')

# 6. Write back
with open(idx_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)
print(f'[4/4] Written {len(lines)} lines back to index.html')

# Verify
with open(idx_path, 'r', encoding='utf-8') as f:
    content = f.read()
checks = [
    ("ref.png", "img: 'ref.png'"),
    ("jawDrop:  43", "amp.jawDrop"),
    ("no mouthNeutral", "mouthNeutral"),
    ("wMouth[k]*P.mouth", "wMouth[k]*P.mouth"),
    ("no mouthOpen guard", "CONFIG.mouthOpen"),
    ("data:image/png", "data:image/png;base64,"),
]
for label, pattern in checks:
    found = pattern in content
    status = "✓" if found else ("✗" if "no " not in label else "✓")
    if "no " in label:
        status = "✓" if not found else "✗"
    print(f'  [{status}] {label}')
