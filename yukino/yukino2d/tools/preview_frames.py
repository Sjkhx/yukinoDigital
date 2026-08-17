# -*- coding: utf-8 -*-
"""
preview_frames.py — 离线渲染 yukino2d 引擎的验证帧（无需浏览器/WebGL）。

逐字复刻 web/yukino2d.js 的网格变形数学（嘴部下巴下沉 + 眨眼压缩 + 口腔纹理带），
用 PIL mesh warp 出图，供调参时快速核对效果。

用法: python tools/preview_frames.py [立绘路径] [输出目录]
"""
import sys, math
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# ---- CONFIG（与 web/yukino2d.js 逐字一致） ----
CFG = dict(
    grid=12,
    face=dict(cx=696, cy=793, w=320, h=300),
    eyeL=dict(cx=606, cy=718, w=50, h=25),
    eyeR=dict(cx=786, cy=716, w=55, h=25),
    mouth=dict(cx=698, cy=852, w=26, h=16),
    jawDrop=12,
)

clamp = lambda v, a, b: a if v < a else b if v > b else v
def sm01(x):
    x = clamp(x, 0.0, 1.0); return x * x * (3 - 2 * x)
def gauss(d, s):
    return math.exp(-(d * d) / (2 * s * s))

def eye_weight(x, y, e):
    ex, ey, eh, ew = e["cx"], e["cy"], e["h"], e["w"]
    fX = gauss(x - ex, ew * 0.55)
    if fX < 0.01:
        return 0.0
    closeLine = ey + 3; lashTop = ey - eh * 0.5; loLash = ey + eh * 0.5
    if y <= lashTop:
        dy = (closeLine - lashTop) * sm01((y - (lashTop - eh * 0.38)) / (eh * 0.38))
    elif y < loLash:
        dy = closeLine - y
    else:
        dy = -(loLash - closeLine) * (1 - sm01((y - loLash) / (eh * 0.3)))
    return dy * fX

def build_displacement(W, H, mouth_open, eye_open):
    """顶点网格位移场（只含嘴+眼，与引擎 P.mouth/P.eye 分支一致）。"""
    g = CFG["grid"]
    cols = W // g + 1; rows = H // g + 1
    m = CFG["mouth"]
    disp = np.zeros((rows, cols, 2), dtype=np.float64)
    for j in range(rows):
        y = min(j * g, H)
        for i in range(cols):
            x = min(i * g, W)
            # 嘴：唇线以下高斯衰减下移（下巴下沉）
            fxM = gauss(x - m["cx"], m["w"] * 0.25 + 2)
            yRel = y - m["cy"]
            fyM = 0.0
            if yRel > 0:
                fyM = 1.0 if yRel < m["h"] else 1 - sm01((yRel - m["h"]) / (m["h"] * 1.2))
            dy = fxM * fyM * CFG["jawDrop"] * mouth_open
            # 眨眼
            dy += eye_weight(x, y, CFG["eyeL"]) * (1 - eye_open)
            dy += eye_weight(x, y, CFG["eyeR"]) * (1 - eye_open)
            disp[j, i, 1] = dy
    return disp, g, cols, rows

def warp(im, disp, g, cols, rows):
    """一阶逆映射 + PIL MESH warp。"""
    W, H = im.size
    mesh = []
    for j in range(rows - 1):
        for i in range(cols - 1):
            x0 = min(i * g, W); y0 = min(j * g, H)
            x1 = min((i + 1) * g, W); y1 = min((j + 1) * g, H)
            if x1 <= x0 or y1 <= y0:
                continue
            nw = (x0 - disp[j, i, 0],     y0 - disp[j, i, 1])
            sw = (x0 - disp[j + 1, i, 0], y1 - disp[j + 1, i, 1])
            se = (x1 - disp[j + 1, i + 1, 0], y1 - disp[j + 1, i + 1, 1])
            ne = (x1 - disp[j, i + 1, 0], y0 - disp[j, i + 1, 1])
            mesh.append(((x0, y0, x1, y1), nw + sw + se + ne))
    return im.transform(im.size, Image.MESH, mesh, Image.BILINEAR)

def draw_interior(im, mouth_open):
    """口腔纹理带（引擎 drawInterior 的 2D 复刻，深色渐变）。"""
    if mouth_open < 0.01:
        return im
    m = CFG["mouth"]; jd = CFG["jawDrop"]
    mcx, mcy = m["cx"] + 0.5, m["cy"]
    hw = m["w"] * 0.37; sx = m["w"] * 0.25 + 2
    x0 = int(mcx - hw - 2); x1 = int(mcx + hw + 2)
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    stops = [(0.0, (0x8B, 0x55, 0x53)), (0.4, (0x6A, 0x34, 0x32)),
             (0.7, (0x57, 0x27, 0x23)), (1.0, (0x4A, 0x1E, 0x1C))]
    for x in range(x0, x1 + 1):
        dx = x - mcx
        yTop = mcy - 3 + abs(dx) / hw * 2
        yBot = mcy + 1 + mouth_open * jd * gauss(dx, sx)
        h = yBot - yTop
        if h <= 0:
            continue
        for y in range(int(yTop), int(yBot) + 1):
            t = clamp((y - yTop) / max(h, 1), 0, 1)
            for k in range(len(stops) - 1):
                if stops[k][0] <= t <= stops[k + 1][0]:
                    t0, c0 = stops[k]; t1, c1 = stops[k + 1]
                    f = (t - t0) / (t1 - t0 + 1e-9)
                    c = tuple(round(c0[q] + (c1[q] - c0[q]) * f) for q in range(3))
                    d.point((x, y), fill=c + (255,))
                    break
    out = im.convert("RGBA")
    out.alpha_composite(overlay)
    return out

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "ref.png"
    outdir = Path(sys.argv[2] if len(sys.argv) > 2 else "../out")
    outdir.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB")
    W, H = im.size
    frames = [("idle", 0.0, 1.0), ("talk_mid", 0.45, 1.0),
              ("talk_full", 0.9, 1.0), ("blink", 0.0, 0.0)]
    face_box = (480, 560, 920, 1020)   # 脸部特写裁剪
    for name, mo, eo in frames:
        disp, g, cols, rows = build_displacement(W, H, mo, eo)
        w = warp(im, disp, g, cols, rows)
        w = draw_interior(w, mo)
        w.convert("RGB").save(outdir / f"preview_{name}.png")
        w.convert("RGB").crop(face_box).save(outdir / f"preview_{name}_face.png")
        print("saved", name)

if __name__ == "__main__":
    main()
