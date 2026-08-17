# -*- coding: utf-8 -*-
"""
detect_features.py — 从动漫立绘中检测面部特征（脸、双眼、嘴），输出 JSON 坐标，
并生成 detect_out.png 标注图供人工核对。

用法: python tools/detect_features.py [图片路径] [输出JSON路径]
"""
import sys, json, math
from collections import deque
from PIL import Image, ImageDraw
import numpy as np

def load(path):
    im = Image.open(path).convert('RGB')
    return im, np.asarray(im).astype(np.int32)

def connected_components(mask):
    """mask: bool 2D -> list of [(label, area, bbox(x0,y0,x1,y1))]"""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    comps = {}
    q = deque()
    label = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] and labels[y, x] == 0:
                label += 1
                labels[y, x] = label
                q.append((x, y))
                cnt = 0
                x0 = x1 = x; y0 = y1 = y
                while q:
                    cx, cy = q.popleft()
                    cnt += 1
                    if cx < x0: x0 = cx
                    if cx > x1: x1 = cx
                    if cy < y0: y0 = cy
                    if cy > y1: y1 = cy
                    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx, ny = cx+dx, cy+dy
                        if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = label
                            q.append((nx, ny))
                comps[label] = (cnt, (x0, y0, x1+1, y1+1))
    return comps

def main(path, out_json):
    im, a = load(path)
    H, W, _ = a.shape
    R, G, B = a[...,0], a[...,1], a[...,2]

    # ---------- 1. 皮肤区域（动漫肤色近似） ----------
    skin = (R > 185) & (G > 130) & (B > 90) & (R >= G) & (G >= B) & ((R-B) > 35) & ((R-G) > 12)
    # 去噪（形态学粗化）：简单方式 - 去掉太小的连通域
    comps = connected_components(skin)
    big = {k: v for k, v in comps.items() if v[0] > 400}
    # 脸部 = 面积最大的皮肤连通域，宽高比合理，位于画面上部 2/3
    cands = []
    for cid, (cnt, (x0,y0,x1,y1)) in big.items():
        bw, bh = x1-x0, y1-y0
        if 0.25 < bw/bh < 3.5 and y0 < H*0.75 and cnt > 1500:
            cands.append((cnt, (x0,y0,x1,y1)))
    cands.sort(reverse=True)
    if not cands:
        raise SystemExit("未检测到脸部，请人工在 CONFIG 中指定坐标")
    face = cands[0][1]
    fx0, fy0, fx1, fy1 = face
    fcx, fcy = (fx0+fx1)/2, (fy0+fy1)/2
    fw, fh = fx1-fx0, fy1-fy0

    # ---------- 2. 眼睛（脸部上半区内的深色聚类） ----------
    eye_region = np.zeros_like(skin)
    ey0, ey1 = int(fy0 + fh*0.10), int(fy0 + fh*0.60)
    ex0, ex1 = int(fcx - fw*0.75), int(fcx + fw*0.75)
    eye_region[ey0:ey1, ex0:ex1] = True
    dark = (np.maximum(R,G)-np.minimum(R,G) < 70) & (R < 95) & (G < 95) & (B < 105)
    dark_eyes = dark & eye_region
    ecomps = connected_components(dark_eyes)
    elist = [v for v in ecomps.values() if 25 < v[0] < 30000]
    elist.sort(key=lambda v: v[0], reverse=True)
    # 聚成左右两组
    eyes = []
    for cnt, (x0,y0,x1,y1) in elist[:12]:
        cx, cy = (x0+x1)/2, (y0+y1)/2
        if abs(cx-fcx) > fw*0.05:
            eyes.append((cx, cy, x1-x0, y1-y0, cnt))
    if len(eyes) >= 2:
        eyes.sort(key=lambda e: e[0])
        left = eyes[0]
        right = min([e for e in eyes if e[0] > fcx], key=lambda e: abs(e[1]-left[1]), default=eyes[1])
        left = min([e for e in eyes if e[0] < fcx], key=lambda e: abs(e[1]-right[1]), default=left)
    else:
        left = right = None

    # ---------- 3. 嘴（脸部下半区内的红/深色聚类） ----------
    my0, my1 = int(fy0 + fh*0.50), int(fy0 + fh*0.92)
    mouth_region = np.zeros_like(skin)
    mouth_region[my0:my1, int(fcx-fw*0.55):int(fcx+fw*0.55)] = True
    reddish = ((R-B) > 20) & (R > 90) & (B < 150) | ((R < 110) & (G < 110) & (B < 120))
    mouth_mask = reddish & mouth_region
    mcomps = connected_components(mouth_mask)
    mlist = [v for v in mcomps.values() if v[0] > 30]
    mlist.sort(key=lambda v: v[0], reverse=True)
    mouth = mlist[0] if mlist else None
    if mouth:
        mx0, my0, mx1, my1 = mouth[1]
        mcx, mcy = (mx0+mx1)/2, (my0+my1)/2
        mw, mh = mx1-mx0, my1-my0
    else:
        mcx = mcy = mw = mh = None

    # ---------- 4. 输出 ----------
    result = {
        "image": {"width": W, "height": H},
        "face": {"cx": round(fcx,1), "cy": round(fcy,1), "w": round(fw,1), "h": round(fh,1),
                 "box": [fx0, fy0, fx1, fy1]},
        "left_eye": ({"cx": round(left[0],1), "cy": round(left[1],1),
                      "w": round(left[2],1), "h": round(left[3],1)} if left else None),
        "right_eye": ({"cx": round(right[0],1), "cy": round(right[1],1),
                       "w": round(right[2],1), "h": round(right[3],1)} if right else None),
        "mouth": ({"cx": round(mcx,1), "cy": round(mcy,1), "w": round(mw,1), "h": round(mh,1)}
                  if mouth else None),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # ---------- 5. 标注图 ----------
    d = ImageDraw.Draw(im)
    d.rectangle([fx0, fy0, fx1, fy1], outline=(0, 255, 0), width=4)
    d.text((fx0, fy0-24), "face", fill=(0, 255, 0))
    for name, e in (("L", left), ("R", right)):
        if e:
            x0, y0 = e[0]-e[2]/2, e[1]-e[3]/2
            d.rectangle([x0, y0, x0+e[2], y0+e[3]], outline=(255, 0, 0), width=4)
            d.text((x0, y0-20), name, fill=(255, 0, 0))
    if mouth:
        d.rectangle([mx0, my0, mx1, my1], outline=(0, 0, 255), width=4)
        d.text((mx0, my1+8), "mouth", fill=(0, 0, 255))
    # 网格参考线
    for gy in range(0, H, H//8):
        d.line([0, gy, W, gy], fill=(128, 128, 128), width=1)
    out_png = out_json.rsplit('.', 1)[0] + '_out.png'
    im.save(out_png)
    print("标注图:", out_png)

if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "ref.png"
    oj = sys.argv[2] if len(sys.argv) > 2 else "tools/features.json"
    main(p, oj)
