"""Precise eye boundary detection for ref.png"""
from PIL import Image
import numpy as np

im = Image.open('ref.png').convert('RGB')
a = np.array(im).astype(float)
H, W = a.shape[:2]
R, G, B = a[...,0], a[...,1], a[...,2]
lum = (R + G + B) / 3

def find_dark_edges(lum, cx, cy, search_w, search_h, dark_thresh=80):
    """Find the bounding box of dark pixels in a region around (cx,cy)"""
    x0, x1 = cx - search_w//2, cx + search_w//2
    y0, y1 = cy - search_h//2, cy + search_h//2

    region = lum[y0:y1, x0:x1]
    dark = region < dark_thresh

    dark_rows, dark_cols = np.where(dark)
    if len(dark_rows) == 0:
        return None

    min_y = y0 + dark_rows.min()
    max_y = y0 + dark_rows.max()
    min_x = x0 + dark_cols.min()
    max_x = x0 + dark_cols.max()

    return {
        'left': min_x, 'right': max_x, 'top': min_y, 'bottom': max_y,
        'cx': (min_x+max_x)/2, 'cy': (min_y+max_y)/2,
        'w': max_x-min_x, 'h': max_y-min_y,
        'n_dark': len(dark_rows)
    }

print("=== LEFT EYE: Testing different center positions ===")
# Try different centers and see which gives the most compact bounding box
for cx_test in range(590, 625, 5):
    for cy_test in range(700, 725, 5):
        edges = find_dark_edges(lum, cx_test, cy_test, 80, 80, dark_thresh=80)
        if edges and edges['n_dark'] > 100:
            print(f"center=({cx_test},{cy_test}) → bbox[cx={edges['cx']:.0f},cy={edges['cy']:.0f},w={edges['w']},h={edges['h']}] n={edges['n_dark']}")

print("\n=== LEFT EYE: Detail scan with optimal center ===")
# From initial data, the eye seems centered around (605, 710)
# Let's scan with tight threshold to find the iris/pupil (darkest) vs eyelid (moderately dark)
for thresh in [50, 60, 70, 80, 90, 100]:
    edges = find_dark_edges(lum, 605, 710, 80, 80, dark_thresh=thresh)
    if edges:
        print(f"thresh<{thresh}: bbox[l={edges['left']},r={edges['right']},t={edges['top']},b={edges['bottom']}] w={edges['w']} h={edges['h']}")

print("\n=== LEFT EYE: Row-by-row dark boundaries (lum<70) ===")
for yc in range(685, 745):
    row = lum[yc, 570:650]
    dark = row < 70
    dark_idx = np.where(dark)[0]
    if len(dark_idx) >= 2:
        print(f"y={yc}: dark x=[{570+dark_idx[0]}, {570+dark_idx[-1]}] width={dark_idx[-1]-dark_idx[0]}")

print("\n=== RIGHT EYE: Row-by-row dark boundaries (lum<70, x=740-840) ===")
for yc in range(700, 745):
    row = lum[yc, 740:840]
    dark = row < 70
    dark_idx = np.where(dark)[0]
    if len(dark_idx) >= 2:
        print(f"y={yc}: dark x=[{740+dark_idx[0]}, {740+dark_idx[-1]}] width={dark_idx[-1]-dark_idx[0]}")

print("\n=== RIGHT EYE: Testing different center positions (excluding hair) ===")
for cx_test in range(770, 810, 3):
    for cy_test in range(705, 730, 3):
        edges = find_dark_edges(lum, cx_test, cy_test, 60, 50, dark_thresh=80)
        if edges and 50 < edges['n_dark'] < 800:
            print(f"center=({cx_test},{cy_test}) → bbox[cx={edges['cx']:.0f},cy={edges['cy']:.0f},w={edges['w']},h={edges['h']}] n={edges['n_dark']}")

print("\n=== RIGHT EYE: x-profile at suspected eye y=718-735, x=760-830 ===")
for yc in range(715, 738, 2):
    parts = []
    for xc in range(760, 830, 3):
        l = lum[yc, xc]
        if l < 100:
            parts.append(f"x{xc}:{l:.0f}")
    if parts:
        print(f"y={yc}: {' '.join(parts)}")
    else:
        print(f"y={yc}: (all > 100)")

print("\n=== MOUTH FINAL: Precise outline ===")
# From earlier: mouth center ~(697, 850), lip line at y=848-858
# Find the exact contour
print("Dark pixels (lum<80) and redness peaks across lip region (x=680-720, y=844-862):")
for yc in range(844, 864):
    dark_parts = []
    red_parts = []
    for xc in range(680, 720):
        l = lum[yc, xc]
        rd = R[yc,xc] - (G[yc,xc] + B[yc,xc]) / 2
        if l < 80:
            dark_parts.append(f"x{xc}:l{l:.0f}")
        if rd > 45:
            red_parts.append(f"x{xc}:r{rd:.0f}")
    if dark_parts or red_parts:
        print(f"y={yc}: dark=[{' '.join(dark_parts)}] red=[{' '.join(red_parts)}]")

print("\n=== MOUTH: Width summary ===")
# Find the widest part of the mouth based on redness
for yc in range(848, 858):
    rd = R[yc, 680:720] - (G[yc, 680:720] + B[yc, 680:720]) / 2
    lip = rd > 40
    lip_idx = np.where(lip)[0]
    if len(lip_idx) > 0:
        left = 680 + lip_idx[0]
        right = 680 + lip_idx[-1]
        print(f"y={yc}: lip redness>40: x=[{left}, {right}] width={right-left}")
