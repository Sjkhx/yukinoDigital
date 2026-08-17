"""Pinpoint mouth edges in ref.png"""
from PIL import Image
import numpy as np

im = Image.open('ref.png').convert('RGB')
a = np.array(im).astype(float)
R, G, B = a[...,0], a[...,1], a[...,2]
lum = (R + G + B) / 3

# Focus on the found lip area: y≈850, x≈650-750
print("=== Redness (R - (G+B)/2) heatmap around mouth ===")
for yc in range(840, 870, 2):
    row_r = R[yc, 620:760]
    row_gb = (G[yc, 620:760] + B[yc, 620:760]) / 2
    redness = row_r - row_gb
    parts = []
    for xc in range(620, 760, 5):
        idx = xc - 620
        val = redness[idx]
        if val > 45:
            parts.append(f"x{xc}:{val:.0f}")
    if parts:
        print(f"y={yc}: {' '.join(parts)}")

print("\n=== Luminance profile at y=848-860 (lip line) ===")
for yc in [848, 850, 852, 854, 856, 858, 860]:
    row = lum[yc, 600:800]
    parts = []
    for xc in range(600, 800, 5):
        idx = xc - 600
        if row[idx] < 100:
            parts.append(f"x{xc}:{row[idx]:.0f}")
    if parts:
        print(f"y={yc}: {' '.join(parts)}")
    else:
        print(f"y={yc}: all > 100")

print("\n=== Left-to-right mouth edge detection ===")
# For each y in the lip region, find the leftmost and rightmost dark/red pixel
for yc in [848, 850, 852, 854, 856, 858]:
    row_lum = lum[yc, 580:780]
    row_r = R[yc, 580:780]
    row_gb = (G[yc, 580:780] + B[yc, 580:780]) / 2
    redness = row_r - row_gb

    # Find where redness > 40 (lip pixels)
    lip_mask = redness > 40
    lip_indices = np.where(lip_mask)[0]
    if len(lip_indices) > 0:
        left_x = 580 + lip_indices[0]
        right_x = 580 + lip_indices[-1]
        center_x = 580 + lip_indices[len(lip_indices)//2]
        print(f"y={yc}: lip x-range [{left_x}, {right_x}] center≈{center_x} width={right_x-left_x}")

    # Alternative: find where lum drops below threshold
    dark_mask = row_lum < 80
    dark_indices = np.where(dark_mask)[0]
    if len(dark_indices) > 0:
        left_x = 580 + dark_indices[0]
        right_x = 580 + dark_indices[-1]
        center_x = 580 + dark_indices[len(dark_indices)//2]
        print(f"y={yc}: dark x-range [{left_x}, {right_x}] center≈{center_x} width={right_x-left_x}")

print("\n=== RGB values at suspected mouth center (x=696, y=852) ===")
for dy in range(-5, 6):
    y = 852 + dy
    r, g, b = int(R[y, 696]), int(G[y, 696]), int(B[y, 696])
    print(f"y={y}: R={r} G={g} B={b} (redness={r-(g+b)/2:.1f}, lum={(r+g+b)/3:.1f})")

print("\n=== RGB values left-to-right across y=852 ===")
for xc in range(620, 770, 5):
    r, g, b = int(R[852, xc]), int(G[852, xc]), int(B[852, xc])
    redness = r - (g+b)/2
    l = (r+g+b)/3
    print(f"x={xc}: R={r} G={g} B={b} (redness={redness:.1f}, lum={l:.1f})")
