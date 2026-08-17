"""Pixel-level mouth detection for ref.png"""
from PIL import Image
import numpy as np

im = Image.open('ref.png').convert('RGB')
a = np.array(im).astype(float)
H, W = a.shape[:2]
R, G, B = a[...,0], a[...,1], a[...,2]

total = R + G + B + 0.001
red_ratio = R / total
lum = total / 3

# Face region from CONFIG: cx=696, cy=793, w=320, h=300
# Scan mouth area: lower face
y0, y1 = 750, 920
x0, x1 = 540, 860

print(f"=== ref.png mouth scan (y={y0}-{y1}, x={x0}-{x1}) ===")
print(f"{'y':>5} {'minLum':>8} {'minLx':>6} {'R@minL':>6} {'maxRed':>8} {'maxRx':>6} {'Rsum@min':>8} {'minRx':>6}")
print("-" * 72)

for yr in range(y0, y1, 3):
    row_lum = lum[yr, x0:x1]
    row_red = red_ratio[yr, x0:x1]
    row_r = R[yr, x0:x1]
    row_sum = R[yr,x0:x1] + G[yr,x0:x1] + B[yr,x0:x1]

    min_lum = np.min(row_lum)
    min_lx = x0 + np.argmin(row_lum)
    r_at_minlum = R[yr, min_lx]

    max_red = np.max(row_red)
    max_rx = x0 + np.argmax(row_red)

    min_sum = np.min(row_sum)
    min_sx = x0 + np.argmin(row_sum)

    if min_lum < 80:  # mark dark rows (potential mouth line)
        print(f"{yr:>5} {min_lum:>8.1f} {min_lx:>6} {r_at_minlum:>6.0f} {max_red:>8.3f} {max_rx:>6} {min_sum:>8.1f} {min_sx:>6} ***")
    else:
        print(f"{yr:>5} {min_lum:>8.1f} {min_lx:>6} {r_at_minlum:>6.0f} {max_red:>8.3f} {max_rx:>6} {min_sum:>8.1f} {min_sx:>6}")

# Find the lip line - where reddish pixels concentrate
print("\n=== Red-ratio peaks (potential lip line) ===")
for yr in range(y0, y1):
    row_red = red_ratio[yr, x0:x1]
    max_red = np.max(row_red)
    if max_red > 0.38:  # high red ratio = lip
        max_rx = x0 + np.argmax(row_red)
        print(f"y={yr} maxRedRatio={max_red:.4f} at x={max_rx}")

print("\n=== Per-column luminance minimums (lip line as darkest y per column) ===")
for xc in range(570, 830, 10):
    col_lum = lum[y0:y1, xc]
    min_y = y0 + np.argmin(col_lum)
    min_val = np.min(col_lum)
    print(f"x={xc}: darkest at y={min_y} (lum={min_val:.1f})")
