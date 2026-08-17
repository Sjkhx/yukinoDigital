"""Focused mouth analysis - find lip opening line in ref.png"""
from PIL import Image
import numpy as np

im = Image.open('ref.png').convert('RGB')
a = np.array(im).astype(float)
H, W = a.shape[:2]
R, G, B = a[...,0], a[...,1], a[...,2]
lum = (R + G + B) / 3

# The mouth should be in the lower-middle of the face
# Face: cx=696, cy=793 → mouth is below cy, roughly y=780-880
# Mouth width: about 1/3 of face width, so about 100-120px, centered around x=696

y0, y1 = 770, 900
x0, x1 = 600, 800

print("=== Vertical luminance profiles at key x positions ===")
for xc in range(620, 780, 10):
    col = lum[y0:y1, xc]
    # Find the sharpest negative transition (light→dark = mouth opening)
    grad = np.diff(col)  # negative = getting darker going down
    min_grad_idx = np.argmin(grad)
    min_grad = grad[min_grad_idx]
    y_at_grad = y0 + min_grad_idx
    col_min_idx = np.argmin(col)
    y_darkest = y0 + col_min_idx
    col_min = col[col_min_idx]
    print(f"x={xc}: sharpest fall y={y_at_grad} grad={min_grad:.1f} | darkest y={y_darkest} lum={col_min:.1f} | lum@y=800={lum[800-y0,xc-x0]:.0f} lum@y=810={lum[810-y0,xc-x0]:.0f} lum@y=820={lum[820-y0,xc-x0]:.0f} lum@y=830={lum[830-y0,xc-x0]:.0f} lum@y=848={lum[848-y0,xc-x0]:.0f} lum@y=855={lum[855-y0,xc-x0]:.0f}")

print("\n=== Horizontal luminance profiles through possible lip lines ===")
for yc in range(790, 870, 4):
    row = lum[yc, x0:x1]
    min_idx = np.argmin(row)
    min_x = x0 + min_idx
    min_val = row[min_idx]
    avg_val = np.mean(row)
    print(f"y={yc}: min lum={min_val:.1f} at x={min_x}, avg lum={avg_val:.1f}")

print("\n=== Red channel vs Green+Blue at possible lip lines ===")
for yc in range(795, 865, 5):
    row_r = R[yc, x0:x1]
    row_gb = (G[yc, x0:x1] + B[yc, x0:x1]) / 2
    # Lips are redder: R > (G+B)/2
    redness = row_r - row_gb
    max_idx = np.argmax(redness)
    max_x = x0 + max_idx
    max_val = redness[max_idx]
    min_idx = np.argmin(redness)
    min_x = x0 + min_idx
    min_val = redness[min_idx]
    print(f"y={yc}: max redness={max_val:.1f} at x={max_x}, min redness={min_val:.1f} at x={min_x}")

print("\n=== Combined: find y where (dark + wide) best matches mouth ===")
# The mouth line should be a horizontal dark streak spanning ~100px
# Let's look for the row where multiple consecutive columns are dark
for yc in range(790, 870, 2):
    row = lum[yc, x0:x1]
    dark_mask = row < 60
    # Find longest run of dark pixels
    runs = []
    run_start = None
    for i, d in enumerate(dark_mask):
        if d and run_start is None:
            run_start = i
        elif not d and run_start is not None:
            runs.append((run_start, i-1, i - run_start))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(dark_mask)-1, len(dark_mask) - run_start))
    if runs:
        best = max(runs, key=lambda r: r[2])
        if best[2] >= 20:  # at least 20px wide dark streak
            print(f"y={yc}: dark run x={x0+best[0]}-{x0+best[1]} width={best[2]}px, avg lum={np.mean(row[best[0]:best[1]+1]):.1f}")
