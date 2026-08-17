"""Full facial feature analysis for ref.png - eyes and mouth"""
from PIL import Image
import numpy as np

im = Image.open('ref.png').convert('RGB')
a = np.array(im).astype(float)
H, W = a.shape[:2]
R, G, B = a[...,0], a[...,1], a[...,2]
lum = (R + G + B) / 3

# ============ EYES ============
# Eyes are in the upper half of the face.
# Face: cx~696, cy~793, w~320, h~300 → face y-range ~643-943
# Eyes should be around y=660-750, x=560-800
# Eyes are darker regions with distinct shapes

print("=" * 60)
print("EYE REGION ANALYSIS")
print("=" * 60)

# Left eye candidate: x=560-660, y=650-760
# Right eye candidate: x=690-820, y=650-760

# Scan the eye region row by row, looking for dark pixels
print("\n--- Left eye region: luminance scan (x=560-660, every row y=680-760) ---")
for yc in range(680, 760):
    row = lum[yc, 560:670]
    min_lum = np.min(row)
    if min_lum < 120:  # dark pixels = eye
        min_x = 560 + np.argmin(row)
        # Find the dark region width
        dark = row < 100
        dark_count = np.sum(dark)
        if dark_count > 0:
            dark_idx = np.where(dark)[0]
            left_x = 560 + dark_idx[0]
            right_x = 560 + dark_idx[-1]
            print(f"y={yc}: darkest x={min_x} lum={min_lum:.0f}  dark_range=[{left_x},{right_x}] width={right_x-left_x}")

print("\n--- Right eye region: luminance scan (x=650-850, every row y=680-760) ---")
for yc in range(680, 760):
    row = lum[yc, 650:850]
    min_lum = np.min(row)
    if min_lum < 120:
        min_x = 650 + np.argmin(row)
        dark = row < 100
        dark_count = np.sum(dark)
        if dark_count > 0:
            dark_idx = np.where(dark)[0]
            left_x = 650 + dark_idx[0]
            right_x = 650 + dark_idx[-1]
            print(f"y={yc}: darkest x={min_x} lum={min_lum:.0f}  dark_range=[{left_x},{right_x}] width={right_x-left_x}")

# Also check for eye white (bright areas near dark areas = sclera)
# and iris (colored circular area)

print("\n--- Left eye: RGB values at key dark points ---")
# Find the darkest cluster in left eye region
left_eye_region = lum[690:750, 560:650]
min_flat = np.argmin(left_eye_region)
min_y = 690 + min_flat // 90
min_x = 560 + min_flat % 90
print(f"Left eye darkest point: ({min_x}, {min_y}) lum={lum[min_y, min_x]:.0f}")
# Sample around it
for dy in range(-8, 9, 2):
    y = min_y + dy
    row_data = []
    for dx in range(-15, 16, 3):
        x = min_x + dx
        if 0 <= x < W and 0 <= y < H:
            r, g, b = int(R[y,x]), int(G[y,x]), int(B[y,x])
            l = (r+g+b)/3
            row_data.append(f"x{x}:{l:.0f}")
    print(f"y={y}: {' '.join(row_data)}")

print("\n--- Right eye: RGB values at key dark points ---")
right_eye_region = lum[685:755, 720:820]
min_flat = np.argmin(right_eye_region)
min_y = 685 + min_flat // 100
min_x = 720 + min_flat % 100
print(f"Right eye darkest point: ({min_x}, {min_y}) lum={lum[min_y, min_x]:.0f}")
for dy in range(-8, 9, 2):
    y = min_y + dy
    row_data = []
    for dx in range(-15, 16, 3):
        x = min_x + dx
        if 0 <= x < W and 0 <= y < H:
            r, g, b = int(R[y,x]), int(G[y,x]), int(B[y,x])
            l = (r+g+b)/3
            row_data.append(f"x{x}:{l:.0f}")
    print(f"y={y}: {' '.join(row_data)}")

# ============ MOUTH ============
print("\n" + "=" * 60)
print("MOUTH REGION - DETAILED SHAPE")
print("=" * 60)

# Mouth center at y≈850, x≈697
# Let's get the exact shape: find the dark lip line contour
print("\n--- Mouth: luminance cross-section at lip line y=848-858 ---")
for yc in range(844, 864):
    row = lum[yc, 670:730]
    parts = []
    for xi in range(0, 60, 2):
        x = 670 + xi
        l = row[xi]
        if l < 150:
            parts.append(f"x{x}:{l:.0f}")
    if parts:
        print(f"y={yc}: {' '.join(parts)}")
    else:
        print(f"y={yc}: (all skin >150)")

print("\n--- Mouth: redness profile at lip line ---")
for yc in range(844, 865):
    row_r = R[yc, 670:730]
    row_gb = (G[yc, 670:730] + B[yc, 670:730]) / 2
    redness = row_r - row_gb
    parts = []
    for xi in range(0, 60, 2):
        x = 670 + xi
        rv = redness[xi]
        if rv > 35:
            parts.append(f"x{x}:{rv:.0f}")
    if parts:
        print(f"y={yc}: redness>35 → {' '.join(parts)}")

print("\n--- Mouth: darkest row in lip region (y=846-860) for each x ---")
for xc in range(685, 715):
    col = lum[844:865, xc]
    min_y = 844 + np.argmin(col)
    min_val = np.min(col)
    print(f"x={xc}: darkest y={min_y} lum={min_val:.0f}")
