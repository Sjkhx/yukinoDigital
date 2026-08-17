# -*- coding: utf-8 -*-
"""THA3 色域 A/B 测试：恒等 pose（闭嘴、无头动）下哪种 输入/输出 转换最接近原图。"""
import numpy as np
import torch
from PIL import Image
from tha3.poser.modes import separable_float
from tha3.util import torch_linear_to_srgb

poser = separable_float.create_poser("cpu")
img = Image.open("assets/yukino/tha3_input_rgba.png").convert("RGBA").resize((512, 512))
x01 = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

pose = torch.zeros(1, 45)
pose[0, 26] = 0.0   # mouth_aaa 闭嘴
pose[0, 31] = 0.8   # mouth_delta
pose[0, 44] = 0.5   # breathing


def to_img(t):  # (1,4,H,W) [0,1] -> PIL，alpha 合成深灰底
    a = np.clip(t[0].permute(1, 2, 0).cpu().numpy(), 0, 1)
    comp = a[..., :3] * a[..., 3:4] + 0.15 * (1 - a[..., 3:4])
    return Image.fromarray((np.clip(comp, 0, 1) * 255).astype(np.uint8))


def l2s(out):  # (1,4,H,W) [-1,1] -> [0,1]，RGB linear->sRGB，alpha 不动
    o = ((out + 1) / 2).clamp(0, 1)
    return torch.cat([torch_linear_to_srgb(o[:, :3]), o[:, 3:4]], 1)


def srgb(out):
    return ((out + 1) / 2).clamp(0, 1)


cases = {
    "A_in-11_out-srgb": (x01 * 2 - 1, srgb),
    "B_in-11_out-l2s": (x01 * 2 - 1, l2s),
    "C_in01_out-srgb": (x01, srgb),
    "E_in01_out-l2s": (x01, l2s),
}
for name, (inp, post) in cases.items():
    with torch.inference_mode():
        out = poser.pose(inp, pose)
    to_img(post(out)).save(f"out/_ab_{name}.png")
    print(name, "ok")

# 参照：原图直接合成
to_img(x01).save("out/_ab_REF.png")
print("REF ok")
