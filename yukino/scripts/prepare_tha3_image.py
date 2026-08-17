#!/usr/bin/env python3
"""THA3 立绘准备工具：全身/半身立绘 → 512×512 正方形胸像（THA3 输入）。

THA3 把输入图非等比 resize 到 512×512，输入必须是正方形；同时模型只在
"脸部清晰、嘴眼完整"的胸像构图上效果好（官方要求正脸/近正脸、≥512 分辨率）。

用法：
    # 无参数：打印推荐的头部窗口（启发式）+ 生成参考网格图供肉眼确认
    python scripts/prepare_tha3_image.py

    # 命令行指定矩形（立绘像素坐标，含头发/脸/颈）：
    python scripts/prepare_tha3_image.py \
        --image assets/yukino/ref.png \
        --rect 222,277,777,832 \
        --output assets/yukino/tha3_input.png

    # 交互框选（需要 GUI 显示环境；WSL2 无 X server 时用 --rect 模式）：
    python scripts/prepare_tha3_image.py --interactive

矩形自动扩成正方形（以矩形中心为心，边长取长边；越界时平移窗口），
裁切后 LANCZOS 缩到 512×512 存 PNG，并打印写入 configs/assistant.yaml 的建议。
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def estimate_head_rect(img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """无矩形参数时的启发式：取图像上部 1/3 高度、水平居中 40% 宽。
    （全身立绘头部通常位于上部；精确位置以参考网格图人工确认。）"""
    w = int(img_w * 0.4)
    h = int(img_h * 0.34)
    x0 = (img_w - w) // 2
    y0 = 0
    return x0, y0, x0 + w, y0 + h


def square_rect(x0: int, y0: int, x1: int, y1: int, img_w: int, img_h: int):
    """矩形扩成正方形（中心对齐，越界平移回图内），返回裁剪窗口。"""
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0)
    nx0 = int(round(cx - side / 2))
    ny0 = int(round(cy - side / 2))
    nx0 = max(0, min(nx0, img_w - side))
    ny0 = max(0, min(ny0, img_h - side))
    return nx0, ny0, nx0 + side, ny0 + side


def main() -> None:
    parser = argparse.ArgumentParser(description="THA3 立绘胸像裁剪")
    parser.add_argument("--image", default=str(REPO_ROOT / "assets" / "yukino" / "ref.png"),
                        help="立绘路径（默认 assets/yukino/ref.png）")
    parser.add_argument("--rect", help="头部矩形 x0,y0,x1,y1（立绘像素坐标）")
    parser.add_argument("--rect-in-grid", help="按网格图格子报位置 c0,r0,c1,r1"
                        "（ref_grid.png 每格标注的坐标），工具换算像素")
    parser.add_argument("--output", default=str(REPO_ROOT / "assets" / "yukino" / "tha3_input.png"),
                        help="输出 PNG 路径")
    parser.add_argument("--interactive", action="store_true",
                        help="cv2 窗口鼠标框选（需 GUI；无 GUI 用 --rect）")
    args = parser.parse_args()

    from PIL import Image

    image_path = Path(args.image)
    if not image_path.is_file():
        parser.error(f"立绘不存在: {image_path}")
    img = Image.open(image_path)
    img_w, img_h = img.size
    print(f"立绘: {image_path} ({img_w}×{img_h})")

    rect = None
    if args.interactive:
        try:
            import cv2

            rect = _pick_rect_interactive(str(image_path))
        except Exception as e:  # cv2.imshow 不可用（无 GUI 环境）
            print(f"交互模式不可用（{e}），请用 --rect 参数")
            return
    elif args.rect_in_grid:
        parts = [int(v) for v in args.rect_in_grid.replace(" ", "").split(",")]
        if len(parts) != 4:
            parser.error("--rect-in-grid 需 4 个数: 列0,行0,列1,行1（格子坐标）")
        c0, r0, c1, r1 = parts
        if not (0 <= c0 < c1 <= 10 and 0 <= r0 < r1 <= 10):
            parser.error(f"--rect-in-grid 越界（网格 10×10，格子坐标 0-9）: {args.rect_in_grid}")
        x0, y0 = c0 * img_w // 10, r0 * img_h // 10
        x1, y1 = c1 * img_w // 10, r1 * img_h // 10
        print(f"格子 {c0},{r0},{c1},{r1} → 像素 {x0},{y0},{x1},{y1}")
        rect = (x0, y0, x1, y1)
    elif args.rect:
        parts = [int(v) for v in args.rect.replace(" ", "").split(",")]
        if len(parts) != 4:
            parser.error("--rect 需 4 个数: x0,y0,x1,y1")
        x0, y0, x1, y1 = parts
        if not (0 <= x0 < x1 <= img_w and 0 <= y0 < y1 <= img_h):
            parser.error(f"--rect 越界（图 {img_w}×{img_h}）: {args.rect}")
        rect = (x0, y0, x1, y1)
    else:
        # 无参数：打印推荐 + 参考网格图
        x0, y0, x1, y1 = estimate_head_rect(img_w, img_h)
        print(f"推荐头部窗口: {x0},{y0},{x1},{y1}（启发式，请对照网格图确认）")
        grid_path = image_path.parent / f"{image_path.stem}_grid.png"
        _save_grid(image_path, grid_path)
        print(f"参考网格图已生成: {grid_path}")
        print(f"确认后运行: python scripts/prepare_tha3_image.py "
              f"--rect {x0},{y0},{x1},{y1} --output {args.output}")
        return

    out_path = Path(args.output)
    sx0, sy0, sx1, sy1 = square_rect(*rect, img_w, img_h)
    print(f"裁剪窗口（已扩为正方形）: {sx0},{sy0},{sx1},{sy1} ({sx1 - sx0}px)")

    from PIL import Image
    from PIL import ImageOps

    crop = img.crop((sx0, sy0, sx1, sy1))
    if crop.mode != "RGBA":
        crop = crop.convert("RGBA")
    # 透明背景立绘：白底合成（THA3 输出 alpha 跟随输入，先不抠图）
    bg = Image.new("RGBA", crop.size, (255, 255, 255, 255))
    crop = Image.alpha_composite(bg, crop).convert("RGB")
    crop = crop.resize((512, 512), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    print(f"已保存: {out_path} (512×512)")
    print()
    print("写入 configs/assistant.yaml（avatar 段）:")
    print(f"  avatar.tha3.image: {out_path.as_posix()}")


def _pick_rect_interactive(image_path: str) -> tuple[int, int, int, int]:
    import cv2

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    # 缩小到屏幕可看（保持比例，最长边 900）
    scale = min(1.0, 900 / max(h, w))
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    rect: list[tuple[int, int]] = []
    state = {"dragging": False}

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            rect.clear()
            rect.append((x, y))
            state["dragging"] = True
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            rect.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = False

    cv2.namedWindow("立绘 - 框选头部胸像（含头发/脸/颈），回车确认，q 退出")
    cv2.setMouseCallback("立绘 - 框选头部胸像（含头发/脸/颈），回车确认，q 退出", on_mouse)
    while True:
        disp = small.copy()
        if len(rect) >= 2:
            cv2.rectangle(disp, rect[0], rect[-1], (0, 255, 0), 2)
        cv2.imshow("立绘 - 框选头部胸像（含头发/脸/颈），回车确认，q 退出", disp)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            raise RuntimeError("用户取消")
        if key in (13, 32) and len(rect) >= 2:  # Enter / 空格
            break
    cv2.destroyAllWindows()
    x0, y0 = rect[0]
    x1, y1 = rect[-1]
    x0, y0 = int(x0 / scale), int(y0 / scale)
    x1, y1 = int(x1 / scale), int(y1 / scale)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _save_grid(image_path: Path, out: Path) -> None:
    """画 10×10 网格 + 每格坐标标注，帮助人工定位矩形坐标。

    标注格式 "c,r"（列,行），行号从左上 (0,0) 开始。报位置示例：
    "脸的左上角在格子 3,1，右下角在格子 6,3" → --rect 对应像素坐标
    由工具换算（--rect-in-grid 模式）。"""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img)
    for i in range(1, 10):
        x, y = w * i // 10, h * i // 10
        d.line([(x, 0), (x, h)], fill=(255, 0, 0), width=3)
        d.line([(0, y), (w, y)], fill=(255, 0, 0), width=3)
    d.line([(0, 0), (w, 0), (w, h), (0, h), (0, 0)], fill=(0, 255, 0), width=4)
    # 每格中心标注 (列,行) 坐标，字大一点便于读
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:
        font = ImageFont.load_default()
    cell_w, cell_h = w // 10, h // 10
    for r in range(10):
        for c in range(10):
            d.text((c * cell_w + 6, r * cell_h + 4), f"{c},{r}",
                   fill=(255, 0, 0), font=font)
    img.save(out)


if __name__ == "__main__":
    main()
