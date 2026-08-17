#!/usr/bin/env python3
"""获取 THA3（Talking Head Anime 3）模型权重到 data/models/<variant>/。

VoxEMW 雪乃说话头（backend=tha3）依赖这 5 个 .pt（官方 talking-head-anime-3-demo
colab 同款来源）：

    editor.pt / eyebrow_decomposer.pt / eyebrow_morphing_combiner.pt /
    face_morpher.pt / two_algo_face_body_rotator.pt

来源（按推荐序）：
  0. --from-package（最推荐）：pip 包装版 34j/tha3 的 wheel 内置 separable_float
     权重（52MB），从已安装的 tha3 包复制到 data/models/separable_float/——
     国内 pip 镜像（阿里云等）装包即得模型，无需访问 HF/Dropbox。
  1. --source dropbox：官方 Dropbox 直链（colab 原样，dl=1 免落地页）
  2. --source hf：HuggingFace 镜像 resolve URL，--endpoint 可换 hf-mirror.com 等
     （repo 文件路径自动猜测，失败时打印提示）

用法：
    python scripts/fetch_tha3_models.py --from-package       # 从 tha3 pip 包复制权重
    python scripts/fetch_tha3_models.py                      # separable_float ← Dropbox
    python scripts/fetch_tha3_models.py --variant standard_float
    python scripts/fetch_tha3_models.py --source hf --endpoint https://hf-mirror.com
    python scripts/fetch_tha3_models.py --verify             # 复制/下载后 torch.load 校验

权重就位后，avatar service 即可以 backend=tha3 启动（引擎加载顺序：
配置 tha3_storage > data/models/<variant> > tha3 包内置）。
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = REPO_ROOT / "data" / "models"

# 官方 colab（pkhungurn/talking-head-anime-3-demo）的 Dropbox 直链
FILES = [
    "editor.pt",
    "eyebrow_decomposer.pt",
    "eyebrow_morphing_combiner.pt",
    "face_morpher.pt",
    "two_algo_face_body_rotator.pt",
]

DROPDOX_LINKS = {
    "standard_float": {
        "editor.pt": "https://www.dropbox.com/s/zp3e5ox57sdws3y/editor.pt?dl=1",
        "eyebrow_decomposer.pt": "https://www.dropbox.com/s/bcp42knbrk7egk8/eyebrow_decomposer.pt?dl=1",
        "eyebrow_morphing_combiner.pt": "https://www.dropbox.com/s/oywaiio2s53lc57/eyebrow_morphing_combiner.pt?dl=1",
        "face_morpher.pt": "https://www.dropbox.com/s/8qvo0u5lw7hqvtq/face_morpher.pt?dl=1",
        "two_algo_face_body_rotator.pt": "https://www.dropbox.com/s/qmq1dnxrmzsxb4h/two_algo_face_body_rotator.pt?dl=1",
    },
    "standard_half": {
        "editor.pt": "https://www.dropbox.com/s/g21ps8gfuvz4kbo/editor.pt?dl=1",
        "eyebrow_decomposer.pt": "https://www.dropbox.com/s/nwwwevzpmxiilgn/eyebrow_decomposer.pt?dl=1",
        "eyebrow_morphing_combiner.pt": "https://www.dropbox.com/s/z5v0amgqif7yup1/eyebrow_morphing_combiner.pt?dl=1",
        "face_morpher.pt": "https://www.dropbox.com/s/g03sfnd5yfs0m65/face_morpher.pt?dl=1",
        "two_algo_face_body_rotator.pt": "https://www.dropbox.com/s/c5lrn7z34x12317/two_algo_face_body_rotator.pt?dl=1",
    },
    "separable_float": {
        "editor.pt": "https://www.dropbox.com/s/nwdxhrpa9fy19r4/editor.pt?dl=1",
        "eyebrow_decomposer.pt": "https://www.dropbox.com/s/hfzjcu9cqr9wm3i/eyebrow_decomposer.pt?dl=1",
        "eyebrow_morphing_combiner.pt": "https://www.dropbox.com/s/g04dyyyavh5o1e2/eyebrow_morphing_combiner.pt?dl=1",
        "face_morpher.pt": "https://www.dropbox.com/s/vgi9dsj95y0rrwv/face_morpher.pt?dl=1",
        "two_algo_face_body_rotator.pt": "https://www.dropbox.com/s/8u0qond8po34l24/two_algo_face_body_rotator.pt?dl=1",
    },
    "separable_half": {
        "editor.pt": "https://www.dropbox.com/s/38pzpxqfnzk6j3j/editor.pt?dl=1",
        "eyebrow_decomposer.pt": "https://www.dropbox.com/s/l7z60cql0c6fjdl/eyebrow_decomposer.pt?dl=1",
        "eyebrow_morphing_combiner.pt": "https://www.dropbox.com/s/t4tdb88vsgsp70y/eyebrow_morphing_combiner.pt?dl=1",
        "face_morpher.pt": "https://www.dropbox.com/s/g93d0rhl1jz9x6s/face_morpher.pt?dl=1",
        "two_algo_face_body_rotator.pt": "https://www.dropbox.com/s/kk8ey2zggbw8pc4/two_algo_face_body_rotator.pt?dl=1",
    },
}

# HF 镜像常见 repo（路径结构以实际 repo 为准，--hf-paths 可覆盖）
HF_REPOS = {
    "OktayAlpk/talking-head-anime-3": "models/{variant}/{file}",
    "ksuriuri/talking-head-anime-3-models": "{file}",
}


def _download(url: str, dst: Path, expected_mb: float | None = None) -> bool:
    """下载单个文件；已存在且体积不小于预期（默认 1MB）则跳过。"""
    if dst.is_file() and dst.stat().st_size >= (expected_mb or 1) * 1024 * 1024:
        print(f"  已存在（{dst.stat().st_size // 1024 // 1024}MB）: {dst.name}")
        return True
    print(f"  下载 {url}")
    print(f"    -> {dst}")
    try:
        with urllib.request.urlopen(url, timeout=300) as resp, open(dst, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            got = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                got += len(chunk)
                if total:
                    pct = got * 100 // total
                    print(f"\r    {pct:3d}% ({got // 1024 // 1024}MB / {total // 1024 // 1024}MB)", end="")
            print()
    except Exception as e:
        print(f"  下载失败: {e}")
        dst.unlink(missing_ok=True)
        return False
    if not dst.is_file() or dst.stat().st_size < 1024 * 1024:
        print("  下载结果异常（文件过小），删除")
        dst.unlink(missing_ok=True)
        return False
    return True


def _copy_from_package(out_dir: Path, variant: str) -> bool:
    """从已安装的 tha3 pip 包复制内置权重（wheel 自带 separable_float）。"""
    import importlib.util

    spec = importlib.util.find_spec("tha3")
    if spec is None or not spec.origin:
        print("tha3 包未安装，先: pip install tha3")
        return False
    pkg_dir = Path(spec.origin).parent
    src = pkg_dir / "data" / "models" / variant
    missing = [f for f in FILES if not (src / f).is_file()]
    if missing:
        print(f"tha3 包内无 {variant} 权重（包自带 separable_float）: 缺 {missing}")
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        dst = out_dir / f
        if dst.is_file() and dst.stat().st_size == (src / f).stat().st_size:
            print(f"  已存在: {dst.name}")
            continue
        import shutil

        shutil.copy2(src / f, dst)
        print(f"  复制 {src / f} -> {dst} ({dst.stat().st_size // 1024 // 1024}MB)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="获取 THA3 模型权重")
    parser.add_argument("--variant", default="separable_float",
                        choices=list(DROPDOX_LINKS), help="模型变体")
    parser.add_argument("--from-package", action="store_true",
                        help="从已安装的 tha3 pip 包复制内置权重（推荐，免外网）")
    parser.add_argument("--source", default="dropbox", choices=["dropbox", "hf"],
                        help="下载来源：dropbox 官方直链 / hf 镜像")
    parser.add_argument("--endpoint", default="https://huggingface.co",
                        help="HF 端点（国内可用 https://hf-mirror.com）")
    parser.add_argument("--repo", default="OktayAlpk/talking-head-anime-3",
                        help="HF 镜像 repo 名")
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR,
                        help="输出根目录（默认 data/models）")
    parser.add_argument("--verify", action="store_true",
                        help="下载后尝试 torch.load 校验（需要 torch）")
    args = parser.parse_args()

    out_dir = args.out / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_package:
        ok = _copy_from_package(out_dir, args.variant)
        if ok:
            print(f"\n完成: {out_dir}/（从 tha3 包复制）")
            if args.verify:
                _verify(out_dir)
            sys.exit(0)
        print("（回退到网络来源）\n")

    if args.source == "dropbox":
        links = DROPDOX_LINKS[args.variant]
    else:
        # HF 镜像：按 repo 的目录结构拼 resolve URL；猜不到结构时给出提示
        links = {}
        for f in FILES:
            for template in (HF_REPOS[args.repo], "models/{variant}/{file}", "{file}"):
                rel = template.format(variant=args.variant, file=f)
                url = f"{args.endpoint}/{args.repo}/resolve/main/{rel}"
                links[f] = url
        print(f"HF 镜像路径按猜测拼接（{args.repo}），若 404 请用 --repo 换镜像仓库")

    ok = True
    for f in FILES:
        ok &= _download(links[f], out_dir / f)

    print(f"\n完成: {out_dir}/（{'全部就绪' if ok else '部分失败，重跑补全'}）")
    if args.verify and ok:
        _verify(out_dir)
    sys.exit(0 if ok else 1)


def _verify(out_dir: Path) -> None:
    """torch.load 校验 5 个权重文件可读（需要 torch）。"""
    try:
        import torch
    except ImportError:
        print("--verify 需要 torch，跳过校验")
        return
    for f in FILES:
        try:
            torch.load(out_dir / f, map_location="cpu", weights_only=True)
            print(f"  torch.load 校验通过: {f}")
        except Exception as e:
            print(f"  torch.load 失败 {f}: {e}")


if __name__ == "__main__":
    main()
