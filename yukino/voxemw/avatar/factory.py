"""Avatar 引擎工厂：按 configs/assistant.yaml 的 avatar.backend 选择引擎。

backend:
  - avtr1（默认）：真人说话头，TensorRT（需 pixi env + AVTR1_LOCAL_STORAGE）
  - tha3：动漫参数化说话头（Talking Head Anime 3），PyTorch（项目 .venv，
    data/models/<variant>/ 权重，见 scripts/fetch_tha3_models.py）
  - viseme：动漫口型贴片合成（方案 A，零模型纯合成，画风 100% 保真；
    锚点见 configs/assistant.yaml avatar.viseme，scripts/prepare_tha3_image.py
    生成网格参考图校准）

各引擎接口完全同构（service.py ws 协议层无感知），切换只改配置。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_engine(avatar_cfg: dict, image_path: str):
    """按配置构造引擎实例。avatar_cfg 是 config["avatar"] 段（dict）。

    AVTR1Engine 与 THA3Engine 构造签名不同（前者收 storage/bg_id/idle_motion/
    cfg_self_audio，后者收整个 avatar 段），工厂在此分派，service.py 不感知。"""
    backend = str(avatar_cfg.get("backend", "avtr1"))
    if backend == "viseme":
        from voxemw.avatar.viseme_engine import VisemeEngine

        return VisemeEngine(image_path, avatar_cfg)
    if backend == "tha3":
        from voxemw.avatar.tha3_engine import THA3Engine

        return THA3Engine(image_path, avatar_cfg)
    if backend != "avtr1":
        logger.warning("未知 avatar.backend=%r，回退 AVTR-1", backend)
    from voxemw.avatar.avtr1_engine import AVTR1Engine

    return AVTR1Engine(
        image_path,
        storage=avatar_cfg.get("avtr1_storage") or None,
        bg_id=str(avatar_cfg.get("avtr1_bg", "plain_white")),
        idle_motion=bool(avatar_cfg.get("idle_motion", True)),
        cfg_self_audio=float(avatar_cfg.get("avtr1_cfg_self_audio", 2.0)),
    )
