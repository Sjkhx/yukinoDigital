"""THA3 引擎纯函数测试（无 torch/模型依赖，仿 test_orchestrator.py 风格）。

覆盖 voxemw.avatar.tha3_engine 的三个纯函数：
- volume_to_mouth：RMS → 开合度映射（单调/clamp/k 缩放）
- build_pose_vector：45 维参数组装（shape、index 与官方布局一致、默认值）
- blink_state：眨眼相位（快闭缓开、周期外为 0）

参数 index 基准（官方 tha3.poser.modes.standard_float.get_pose_parameters）：
mouth_aaa=26, mouth_delta=31, eye_wink=(12,13), breathing=44,
head_x=39, head_y=40, neck_z=41, iris_rotation_x=37, iris_rotation_y=38。
"""

import numpy as np

from voxemw.avatar.tha3_engine import (
    PARAM,
    blink_state,
    build_pose_vector,
    volume_to_mouth,
)


def test_volume_to_mouth_zero_is_floor():
    assert volume_to_mouth(0.0) == 0.0
    assert volume_to_mouth(0.0001) == 0.0  # 低于 floor 也归零


def test_volume_to_mouth_monotonic():
    vals = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    outs = [volume_to_mouth(v) for v in vals]
    assert all(outs[i] <= outs[i + 1] for i in range(len(outs) - 1))


def test_volume_to_mouth_clamps_and_scales():
    # 强音不开满（k=0.9 缩放）
    assert volume_to_mouth(10.0) == 0.9
    # 弱音不张嘴（幂曲线压低）
    assert volume_to_mouth(0.01) < 0.5
    # 自定义参数生效
    assert volume_to_mouth(1.0, k=1.0) == 1.0
    assert volume_to_mouth(0.01, rms_peak=0.1) > volume_to_mouth(0.01, rms_peak=0.5)


def test_build_pose_vector_shape_and_defaults():
    pose = build_pose_vector()
    assert pose.shape == (45,)
    assert pose.dtype == np.float32
    # 官方默认：mouth_aaa=1.0（中立口型），其余口型参数 0
    assert pose[PARAM["mouth_aaa"]] == 0.0
    assert pose[PARAM["mouth_delta"]] == 0.8  # 默认嘴略收


def test_build_pose_vector_index_layout():
    pose = build_pose_vector(mouth_open=0.7, mouth_scale=1.0, wink=0.5,
                             breathing=0.3, head_x=-0.1, head_y=0.2,
                             neck_z=0.05, iris_x=0.1, iris_y=-0.2)
    # 与官方 get_pose_parameters 顺序一一对应
    assert pose[26] == 0.7    # mouth_aaa
    assert pose[31] == 1.0    # mouth_delta
    assert pose[12] == 0.5    # eye_wink_left
    assert pose[13] == 0.5    # eye_wink_right（双眼同步）
    assert pose[44] == 0.3    # breathing
    assert pose[39] == -0.1   # head_x
    assert pose[40] == 0.2    # head_y
    assert pose[41] == 0.05   # neck_z
    assert pose[37] == 0.1    # iris_rotation_x
    assert pose[38] == -0.2   # iris_rotation_y
    # 未驱动的参数为 0（眉毛/其余口型/眼）
    assert pose[0] == 0.0     # eyebrow_troubled_left
    assert pose[27] == 0.0    # mouth_iii
    assert pose[24] == 0.0    # iris_small_left
    assert pose[42] == 0.0    # body_y


def test_build_pose_vector_clamps():
    pose = build_pose_vector(mouth_open=5.0, wink=-1.0)
    assert pose[26] == 1.0  # 开合 clamp [0,1]
    assert pose[12] == 0.0  # 眨眼 clamp [0,1]


def test_blink_state_cycle():
    # 未在眨眼 / 已结束：0
    assert blink_state(-1.0) == 0.0
    assert blink_state(0.4) == 0.0  # 时长 0.4s 整
    assert blink_state(5.0) == 0.0
    # 快闭：前 35% 升到 1
    assert 0.0 < blink_state(0.05) < blink_state(0.1) < 1.0
    assert blink_state(0.14) > 0.99  # 0.4*0.35=0.14 闭满（浮点边界避免 ==1.0）
    # 缓开：后 65% 降回 0
    assert 0.0 < blink_state(0.2) < blink_state(0.15)
    assert 0.0 < blink_state(0.39)
