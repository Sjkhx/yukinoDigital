"""演出指令白名单解析测试（voxemw.ai_log.parse_perf_directive_verbose）。

覆盖：合法步骤全保留、英文别名/数字/姿态写法放行、白名单外的值丢弃并记录原因、
parse_perf_directive 兼容入口、write_ai_log 落盘丢弃段。
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from voxemw.ai_log import parse_perf_directive, parse_perf_directive_verbose, write_ai_log  # noqa: E402


def test_valid_steps_kept_in_order():
    text = ("expression:surprised; motion:2a; pause:300; pose:B; "
            "motion:sweat; motion:3b; expression:0")
    steps, dropped = parse_perf_directive_verbose(text)
    assert [s["type"] for s in steps] == [
        "expression", "motion", "pause", "pose", "motion", "motion", "expression"]
    assert steps[0] == {"type": "expression", "value": "surprised"}
    assert steps[1]["value"] == "2a"
    assert steps[2] == {"type": "pause", "value": 300}
    assert steps[3]["value"] == "B"
    assert dropped == []


def test_aliases_and_numeric_accepted():
    steps, dropped = parse_perf_directive_verbose(
        "expression:9; motion:ikari; motion:angry; pose:poseC; pause:800")
    assert len(steps) == 5
    assert steps[0]["value"] == "9"
    assert steps[2]["value"] == "angry"   # 特效英文别名（前端 alias 表）
    assert steps[3]["value"] == "poseC"   # 带前缀姿态写法（前端 setPose 规范化）
    assert dropped == []


def test_unknown_values_dropped_with_reason():
    steps, dropped = parse_perf_directive_verbose(
        "motion:摇头; expression:happy; pause:半秒; pose:D")
    assert steps == []
    assert dropped == [
        "motion:摇头 (未知动作)",
        "expression:happy (未知表情)",
        "pause:半秒 (停顿不是整数毫秒)",
        "pose:D (未知姿态)",
    ]


def test_expression_index_range():
    _, ok = parse_perf_directive_verbose("expression:13")
    assert ok == []
    _, dropped = parse_perf_directive_verbose("expression:14")
    assert dropped == ["expression:14 (未知表情)"]


def test_motion_numeric_index_passthrough():
    # 纯数字按 Action 序号放行（前端按 model3.json 顺序取，越界由前端警告兜底）
    steps, dropped = parse_perf_directive_verbose("motion:6; motion:sweat")
    assert steps[0]["value"] == "6"
    assert dropped == []


def test_chinese_colon_and_noise_tolerated():
    # 中文冒号 + 夹杂散文：切分容忍度不变（白名单只影响值域）；
    # LLM 漂移写中文值时正确进入丢弃记录而非混进步骤
    steps, dropped = parse_perf_directive_verbose("开场顿一下。expression：冷静；motion:1c 结束")
    assert {"type": "motion", "value": "1c"} in steps
    assert dropped == ["expression:冷静 (未知表情)"]


def test_compat_entrypoint_matches():
    text = "motion:2a; pause:500; motion:yami"
    assert parse_perf_directive(text) == parse_perf_directive_verbose(text)[0]


def test_write_ai_log_records_dropped(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXEMW_AI_LOG_DIR", str(tmp_path))
    import importlib

    import voxemw.ai_log as ai_log
    importlib.reload(ai_log)
    ai_log.write_ai_log("test", "raw【演出】motion:摇头", "ja", "zh",
                        "motion:摇头", [], conn_id="t-1",
                        dropped=["motion:摇头 (未知动作)"])
    log = next(f for f in tmp_path.iterdir() if f.name.startswith("ai_"))
    body = log.read_text(encoding="utf-8")
    assert "--- 编排步骤（vox.choreo） ---\n(无)" in body
    assert "--- 丢弃的演出步骤 ---" in body
    assert "motion:摇头 (未知动作)" in body


if __name__ == "__main__":
    import traceback

    # 手动跑入口不含 write_ai_log 用例（依赖 pytest 的 tmp_path），仅覆盖纯解析
    tests = [
        test_valid_steps_kept_in_order,
        test_aliases_and_numeric_accepted,
        test_unknown_values_dropped_with_reason,
        test_expression_index_range,
        test_motion_numeric_index_passthrough,
        test_chinese_colon_and_noise_tolerated,
        test_compat_entrypoint_matches,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed（pytest 下另有 tmp_path 用例）")
    sys.exit(1 if failed else 0)
