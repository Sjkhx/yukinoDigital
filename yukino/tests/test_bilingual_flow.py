"""BilingualFlow 流式切分器测试：「【日语】…【译文】…【演出】…」LLM 输出格式。

覆盖：完整段、标记被 chunk 切开、无标记回退（none 模式）、重置、端到端拼接、
【演出】段（不进日语/译文、perf_buf 累积、strip_perf 兜底）。
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from voxemw.pipeline.tts_voxcpm import BilingualFlow  # noqa: E402


def test_full_stream():
    f = BilingualFlow()
    ja, zh, perf = f.feed("【日语】こんにちは。")
    # 安全前缀（防【译文】被切开）暂留尾部 len(ZH)-1=3 字符：迟一个 chunk 输出
    assert ja == "こんに" and zh == "" and perf == ""
    ja2, zh2, perf2 = f.feed("【译文】你好。")
    assert ja2 == "ちは。"  # 上一段暂留的字符补齐
    assert zh2 == "你好。"
    assert f.mode == "zh"


def test_marker_split_across_chunks():
    f = BilingualFlow()
    ja, zh, _ = f.feed("【日")
    assert ja == "" and zh == ""
    ja, zh, _ = f.feed("语】お元気ですか？")
    # 尾部"すか？"被当可能被切开的标记前缀暂留
    assert ja == "お元気で"
    ja, zh, _ = f.feed("【译")
    assert ja == "すか"  # 暂留的日语字符先输出
    ja, zh, _ = f.feed("文】你还好吗？")
    assert ja == "？" and zh == "你还好吗？"  # 最后暂留字符补齐


def test_no_markers_keeps_buffer():
    # LLM 未按格式输出：增量恒空（TTS 静音），缓冲完整保留，EndOfResponse 兜底用
    f = BilingualFlow()
    ja, zh, _ = f.feed("こんにちは。")
    assert ja == "" and zh == ""
    assert f.mode == "none"
    assert f.buf == "こんにちは。"
    # 连续多段也完整累积
    ja, zh, _ = f.feed("今日はいい天気ですね。")
    assert ja == "" and zh == ""
    assert f.buf == "こんにちは。今日はいい天気ですね。"


def test_reset():
    f = BilingualFlow()
    f.feed("【日语】こんにちは。")
    f.feed("【译文】你好。")
    f.reset()
    assert f.mode == "none" and f.buf == "" and f.perf_buf == ""
    ja, zh, _ = f.feed("【日语】次は？")
    # "次は？" 恰好 3 字符 = 安全前缀长度，全部暂留等下一个 chunk
    assert ja == ""
    ja, zh, _ = f.feed("【译文】次はどう思う？")
    assert ja == "次は？" and zh == "次はどう思う？"


def test_end_to_end_join():
    # 模拟流式 chunk（逐句/逐字符边界）端到端拼接，日语/译文还原无损
    f = BilingualFlow()
    chunks = [
        "【日语】そうですね。",
        "確かにその通りです。",
        "【译文】是啊。",
        "确实如此。",
    ]
    ja_parts, zh_parts = [], []
    for c in chunks:
        ja, zh, _ = f.feed(c)
        ja_parts.append(ja)
        zh_parts.append(zh)
    assert "".join(ja_parts) == "そうですね。確かにその通りです。"
    assert "".join(zh_parts) == "是啊。确实如此。"


def test_perf_section_basic():
    # 【演出】在【译文】之后：不进日语/译文，perf_buf 累积
    f = BilingualFlow()
    ja, zh, perf = f.feed("【日语】こんにちは。")
    ja2, zh2, perf2 = f.feed("【译文】你好。")
    ja3, zh3, perf3 = f.feed("【演出】expression:1; motion:1a; pause:800; pose:A")
    assert ja3 == "" and zh3 == ""
    assert "".join([ja, ja2, ja3]) == "こんにちは。"
    assert "".join([zh, zh2, zh3]) == "你好。"
    assert f.perf_buf == "expression:1; motion:1a; pause:800; pose:A"
    assert f.mode == "perf"


def test_perf_marker_split():
    # 【演出】标记本身被 chunk 切开：演出段收尾留安全前缀，补齐后进 perf_buf
    f = BilingualFlow()
    ja, zh, perf = f.feed("【日语】はい。")
    # "はい。" 3 字符 = 安全前缀长度，暂留
    assert ja == "" and zh == "" and perf == ""
    ja, zh, perf = f.feed("【译文】好的。")
    assert ja == "はい。" and zh == "好的。"  # 译文立即输出（zh 尾段无常驻延迟）
    ja, zh, perf = f.feed("【演")
    assert ja == "" and zh == "" and perf == ""  # 标记前缀暂留，不进译文
    ja, zh, perf = f.feed("出】motion:1b")
    assert ja == "" and zh == "" and perf == "motion:1b"
    assert f.perf_buf == "motion:1b"
    assert f.mode == "perf"


def test_perf_no_markers_fallback_strip():
    # LLM 只给了【演出】没给【日语】/【译文】：none 模式直接命中 PERF → perf 模式
    f = BilingualFlow()
    ja, zh, perf = f.feed("【演出】pose:B")
    assert ja == "" and zh == ""
    assert f.perf_buf == "pose:B"
    assert f.mode == "perf"
    # strip_perf：兜底朗读前剥掉演出段
    assert BilingualFlow.strip_perf("こんにちは【演出】pose:B") == "こんにちは"
    assert BilingualFlow.strip_perf("【演出】pose:B") == ""
    assert BilingualFlow.strip_perf("普通の文") == "普通の文"


def test_perf_after_zh_stream():
    # 流式：译文输出中途插入【演出】，译文止于标记、演出段入 perf_buf
    f = BilingualFlow()
    ja, zh, _ = f.feed("【日语】そうだね。")
    assert ja == "そう"  # 尾部 3 字符安全前缀暂留
    ja, zh, _ = f.feed("【译文】说得")
    assert ja == "だね。" and zh == "说得"  # 译文立即输出
    ja, zh, perf = f.feed("是。【演出】expression:8")
    assert zh == "是。"  # 译文补齐至标记
    assert f.perf_buf == "expression:8"
    assert f.mode == "perf"


def test_reset_clears_perf():
    f = BilingualFlow()
    f.feed("【日语】はい。")
    f.feed("【译文】好的。")
    f.feed("【演出】motion:sweat")
    assert f.perf_buf == "motion:sweat"
    f.reset()
    assert f.perf_buf == "" and f.mode == "none"


if __name__ == "__main__":
    import traceback

    tests = [
        test_full_stream,
        test_marker_split_across_chunks,
        test_no_markers_keeps_buffer,
        test_reset,
        test_end_to_end_join,
        test_perf_section_basic,
        test_perf_marker_split,
        test_perf_no_markers_fallback_strip,
        test_perf_after_zh_stream,
        test_reset_clears_perf,
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
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
