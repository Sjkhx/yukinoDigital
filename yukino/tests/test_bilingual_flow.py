"""BilingualFlow 流式切分器测试：「【日语】…【译文】…」LLM 输出格式。

覆盖：完整段、标记被 chunk 切开、无标记回退（none 模式）、重置、端到端拼接。
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from voxemw.pipeline.tts_voxcpm import BilingualFlow  # noqa: E402


def test_full_stream():
    f = BilingualFlow()
    ja, zh = f.feed("【日语】こんにちは。")
    # 安全前缀（防【译文】被切开）暂留尾部 len(ZH)-1=3 字符：迟一个 chunk 输出
    assert ja == "こんに"
    ja2, zh2 = f.feed("【译文】你好。")
    assert ja2 == "ちは。"  # 上一段暂留的字符补齐
    assert zh2 == "你好。"
    assert f.mode == "zh"


def test_marker_split_across_chunks():
    f = BilingualFlow()
    ja, zh = f.feed("【日")
    assert ja == "" and zh == ""
    ja, zh = f.feed("语】お元気ですか？")
    # 尾部"すか？"被当可能被切开的标记前缀暂留
    assert ja == "お元気で"
    ja, zh = f.feed("【译")
    assert ja == "すか"  # 暂留的日语字符先输出
    ja, zh = f.feed("文】你还好吗？")
    assert ja == "？" and zh == "你还好吗？"  # 最后暂留字符补齐


def test_no_markers_keeps_buffer():
    # LLM 未按格式输出：增量恒空（TTS 静音），缓冲完整保留，EndOfResponse 兜底用
    f = BilingualFlow()
    ja, zh = f.feed("こんにちは。")
    assert ja == "" and zh == ""
    assert f.mode == "none"
    assert f.buf == "こんにちは。"
    # 连续多段也完整累积
    ja, zh = f.feed("今日はいい天気ですね。")
    assert ja == "" and zh == ""
    assert f.buf == "こんにちは。今日はいい天気ですね。"


def test_reset():
    f = BilingualFlow()
    f.feed("【日语】こんにちは。")
    f.feed("【译文】你好。")
    f.reset()
    assert f.mode == "none" and f.buf == ""
    ja, zh = f.feed("【日语】次は？")
    # "次は？" 恰好 3 字符 = 安全前缀长度，全部暂留等下一个 chunk
    assert ja == ""
    ja, zh = f.feed("【译文】次はどう思う？")
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
        ja, zh = f.feed(c)
        ja_parts.append(ja)
        zh_parts.append(zh)
    assert "".join(ja_parts) == "そうですね。確かにその通りです。"
    assert "".join(zh_parts) == "是啊。确实如此。"


if __name__ == "__main__":
    import traceback

    tests = [
        test_full_stream,
        test_marker_split_across_chunks,
        test_no_markers_keeps_buffer,
        test_reset,
        test_end_to_end_join,
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
