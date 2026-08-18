"""pending_speech（task-done 待播报音频）纯逻辑单测：sqlite 读写，无需 aiohttp。"""

from voxemw.chat_history import ChatHistoryStore


def test_pending_speech_roundtrip(tmp_path):
    store = ChatHistoryStore(tmp_path / "h.db")
    assert store.list_unplayed_speeches() == []

    store.add_pending_speech("dsh-tasks-20260818", "こんにちは", "你好", b"\x01\x02\x03")
    store.add_pending_speech("dsh-tasks-20260818", "おはよう", "早上好", b"\x04\x05")

    items = store.list_unplayed_speeches()
    assert len(items) == 2
    assert items[0]["ja_text"] == "こんにちは"
    assert items[0]["pcm"] == b"\x01\x02\x03"
    assert items[0]["played"] == 0

    store.mark_speech_played(items[0]["id"])
    remaining = store.list_unplayed_speeches()
    assert len(remaining) == 1
    assert remaining[0]["ja_text"] == "おはよう"


def test_pending_speech_limit(tmp_path):
    store = ChatHistoryStore(tmp_path / "h.db")
    for i in range(5):
        store.add_pending_speech("s", f"t{i}", "", b"\x00")
    items = store.list_unplayed_speeches(2)
    assert len(items) == 2
    assert items[0]["ja_text"] == "t0"  # 按 id 升序，最早的先补播


def test_pending_speech_old_db_upgrade(tmp_path):
    """老库无 pending_speech 表时也能自动建表（CREATE TABLE IF NOT EXISTS）。"""
    import sqlite3

    db = tmp_path / "h.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE conversations (session_id TEXT PRIMARY KEY, persona TEXT, "
        "title TEXT, created_at TEXT, updated_at TEXT)")
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, "
        "role TEXT, content TEXT, translation TEXT, emotion TEXT, created_at TEXT)")
    conn.commit()
    conn.close()

    store = ChatHistoryStore(db)
    store.add_pending_speech("s", "こんにちは", "你好", b"\x01")
    assert len(store.list_unplayed_speeches()) == 1
