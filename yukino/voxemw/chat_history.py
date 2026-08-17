"""SQLite 对话历史存储：像常见 AI 产品一样按会话（session）管理消息。

- 数据库默认落在 REPO_ROOT/log/chat_history.db（本地 log 文件夹）。
- conversations: 一次浏览器连接 = 一个会话；首次写入消息时才会创建会话记录，
  避免浏览器自动重连/空连接产生大量空会话。
- messages: 每轮用户消息与助手回复各一行，含译文/情绪（如有）。
- 纯标准库 sqlite3，不引入额外依赖；写操作有 threading.Lock 串行化。
"""

from __future__ import annotations

import datetime
import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    session_id  TEXT PRIMARY KEY,
    persona     TEXT NOT NULL DEFAULT '',
    title       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
    content     TEXT NOT NULL,
    translation TEXT NOT NULL DEFAULT '',
    emotion     TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES conversations(session_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ChatHistoryStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.executescript(SCHEMA)

    def _ensure_conversation(self, conn: sqlite3.Connection, session_id: str, persona: str) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO conversations(session_id, persona, title, created_at, updated_at) "
            "VALUES (?, ?, '', ?, ?)",
            (session_id, persona, _now(), _now()),
        )

    def add_message(self, session_id: str, role: str, content: str,
                    translation: str = "", emotion: str = "",
                    persona: str = "yukino") -> None:
        content = (content or "").strip()
        if not content:
            return
        with self._lock:
            with self._connect() as conn:
                self._ensure_conversation(conn, session_id, persona)
                conn.execute(
                    "INSERT INTO messages(session_id, role, content, translation, emotion, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (session_id, role, content, translation or "", emotion or "", _now()),
                )
                # 用第一条用户消息前 30 字自动命名会话标题
                if role == "user":
                    title = content[:30] + ("…" if len(content) > 30 else "")
                    conn.execute(
                        "UPDATE conversations SET title = CASE WHEN title='' THEN ? ELSE title END, "
                        "updated_at = ? WHERE session_id = ?",
                        (title, _now(), session_id),
                    )
                else:
                    conn.execute(
                        "UPDATE conversations SET updated_at = ? WHERE session_id = ?",
                        (_now(), session_id),
                    )

    def list_conversations(self, limit: int = 50) -> list[dict]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT c.session_id, c.persona, c.title, c.created_at, c.updated_at,
                           COUNT(m.id) AS message_count,
                           (SELECT m2.content FROM messages m2
                            WHERE m2.session_id = c.session_id
                            ORDER BY m2.id DESC LIMIT 1) AS last_message
                    FROM conversations c
                    LEFT JOIN messages m ON m.session_id = c.session_id
                    GROUP BY c.session_id
                    ORDER BY c.updated_at DESC, c.created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def get_messages(self, session_id: str) -> list[dict]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT id, role, content, translation, emotion, created_at "
                    "FROM messages WHERE session_id = ? ORDER BY id ASC",
                    (session_id,),
                ).fetchall()
        return [dict(r) for r in rows]

    def delete_conversation(self, session_id: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
                return cur.rowcount > 0

    def close(self) -> None:
        # 每个操作自开自关连接，这里仅保留接口用于未来常驻连接
        pass


def create_chat_history_store(config: dict | None = None) -> ChatHistoryStore:
    """从配置创建存储。默认 db_path = REPO_ROOT/log/chat_history.db。"""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    cfg = (config or {}).get("history") or {}
    db_path = Path(cfg.get("db_path", repo_root / "log" / "chat_history.db"))
    if not db_path.is_absolute():
        db_path = repo_root / db_path
    return ChatHistoryStore(db_path)
