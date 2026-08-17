"""记忆积木：核心资料文档（精确层）+ 对话 summary 向量库（笼统层）。

两类记忆（2026-08-11 分层设计）：
1. 核心资料文档（精确）：data/memory/core/<persona>.md，用户持久关键信息
   （称呼/性别/职业/偏好/约定等）。对话后 DeepSeek 提取合并更新，对话前
   整文档注入 instructions（最优先）。人可读、模型直读、零检索成本。
2. 对话 summary 向量库（笼统）：每轮对话 → DeepSeek 压缩成 2-3 句摘要 →
   qwen3.7-text-embedding API 向量化 → qdrant LocalMode 落盘（data/memory/
   vectors）；会话开始时固定查询向量化、按相似度召回 top_k 条注入。
   2026-08-11 自建，替代 Mem0：其事实抽取丢失大部分对话信息，且 DeepSeek
   响应的抽取 JSON 解析频繁失败（"Unterminated string"，日志可见），
   导致对话根本进不了向量库。

- 写入：response.done 后异步（core 提取 + summary 生成，不占语音延迟）
- 降级：embedding/LLM API 失败静默跳过（该轮不入库，不阻塞对话）
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_MEMORY_CHARS = 80  # 单条注入截断（防人设正文被稀释）

CORE_HEADER = "# 用户核心资料"  # core.md 标题行（注入/校验用）

CORE_EXTRACT_PROMPT = """你是记忆管家。根据【最新对话】，更新关于用户的【核心资料】文档。

核心资料 = 用户持久的关键个人信息，例如：称呼/名字、性别、年龄、职业、所在地、\
性格与偏好、重要关系、长期目标、重要约定（如"用户请雪乃吃饭""约好出去玩"这类\
跨会话持续有意义的事）。不含一次性的闲聊内容。

规则：
1. 保留现有文档中仍然有效的信息
2. 从最新对话中提取新的核心信息（没有新信息则不添加）
3. 删除已过时或与事实矛盾的信息
4. 时间表述一律用绝对日期（如 2026-08-15、8月15日），对话中的「今天/明天/周末/周X」
   按「今天是 {today}（{weekday}）」换算后写入——与向量库摘要的日期口径一致，
   避免两处记忆时间冲突
5. 输出【完整】的更新后文档（Markdown 列表，首行为「{header}」），只输出文档内容，不要任何解释"""

SUMMARY_PROMPT = """你是记忆摘要助手。把下面的对话压缩成 2-3 句中文摘要。

要求：
- 保留具体信息：对话主题、用户提到的个人信息、决定、约定、偏好、情绪
- 时间锚点必须保留：对话中出现的"今天/明天/周末/周X"等相对时间，换算成
  绝对日期写进摘要（如"2026-08-15 周六爬山"）；无法换算时保留原始表述并注明
  参照日（如"周一(8/10)说的周末爬山"）
- 第三人称客观陈述（用户/雪乃）
- 只输出摘要文本，不要任何解释或标题

今天是 {today}（{weekday}）。"""

# 召回查询：向量化后与所有历史摘要算相似度（固定串即可，语义空间稳定）。
# 近期事件类问题（"我周日要干嘛"）靠时间兜底命中——search 合并最近 RECENT_TOP 条。
SEARCH_QUERY = "用户与雪乃的对话内容、约定、计划安排、时间约定"
RECENT_TOP = 3  # 召回时额外注入最近 N 条摘要（时间兜底，覆盖"我周日要干嘛"这类近期事件问题）


def build_memory_block(memories: list[str]) -> str:
    """记忆条目 → 注入 instructions 的文本块（纯函数，便于单测）。"""
    if not memories:
        return ""
    lines = [f"- {m[:MAX_MEMORY_CHARS]}" for m in memories]
    return "# 关于用户的记忆（历史对话摘要，自然引用，不要逐条复述）\n" + "\n".join(lines)


def build_core_block(core_text: str) -> str:
    """核心资料文档 → 注入 instructions 的文本块（纯函数，便于单测）。

    核心资料是精确的个人信息，注入时标题醒目、要求模型直接采用（区别于
    向量召回记忆的"自然引用"）。
    """
    if not core_text.strip():
        return ""
    return core_text.strip() + "\n\n# 使用说明\n以上是用户的核心资料，回答与用户相关的问题时直接依据它。\n"


class MemoryStore:
    """核心资料文档 + summary 向量库。所有方法同步阻塞，调用方用 asyncio.to_thread。"""

    def __init__(self, cfg: dict, llm_cfg: dict, api_key: str, embed_api_key: str):
        import os

        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        store_dir = Path(cfg.get("store_dir", "data/memory"))
        store_dir.mkdir(parents=True, exist_ok=True)
        self._core_dir = store_dir / "core"  # 核心资料文档目录（按 persona 一个 .md）
        self._llm_model = llm_cfg.get("model_name", "deepseek-v4-flash")
        self._llm_base_url = llm_cfg.get("base_url", "https://api.deepseek.com/v1")
        self._llm_api_key = api_key
        self.user_id = cfg.get("user_id", "default_user")
        self.top_k = int(cfg.get("top_k", 5))

        # summary 向量库：qdrant LocalMode（独立目录，避免与旧 Mem0 实例锁冲突）
        self._vectors_dir = store_dir / "vectors"
        self._vectors_dir.mkdir(parents=True, exist_ok=True)
        self._qdrant = QdrantClient(path=str(self._vectors_dir))
        self._collection = "summary"
        self._embed_dims = int(cfg.get("embedder_dims", 1024))
        # 已存在则复用（重启不重建，保记忆）
        if self._collection not in [c.name for c in self._qdrant.get_collections().collections]:
            self._qdrant.recreate_collection(
                self._collection,
                vectors_config=VectorParams(size=self._embed_dims, distance=Distance.COSINE),
            )
        # embedding 客户端（OpenAI 兼容）
        from openai import OpenAI

        self._embed_client = OpenAI(
            api_key=embed_api_key,
            base_url=cfg.get("embedder_base_url", "https://api.aiqingxuan.top/v1"),
        )
        self._embed_model = cfg.get("embedder_model", "qwen3.7-text-embedding")
        os.environ.setdefault("VOXEMW_MEMORY_INIT", "1")

    # ── 核心资料文档（精确层）──

    def core_path(self, agent_id: str) -> Path:
        """核心资料文件路径（按 persona 隔离，data/memory/core/<persona>.md）。"""
        return self._core_dir / f"{agent_id}.md"

    def load_core(self, agent_id: str) -> str:
        """读取核心资料文档（无文件/空 → 空串）。"""
        try:
            return self.core_path(agent_id).read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def update_core(self, agent_id: str, user_text: str, assistant_text: str) -> None:
        """一轮对话 → DeepSeek 提取/合并 → 原子写回核心资料文档（阻塞）。

        失败静默（保留旧文档）：LLM 异常/输出非法都不破坏已有资料。
        """
        try:
            import os

            from openai import OpenAI

            import datetime

            cur = self.load_core(agent_id) or f"{CORE_HEADER}\n（暂无资料）"
            dialog = "\n".join(
                f"{'用户' if r == 'user' else '雪乃'}：{t}" for r, t in
                (("user", user_text), ("assistant", assistant_text)) if t
            )
            if not dialog:
                return
            now = datetime.datetime.now()
            weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
            client = OpenAI(api_key=self._llm_api_key, base_url=self._llm_base_url)
            resp = client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": CORE_EXTRACT_PROMPT.format(
                        header=CORE_HEADER,
                        today=f"{now.year}-{now.month:02d}-{now.day:02d}",
                        weekday=weekdays[now.weekday()])},
                    {"role": "user", "content": f"现有文档：\n{cur}\n\n最新对话：\n{dialog}"},
                ],
                temperature=0.0,
                max_tokens=1200,
            )
            new_text = (resp.choices[0].message.content or "").strip()
            # 校验：非空、以标题开头、长度合理，否则保留旧文档
            if (new_text.startswith(CORE_HEADER) and 5 < len(new_text) < 3000):
                self._core_dir.mkdir(parents=True, exist_ok=True)
                p = self.core_path(agent_id)
                tmp = p.with_suffix(".tmp")
                tmp.write_text(new_text + "\n", encoding="utf-8")
                tmp.replace(p)
                logger.info("核心资料更新（%s）：%d 字符", agent_id, len(new_text))
            else:
                logger.warning("核心资料提取输出非法，保留旧文档: %r", new_text[:60])
        except Exception as e:
            logger.warning("核心资料更新失败（保留旧文档）: %s", e)

    # ── 对话 summary 向量库（笼统层）──

    def _embed(self, text: str) -> list[float]:
        resp = self._embed_client.embeddings.create(model=self._embed_model, input=[text])
        return resp.data[0].embedding

    def _summarize(self, user_text: str, assistant_text: str) -> str:
        """DeepSeek 压缩对话为摘要；失败降级为原文拼接截断（不阻塞入库）。"""
        import datetime

        dialog = "\n".join(
            f"{'用户' if r == 'user' else '雪乃'}：{t}" for r, t in
            (("user", user_text), ("assistant", assistant_text)) if t
        )
        now = datetime.datetime.now()
        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        try:
            import os

            from openai import OpenAI

            client = OpenAI(api_key=self._llm_api_key, base_url=self._llm_base_url)
            resp = client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": SUMMARY_PROMPT.format(
                        today=f"{now.year}-{now.month:02d}-{now.day:02d}",
                        weekday=weekdays[now.weekday()])},
                    {"role": "user", "content": dialog},
                ],
                temperature=0.0,
                max_tokens=400,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as e:
            logger.warning("摘要生成失败，降级原文截断: %s", e)
        return dialog[:400]

    def add_turn(self, user_text: str, assistant_text: str, agent_id: str) -> None:
        """一轮对话 → 摘要 → 向量化 → qdrant 落盘（阻塞，失败静默）。"""
        try:
            import time

            summary = self._summarize(user_text, assistant_text)
            if not summary:
                return
            vec = self._embed(summary)
            point_id = int(hashlib.md5(f"{agent_id}:{summary}".encode("utf-8")).hexdigest()[:15], 16)
            from qdrant_client.models import PointStruct

            self._qdrant.upsert(
                self._collection,
                points=[PointStruct(
                    id=point_id,
                    vector=vec,
                    payload={
                        "agent_id": agent_id,
                        "user_id": self.user_id,
                        "summary": summary,
                        "ts": time.time(),  # 最近召回兜底用
                    },
                )],
            )
            logger.info("对话摘要入库（%s）：%d 字符", agent_id, len(summary))
        except Exception as e:
            logger.warning("对话摘要入库失败（跳过该轮）: %s", e)

    def list_summaries(self, agent_id: str, limit: int = 100) -> list[dict]:
        """列出该 persona 的摘要（按时间倒序，最新在前）。查看/编辑用（/api/memory）。"""
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            hits = self._qdrant.scroll(
                self._collection,
                limit=limit,
                scroll_filter=Filter(must=[
                    FieldCondition(key="agent_id", match=MatchValue(value=agent_id)),
                ]),
                with_payload=True,
                order_by="ts",
            )[0]
            return [
                {
                    "id": str(p.id),
                    "time": p.payload.get("ts"),
                    "summary": p.payload.get("summary", ""),
                }
                for p in reversed(hits)  # order_by 倒序 → 反转成最新在前
            ]
        except Exception as e:
            logger.warning("摘要列表读取失败: %s", e)
            return []

    def delete_summary(self, agent_id: str, point_id: str) -> bool:
        """删除一条摘要（按 qdrant point id）。"""
        try:
            self._qdrant.delete(self._collection, points_selector=[int(point_id)])
            logger.info("摘要已删除（%s，id=%s）", agent_id, point_id)
            return True
        except Exception as e:
            logger.warning("摘要删除失败: %s", e)
            return False

    def update_summary(self, agent_id: str, point_id: str, new_text: str) -> bool:
        """修改一条摘要：重新向量化后覆盖原 point（保持时间戳与 id）。"""
        try:
            new_text = new_text.strip()
            if not new_text:
                return False
            vec = self._embed(new_text)
            # 读原 payload 保 ts/user_id
            old = self._qdrant.retrieve(self._collection, ids=[int(point_id)], with_payload=True)
            if not old:
                logger.warning("摘要不存在（id=%s），跳过修改", point_id)
                return False
            payload = dict(old[0].payload)
            payload["summary"] = new_text
            from qdrant_client.models import PointStruct

            self._qdrant.upsert(
                self._collection,
                points=[PointStruct(id=int(point_id), vector=vec, payload=payload)],
            )
            logger.info("摘要已修改（%s，id=%s）：%d 字符", agent_id, point_id, len(new_text))
            return True
        except Exception as e:
            logger.warning("摘要修改失败: %s", e)
            return False

    def save_core(self, agent_id: str, new_text: str) -> bool:
        """手动保存核心资料文档（美化页面编辑用）。非法输入拒绝。"""
        new_text = new_text.strip()
        if not new_text.startswith(CORE_HEADER) or len(new_text) > 5000:
            return False
        try:
            self._core_dir.mkdir(parents=True, exist_ok=True)
            p = self.core_path(agent_id)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(new_text + "\n", encoding="utf-8")
            tmp.replace(p)
            logger.info("核心资料手动保存（%s）：%d 字符", agent_id, len(new_text))
            return True
        except Exception as e:
            logger.warning("核心资料保存失败: %s", e)
            return False

    def search(self, agent_id: str) -> list[str]:
        """相似度召回 top_k 条 + 最近 RECENT_TOP 条（时间兜底），去重返回。

        近期事件问题（"我周日要干嘛"）与固定查询串的相似度未必高，靠最近
        摘要兜底命中；历史话题靠向量相似度。"""
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            vec = self._embed(SEARCH_QUERY)
            # qdrant-client 1.19：本地模式用 query_points（search 已移除）
            res = self._qdrant.query_points(
                self._collection,
                query=vec,
                limit=self.top_k,
                query_filter=Filter(must=[
                    FieldCondition(key="agent_id", match=MatchValue(value=agent_id)),
                ]),
            )
            out: list[str] = []
            seen: set[str] = set()
            for h in res.points:
                s = h.payload.get("summary")
                if s and s not in seen:
                    seen.add(s)
                    out.append(s)
            # 最近 N 条兜底（scroll 按时间倒序取）
            recent = self._qdrant.scroll(
                self._collection,
                limit=RECENT_TOP,
                scroll_filter=Filter(must=[
                    FieldCondition(key="agent_id", match=MatchValue(value=agent_id)),
                ]),
                with_payload=True,
                order_by="ts",  # 按时间戳倒序（qdrant 1.19 支持 order_by）
            )[0]
            for p in reversed(recent):  # 倒序取最近
                s = p.payload.get("summary")
                if s and s not in seen:
                    seen.add(s)
                    out.append(s)
            return out
        except Exception as e:
            logger.warning("记忆召回失败（跳过）: %s", e)
            return []


def create_memory_store(config: dict) -> MemoryStore | None:
    """从全局配置构建记忆仓库；未启用/依赖缺失/初始化失败 → None（静默降级）。"""
    cfg = config.get("memory") or {}
    if not cfg.get("enabled", False):
        return None
    import os

    llm_cfg = config.get("llm") or {}
    api_key = os.environ.get(llm_cfg.get("api_key_env", "DEEPSEEK_API_KEY"), "")
    if not api_key:
        logger.warning("memory 启用但缺 LLM api_key，记忆关闭")
        return None
    embed_api_key = os.environ.get(cfg.get("embedder_api_key_env", "MEMORY_EMBEDDER_API_KEY"), "")
    if not embed_api_key:
        logger.warning("memory 启用但缺 embedding api_key（%s），记忆关闭",
                       cfg.get("embedder_api_key_env", "MEMORY_EMBEDDER_API_KEY"))
        return None
    import time

    # qdrant LocalMode 锁冲突重试：重启时旧进程未完全退出会占锁，
    # 等 2s 重试（最多 3 次），避免记忆静默降级（2026-08-11 事故）
    for attempt in range(3):
        try:
            store = MemoryStore(cfg, llm_cfg, api_key, embed_api_key)
            logger.info("记忆积木就绪（core+summary 向量库，store=%s，embedder=%s）",
                        cfg.get("store_dir", "data/memory"), cfg.get("embedder_model", "qwen3.7-text-embedding"))
            return store
        except Exception as e:
            if "already accessed" in str(e) and attempt < 2:
                logger.warning("记忆初始化 qdrant 锁冲突（第 %d 次），%.0fs 后重试", attempt + 1, 2.0)
                time.sleep(2)
                continue
            logger.warning("记忆积木初始化失败，降级关闭: %s", e)
            return None
    return None
