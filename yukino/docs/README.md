# VoxEMW · 数字人实时语音聊天助手

对着浏览器说话，屏幕里的雪之下雪乃（Live2D Cubism 4）开口回答你。
人设、音色、形象、记忆、对话历史五位一体。当前默认形象为前端本地渲染的
雪乃 Live2D 模型（`avatar.backend=2dlive`），不依赖后端 GPU 数字人服务。

语音链路基于 [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech)
（VAD → STT → LLM → TTS 实时管线），TTS 使用 GPT-SoVITS V2ProPlus 雪乃音色。

## 架构

```
浏览器（web/，Live2D 前端渲染 + 聊天面板 + 历史抽屉）
   │  ws :8000
   ▼
orchestrator（CPU，Windows 或 WSL 均可）
   ├─→ s2s 语音管线（:8765）  VAD → STT → LLM → TTS
   └─→ avatar.backend=2dlive 时无 avatar ws（口型由浏览器本地音频分析驱动）
```

- s2s 管线：WSL2 内 `voxemw.pipeline.launch`（:8765）
- GPT-SoVITS V2ProPlus：WSL2 内常驻服务（:8899）
- orchestrator：`voxemw.avatar.orchestrator`（:8000，浏览器唯一入口）

## 七块积木（configs/assistant.yaml）

| 积木 | 当前选型 |
|---|---|
| vad | silero-vad（s2s 内置，判停 500ms） |
| stt | `qwen3asr_flash`（阿里云百炼 DashScope Qwen3-ASR-Flash） |
| llm | DeepSeek `deepseek-v4-flash`（chat-completions，流式逐句送 TTS） |
| tts | `gptsovits`（GPT-SoVITS V2ProPlus，WSL 常驻服务 127.0.0.1:8899） |
| avatar | `2dlive`（雪乃 Live2D Cubism 4，前端 WebGL 本地渲染） |
| persona | `personas/<id>.md`（女娲蒸馏，frontmatter 绑定音色三件套） |
| memory | 自建记忆（DeepSeek 抽取 + qdrant LocalMode 向量库，默认开） |
| history | SQLite 对话历史（`log/chat_history.db`，每轮自动写入） |

## 快速启动

环境基于 WSL2（TTS/语音管线需 CUDA GPU）。首次安装见 `docs/SETUP_WSL.md`，
之后启动只需一条命令：

```bash
# 进入 VoxEMW 目录（仓库的 yukino/ 子目录，或独立克隆的仓库根）
cd yukino

# 首次：安装环境（建 venv、装依赖、生成 .env.local，幂等）
bash scripts/setup_wsl.sh

# 启动三服务：GPT-SoVITS（:8899）→ s2s 管线（:8765）→ orchestrator（:8000）
bash scripts/start_assistant_wsl.sh
```

打开：

```text
http://127.0.0.1:8000/
```

> orchestrator（:8000）也可在 Windows 上单独跑：
> `python -m voxemw.avatar.orchestrator`（`avatar.backend=2dlive` 前端本地渲染，
> 无需后端 GPU；但完整语音对话仍需 WSL 里的 GPT-SoVITS 与 s2s 管线）。

## 对话历史

- 页面左侧 🕘 按钮展开/收起历史抽屉；
- 支持 **＋ 新会话**、查看历史消息、删除会话；
- 数据保存在本地 SQLite：`log/chat_history.db`；
- 调试 API：
  - `GET /api/history`
  - `GET /api/history/{session_id}`
  - `DELETE /api/history/{session_id}`
- 独立管理页：`http://127.0.0.1:8000/history`

## Live2D 雪乃

- 模型资产在 `web/live2d/`（Cubism 4 .moc3 + pixi-live2d-display 运行时）；
- 默认制服，`?outfit=shihuku` 切换私服；
- 待机有自然微动，说话时有轻微点头/身体摆动；
- 待机姿势自动轮换（Pose1/2/3，每 18~32s），切换时手臂 ease + 动作淡入一次平滑过渡；
- 前端内置情绪 agent，根据本轮日语回复自动选择表情/动作；
- `web/yukino2d.js` 含运行时 bug 补丁（pixi-live2d-display 忽略 `Meta.Loop` → 补 `setIsLoop`、
  循环曲线首尾对齐、手臂噪声过渡包络、动作播放在过渡开始即启动）。改前端后需同步 WSL
  运行副本并升 `index.html` 的 `?v=` 版本号，见 `docs/wsl2-troubleshooting.md` 坑 11；
- 暴露全局 API 供 agent 调用：

```js
window.yukino2dAgent.playMotion("yuk_ikari")
window.yukino2dAgent.setExpression("em1")
window.yukino2dAgent.setPose("pose2")
window.yukino2dAgent.listMotions()
```

## 大文件下载

`GPTV2proplus/`（雪之下雪乃 GPT-SoVITS V2ProPlus 权重）未上传，下载与放置说明见：

```text
MODEL_DOWNLOAD.md
```

下载地址：

```text
https://www.ai-hobbyist.com/forum.php?mod=viewthread&tid=159980&page=1&mobile=no
```

## 环境变量（.env.local）

```text
DEEPSEEK_API_KEY=...          # LLM
DASHSCOPE_API_KEY=...         # qwen3asr_flash STT
MEMORY_EMBEDDER_API_KEY=...   # memory embedding（可选）
```

## 目录

- `configs/assistant.yaml` — 唯一配置（积木 + server + history）
- `voxemw/pipeline/` — s2s 集成（STT/TTS 积木 + 启动器）
- `voxemw/avatar/` — orchestrator + 数字人服务
- `voxemw/chat_history.py` — SQLite 对话历史存储
- `web/` — 通话页（无构建：Live2D 引擎 + 聊天 + 历史抽屉）
- `web/live2d/` — 雪乃 Cubism 4 模型与运行时
- `personas/` — 人设（雪乃/峰哥）
- `yukino-perspective/` — 雪乃人设研究资料
- `春物角色 live2d-win/` — 春物 Live2D 原始素材与一色/结衣/雪乃 win 版
- `log/` — 本地 SQLite 历史库（已 gitignore）
- `tests/` — 纯逻辑单测

## 合规

音色与肖像素材由使用者本人提供/授权；AI 生成内容需标注，不得用于冒充、欺诈。

## 相关链接

- speech-to-speech: https://github.com/huggingface/speech-to-speech
- GPT-SoVITS V2ProPlus 代码: https://github.com/jdc4429/GPT-SoVITS-V2ProPlus-Windows
- DeepSeek API: https://platform.deepseek.com
- 阿里云百炼 DashScope: https://bailian.console.aliyun.com
- 雪乃 GPT-SoVITS V2ProPlus 模型包: https://www.ai-hobbyist.com/forum.php?mod=viewthread&tid=159980&page=1&mobile=no
