# yukino-dsh · 把雪之下雪乃接入 DSH

把「雪之下雪乃」（VoxEMW 数字人语音助手）接进 DSH（DeepSeek Harness）web 的插件仓库。
雪乃有 Live2D 形象、日语音色、人设、记忆、对话历史五位一体；既可以嵌入 DSH 右侧做陪伴面板，
也能独立打开全屏语音对话页。

- **插件本体**（`lib/`）：注入 DSH web，新增 `/yukino` 路由 + 右侧基础面板，DSH 任务完成时把总结交给雪乃点评。
- **雪乃服务**（`yukino/` 子目录）：完整数字人语音助手源码与素材（Python orchestrator + Live2D 前端 + 配置 + 人设 + 技能）。
- **当前版本**：0.1.0 · MIT License。

## 功能

- **双模式**
  - **基础模式**：嵌入 DSH 右侧 1/4 屏竖屏面板 —— 上部 Live2D、下部对话 + **打字输入**；面板可隐藏、可一键跳出到独立页（新窗口）。
  - **完整模式**：`/yukino` 全屏路由或独立打开 `http://127.0.0.1:8000/`，语音对话。
- **语音 + 打字**：语音走麦克风（VAD→STT→LLM→TTS），打字走文本进同一会话，可混用。
- **DSH 任务完成点评**：DSH 会话跑完，插件把任务总结 POST 给 orchestrator，雪乃用日语评论（先日语后译文）。
- **项目背景注入**：`yukino/log/project_context.md` 描述项目本身，每次会话注入雪乃上下文，被问"这项目做了什么"有依据。
- **Live2D 雪乃**：Cubism 4 前端本地渲染（`avatar.backend=2dlive`），**无需后端 GPU**；待机姿势自动轮换、情绪动作、口型同步。
- **记忆与历史**：mem0 长期记忆（LLM 抽取 + qdrant 向量库）+ SQLite 对话历史（`/history` 管理页）。

## 目录结构

```
yukino-dsh-plugin/
├── lib/                        # DSH 插件本体（client.js 注入 / index.js 入口）
├── cordis.patch.yml            # DSH profile 补丁（启用 yukino-dsh）
├── package.json                # npm 包声明（DSH 发现的入口）
├── yukino/                     # ★ VoxEMW 雪乃完整源码+素材（本仓库主体）
│   ├── voxemw/                 #   Python 核心包（orchestrator / s2s 管线 / avatar）
│   ├── web/                    #   Live2D 前端（无构建纯静态，iframe 内容源）
│   ├── scripts/                #   启动/安装/工具脚本
│   ├── configs/assistant.yaml  #   唯一配置（七积木选型 + 服务端口）
│   ├── personas/ skills/ assets/
│   ├── docs/                   #   SETUP_WSL / wsl2-troubleshooting / upgrade-regression / MODEL_DOWNLOAD
│   ├── tests/                  #   纯逻辑单测
│   ├── requirements.txt
│   └── docs/README.md          #   VoxEMW 自身文档
└── CLAUDE.md                   # 项目约定（AI 协作用）
```

## 快速开始（WSL2，含完整语音）

环境要求：WSL2 + NVIDIA GPU（GPT-SoVITS 需 CUDA）。首次安装见 [`yukino/docs/SETUP_WSL.md`](yukino/docs/SETUP_WSL.md)，之后：

```bash
cd yukino
bash scripts/setup_wsl.sh             # 首次：建 .venv、装依赖、生成 .env.local（幂等）
bash scripts/start_assistant_wsl.sh   # 启动三服务：GPT-SoVITS(:8899) → s2s(:8765) → orchestrator(:8000)
```

浏览器打开 `http://127.0.0.1:8000/`（完整页）或 `?compact=1`（紧凑面板版）。停止/状态：`start_assistant_wsl.sh stop|status`。

> orchestrator 也可在 Windows 直接跑：`python -m voxemw.avatar.orchestrator`（2dlive 前端本地渲染，无需 GPU；但完整语音仍需 WSL 的 GPT-SoVITS 与 s2s 管线）。

## 装配 DSH 插件（可选，用于嵌入 DSH）

在 DSH profile 目录装配（需已安装 pnpm）：

```bash
cd <你的 profile 目录>     # 例如 C:\Users\<you>\.dsh\profiles\web
pnpm install              # 依赖 package.json 里 file: 指向本仓库的 yukino-dsh
```

profile 的 `cordis.patch.yml` 里启用：

```yaml
- id: yukino-dsh
  name: 'yukino-dsh'
```

装上后 DSH web 右下角出现 ❄ 按钮，点开即基础面板；`#/yukino` 进全屏路由；任务完成自动触发雪乃点评。

> **开发注意**：DSH 读的是 `node_modules/yukino-dsh`，它是**指向本仓库的 junction**，改 `lib/` 源码实时生效，无需手动复制（浏览器强刷即可，`?rev=` 内容哈希会更新）。

## 配置

- 七积木选型（VAD/STT/LLM/TTS/Avatar/Persona/Memory）都在 `yukino/configs/assistant.yaml`；默认 `avatar.backend=2dlive`、STT 走阿里云百炼、LLM 走 DeepSeek、TTS 走 GPT-SoVITS。
- 环境变量（`.env.local`，已 gitignore）：`DEEPSEEK_API_KEY` / `DASHSCOPE_API_KEY` / `MEMORY_EMBEDDER_API_KEY`。
- 项目背景注入：自行创建 `yukino/log/project_context.md`（gitignored，不随仓库分发），描述项目本身；会话开始注入雪乃上下文，保存后重开会话生效。
- 路径约定：脚本/配置已相对化；GPT-SoVITS 安装位置用 `GPT_SOVITS_ROOT` / `GPT_SOVITS_PY` 覆盖。

## 文档

- [`yukino/docs/SETUP_WSL.md`](yukino/docs/SETUP_WSL.md) — 新用户完整安装指南
- [`yukino/docs/wsl2-troubleshooting.md`](yukino/docs/wsl2-troubleshooting.md) — WSL2 部署踩坑
- [`yukino/docs/upgrade-regression.md`](yukino/docs/upgrade-regression.md) — 上游 speech-to-speech 升级回归
- [`yukino/docs/MODEL_DOWNLOAD.md`](yukino/docs/MODEL_DOWNLOAD.md) — GPT-SoVITS 权重下载放置

## 致谢

- 感谢 B站 up主 [嘘暖liu](https://space.bilibili.com/1687568228) 提供的雪乃 Live2D 模型
- 感谢 [AI-Hobbyist](https://www.ai-hobbyist.com/forum.php?mod=viewthread&tid=159980&page=1&mobile=no) 提供的雪乃 GPT-SoVITS 语音权重

## License

MIT。雪乃形象与音色素材来自用户授权渠道，见 `yukino/LICENSE` 与各素材目录说明；GPT-SoVITS 权重按 `yukino/docs/MODEL_DOWNLOAD.md` 的授权渠道获取。
