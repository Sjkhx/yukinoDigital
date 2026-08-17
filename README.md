# yukino-dsh

把雪之下雪乃（VoxEMW 数字人语音助手）接入 DSH web 的插件仓库。

- **插件**：`lib/client.js` 注入 DSH web，新增 `/yukino` 路由（iframe 嵌入雪乃页面），并在 DSH 会话 `running → done` 时把任务总结 POST 到雪乃 orchestrator。
- **雪乃服务（VoxEMW）**：`yukino/` 子目录收编的完整数字人语音聊天助手源码与素材（Python orchestrator + Live2D 前端 + 配置 + 人设 + 技能 + 素材）。

```
yukino-dsh-plugin/
├── lib/                    # DSH 插件本体（client.js / index.js）
├── cordis.patch.yml        # DSH profile 补丁（启用 yukino-dsh 等）
├── package.json            # npm 包声明（DSH 发现的入口）
├── yukino/                 # ★ VoxEMW 雪乃完整源码+素材（本仓库主体）
│   ├── voxemw/             #   Python 核心包（orchestrator / s2s 管线 / avatar）
│   ├── web/                #   Live2D 前端（无构建纯静态，iframe 内容源）
│   ├── yukino2d/           #   另一套 2D 立绘前端
│   ├── scripts/            #   启动/工具脚本（相对路径 + env，已可移植）
│   ├── configs/assistant.yaml # 唯一配置（七积木选型 + 服务端口）
│   ├── personas/ skills/ assets/ 春物角色 live2d-win/
│   ├── requirements.txt    #   Python 依赖
│   └── README.md           #   VoxEMW 自身的使用文档
└── MODEL_DOWNLOAD.md
```

## 一、安装 DSH 插件

在 DSH profile 目录装配本插件（需已安装 pnpm）：

```bash
cd <你的 profile 目录>   # 例如 C:\Users\<you>\.dsh\profiles\web
pnpm install   # 依赖 package.json 里 file: 指向本仓库的 yukino-dsh
```

profile 的 `cordis.patch.yml` 里启用：

```yaml
- id: yukino-dsh
  name: 'yukino-dsh'
```

> **开发时改插件的坑**：DSH host 读的是
> `C:\Users\<you>\.dsh\profiles\web\node_modules\yukino-dsh\lib\client.js`，
> 它是**独立副本不是 symlink**——改 `lib/client.js` 后必须手动复制过去才能生效：
> ```bash
> cp lib/client.js C:\Users\<you>\.dsh\profiles\web\node_modules\yukino-dsh\lib\client.js
> ```
> 浏览器强刷后即可（`?rev=` 内容哈希变化，无需重启 DSH）。

## 二、装配雪乃服务（VoxEMW）

### 1. Python 环境（uv 管理）

```bash
cd yukino
uv venv .venv --python 3.12
```

- **最小集**（Windows 上跑 orchestrator，含记忆积木所需的 qdrant + OpenAI 兼容 API）：
  ```bash
  uv pip install --python .venv aiohttp websockets pyyaml numpy qdrant-client scipy openai
  ```
- **全量**（语音管线 VoxCPM/FunASR/THA3 等，GPU 机器/WSL 用；连带安装 torch，建议参考 `scripts/autodl_setup.sh` 按卡型单独装 torch 后再执行）：
  ```bash
  uv pip install --python .venv -r requirements.txt
  ```

> 记忆积木（`voxemw/memory.py`）是 qdrant LocalMode 向量库 + OpenAI 兼容 embedding API 实现，不依赖 mem0ai / sentence-transformers（旧积木）。

### 2. 语音模型下载

GPT-SoVITS 权重（雪乃音色，约 325MB）太大不进仓库，按 [`MODEL_DOWNLOAD.md`](yukino/MODEL_DOWNLOAD.md) 下载并放回 `yukino/GPTV2proplus/`。

### 3. 环境变量

复制 `.env.example` 为 `.env.local` 并填入密钥（LLM / ASR / 记忆 embedding）：

```bash
cp yukino/.env.example yukino/.env.local
# 编辑填入 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY / MEMORY_EMBEDDER_API_KEY
```

> `.env.local` 已被 gitignore，永不入库。

## 三、启动

### 方式 A（推荐）：DSH 内使用

1. 启动雪乃 orchestrator（独立进程，Windows 或 WSL 均可）：
   ```bash
   cd yukino
   uv run --python .venv python -m voxemw.avatar.orchestrator --config configs/assistant.yaml
   ```
   确认 `http://127.0.0.1:8000/` 可访问。
2. 打开 DSH web，访问路由 `#/yukino`，页面 iframe 即嵌入雪乃。

### 方式 B：完整语音链路（含 GPT-SoVITS + s2s 管线）

在 WSL2 内一键启停：

```bash
cd yukino
bash scripts/start_assistant_wsl.sh          # 启动 8899 → 8765 → 8000
bash scripts/start_assistant_wsl.sh stop     # 停止
bash scripts/start_assistant_wsl.sh status   # 状态
```

## 四、配置项

- 七积木选型（VAD/STT/LLM/TTS/Avatar/Persona/Memory）都在 `yukino/configs/assistant.yaml`，当前默认 `avatar.backend=2dlive`（前端本地渲染，无需后端数字人进程）。
- 路径约定：所有脚本/配置已相对化（相对 `yukino/` 根）；GPT-SoVITS 安装位置用 `GPT_SOVITS_ROOT` / `GPT_SOVITS_PY` 环境变量覆盖（默认 `~/GPT-SoVITS`）。

## 五、开发与维护

- 插件卡死修复：任务完成监听在流式高频推送下走 cordis 服务代理（`ctx.sessions.list`）会占满主线程，现已缓存 list 引用 + `requestAnimationFrame` 合并扫描（见 `lib/client.js` 的 `createTaskDoneWatcher`），每帧最多扫描一次。改 `lib/client.js` 后记得同步副本（见上文）。

## License

MIT。雪乃形象与音色素材来自用户授权渠道，见 `yukino/LICENSE` 与各素材目录说明；GPT-SoVITS 权重按 `MODEL_DOWNLOAD.md` 的授权渠道获取。