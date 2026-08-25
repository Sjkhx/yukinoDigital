# VoxEMW 安装指南（Windows + WSL2 双机部署）

从零到跑起来，约 1-2 小时（大头在 GPT-SoVITS 环境与模型下载）。当前部署形态：

- **Windows**：仓库主体 + `.venv`，跑 pipeline（:8765）+ orchestrator（:8000），
  读仓库代码改完即生效
- **WSL2**：只装 GPT-SoVITS V2ProPlus（conda 环境 + 代码 + 雪乃权重），常驻 :8899

## 1. 硬件要求

- **NVIDIA GPU（WSL2 侧必需）**：GPT-SoVITS V2ProPlus 需要 CUDA 推理。RTX 30/40/50 系均可；
  RTX 50 系（sm_120）必须用 cu128+ 版 torch。
- Windows 侧 pipeline/orchestrator 是 **CPU 进程**（TTS 推理在 WSL 的 GPU 上），普通机器即可。
- 显存建议 **8G+**。
- 磁盘：GPT-SoVITS 权重 ~325MB + torch ~2GB + 各模型缓存，预留 20G+。
- 内存 16G+。

> 无 GPU 时：前端静态形象 + 聊天面板仍可显示，但**没有语音对话**（TTS 需要 GPU）。

## 2. WSL2 前置

1. **WSL2 + Ubuntu 22.04/24.04**（`wsl --install -d Ubuntu-22.04`）。
2. **Windows NVIDIA 驱动**：装好驱动后 WSL 内 `nvidia-smi` 应能显示显卡。
3. **miniconda**：`https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/` 装到 `~/miniconda3`。

> 仓库本体放 Windows（如 `D:\...\yukino`）即可，无需再复制进 WSL。
> `scripts/sync_to_wsl.sh` 的「开发副本 → WSL 运行副本」双副本流程已停用。

## 3. Windows 侧：venv + 依赖（一次性）

pipeline/orchestrator 在 Windows 跑，读仓库代码。仓库根建 venv，装最小依赖集
（STT 走 DashScope API、TTS 走 :8899 HTTP，**不需要** voxcpm/funasr/tha3 等本地模型依赖）：

```powershell
cd "D:\...\yukino"          # 仓库根
uv venv                     # 建 .venv（uv 管理，py3.12；无 uv 用 python -m venv .venv）
# torch 用 CPU 版（Windows PyPI 默认即 CPU；TTS 推理不在本机）
.\.venv\Scripts\python.exe -m pip install torch torchaudio -i https://mirrors.aliyun.com/pypi/simple
.\.venv\Scripts\python.exe -m pip install speech-to-speech==0.2.11 silero-vad aiohttp websockets uvicorn fastapi pyyaml numpy scipy -i https://mirrors.aliyun.com/pypi/simple
```

- `speech-to-speech` 必须 **0.2.11**（pip 版；requirements.txt 里 git+ main 版仅 GPU 服务器用）
- `silero-vad` 上游隐式依赖，必须手动补（坑 3）
- 现役 `.venv` 是 uv 管理：`stop`/`status` 会按 cmdline 匹配 launcher shim + 解释器双进程

## 4. WSL 侧：GPT-SoVITS V2ProPlus（手动，一次性）

语音合成靠它，唯一留在 WSL 的服务。分三块：conda 环境 / 代码 / 雪乃权重。

### 4.1 conda 环境（myenv + torch cu130）

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n myenv python=3.10
conda activate myenv
# 国内可加清华源：conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu130
```

> torch 版本以 GPT-SoVITS-V2ProPlus 代码要求的为准；RTX 50 系需 cu128+。

### 4.2 GPT-SoVITS 代码

```bash
cd ~
git clone https://github.com/jdc4429/GPT-SoVITS-V2ProPlus-Windows GPT-SoVITS
cd GPT-SoVITS
# 按该仓库 README 安装其依赖（pip install -r requirements.txt 等，在 myenv 里）
```

> 国内 clone 慢可用镜像：`git clone https://ghproxy.com/https://github.com/jdc4429/GPT-SoVITS-V2ProPlus-Windows GPT-SoVITS`

### 4.3 雪乃 V2ProPlus 权重

按 `docs/MODEL_DOWNLOAD.md` 从论坛下载模型包，解压到：

```text
~/GPT-SoVITS/GPT_SoVITS/pretrained_models/yukino/
  ├── yukino-e15.ckpt          # GPT 权重
  ├── yukino_e8_s1744.pth      # SoVITS
  └── reference_audios/...     # 参考音频
```

权重目录名 `yukino` 是 `scripts/gptsovits_v2proplus_server.py` 写死的，勿改名。

装完可快速核对 WSL 侧环境：WSL 内仓库根执行 `bash scripts/setup_wsl.sh check`（检查
GPT-SoVITS 三件套；脚本里的 `.venv`/torch cu128 段为 GPU 服务器/旧形态所用，本地已不用，
报缺失忽略即可）。

## 5. 配置 API key（.env.local）

仓库根（Windows）复制生成：

```powershell
copy .env.example .env.local
# 编辑 .env.local 填 key
```

| 变量 | 用途 | 申请 |
|---|---|---|
| `DEEPSEEK_API_KEY` | LLM 对话（必填） | platform.deepseek.com |
| `DASHSCOPE_API_KEY` | ASR 语音识别（必填，stt 走阿里云百炼） | bailian.console.aliyun.com |
| `MEMORY_EMBEDDER_API_KEY` | 记忆 embedding（memory 用） | 见 configs/assistant.yaml memory 段 |

> memory 不想配：`configs/assistant.yaml` 里 `memory.enabled: false`。

## 6. 启动（两步）

先起 GPT-SoVITS，再起 Windows 两服务：

```bash
# 第 1 步（WSL2 内，进仓库目录）：GPT-SoVITS :8899
bash scripts/start_assistant_wsl.sh gptsovits
```

```powershell
# 第 2 步（Windows PowerShell，任意目录）：pipeline + orchestrator
powershell -File scripts\start_assistant_win.ps1
```

浏览器打开：

```text
http://localhost:8000
```

- 停止：Windows `powershell -File scripts\start_assistant_win.ps1 stop`；
  WSL `bash scripts/start_assistant_wsl.sh stop`（只影响 WSL 内进程）
- 状态：Windows `...ps1 status`；WSL `bash scripts/start_assistant_wsl.sh status`
- 日志：Windows `logs/{pipeline,orchestrator}.{out,err}.log`；
  WSL `logs/gptsovits_v2proplus.log`

## 7. 排障速查

- WSL 全套坑：`docs/wsl2-troubleshooting.md`（坑 1/3/4/5/10 仍适用；坑 2/11 及 WSL 起三服务
  的写法为 2026-08-19 前的旧形态，已过时）
- 上游升级回归：`docs/upgrade-regression.md`
- 权重缺失/放置：`docs/MODEL_DOWNLOAD.md`
- 前端改动不生效：orchestrator 在 Windows 直接读仓库 `web/`，改完硬刷新即可；仍旧则升
  `index.html` 的 `?v=` 版本号。

## 8. 可覆盖路径（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `GPT_SOVITS_PY` | `~/miniconda3/envs/myenv/bin/python` | GPT-SoVITS 环境 python（WSL） |
| `GPT_SOVITS_ROOT` | `~/GPT-SoVITS` | GPT-SoVITS 代码根（WSL） |
| `VOXEMW_CONFIG` | `configs/assistant.yaml` | 主配置 |

## 相关链接

- GPT-SoVITS V2ProPlus 代码：https://github.com/jdc4429/GPT-SoVITS-V2ProPlus-Windows
- 雪乃 V2ProPlus 权重：见 `docs/MODEL_DOWNLOAD.md`
