# VoxEMW WSL2 安装指南（新用户）

从零到跑起来，约 1-2 小时（大头在 GPT-SoVITS 环境与模型下载）。安装分两段：

- **可自动化**（venv + 依赖）：`bash scripts/setup_wsl.sh`
- **需手动**（GPT-SoVITS conda 环境 + 代码 + 雪乃权重）：本文第 3 步

## 1. 硬件要求

- **NVIDIA GPU**（必须）：GPT-SoVITS V2ProPlus 需要 CUDA 推理。RTX 30/40/50 系均可；
  RTX 50 系（sm_120）必须用 cu128 版 torch（setup 脚本默认就是）。
- 显存建议 **8G+**。
- 磁盘：GPT-SoVITS 权重 ~325MB + torch ~2GB + 各模型缓存，预留 20G+。
- 内存 16G+。

> 无 GPU 时：前端静态形象 + 聊天面板仍可显示，但**没有语音对话**（TTS 需要 GPU）。

## 2. 前置

1. **WSL2 + Ubuntu 22.04/24.04**（`wsl --install -d Ubuntu-22.04`）。
2. **Windows NVIDIA 驱动**：装好驱动后 WSL 内 `nvidia-smi` 应能显示显卡。
3. **miniconda**：`https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/` 装到 `~/miniconda3`。
4. **系统 python3.10+ 与 venv**：Ubuntu 22.04 自带 python3.10；缺 venv 模块先
   `sudo apt install python3-venv python3-pip`。
5. 建议把仓库放到 WSL 内（如 `~/VoxEMW`）。若放 `/mnt/c/...`，I/O 慢且模型加载可能卡。

## 3. GPT-SoVITS V2ProPlus（手动，一次性）

语音合成靠它。分三块：conda 环境 / 代码 / 雪乃权重。

### 3.1 conda 环境（myenv + torch cu130）

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n myenv python=3.10
conda activate myenv
# 国内可加清华源：conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu130
```

> torch 版本以 GPT-SoVITS-V2ProPlus 代码要求的为准；RTX 50 系需 cu128+。

### 3.2 GPT-SoVITS 代码

```bash
cd ~
git clone https://github.com/jdc4429/GPT-SoVITS-V2ProPlus-Windows GPT-SoVITS
cd GPT-SoVITS
# 按该仓库 README 安装其依赖（pip install -r requirements.txt 等，在 myenv 里）
```

> 国内 clone 慢可用镜像：`git clone https://ghproxy.com/https://github.com/jdc4429/GPT-SoVITS-V2ProPlus-Windows GPT-SoVITS`

### 3.3 雪乃 V2ProPlus 权重

按 `docs/MODEL_DOWNLOAD.md` 从论坛下载模型包，解压到：

```text
~/GPT-SoVITS/GPT_SoVITS/pretrained_models/yukino/
  ├── yukino-e15.ckpt          # GPT 权重
  ├── yukino_e8_s1744.pth      # SoVITS
  └── reference_audios/...     # 参考音频
```

权重目录名 `yukino` 是 `scripts/gptsovits_v2proplus_server.py` 写死的，勿改名。

## 4. VoxEMW 依赖（自动化）

在仓库根执行：

```bash
bash scripts/setup_wsl.sh
```

脚本会（幂等，重复执行安全）：
1. 创建 `.venv`（python3.10+）
2. 装 torch 2.8.0 cu128（RTX 50 系必需）
3. 装 `requirements.txt`（默认阿里云镜像，`PIP_INDEX_URL` 可覆盖；
   GitHub 被墙时 `speech-to-speech` 自动回退 pip 版 0.2.11）
4. 补装 `silero-vad`、`tha3 --no-deps`（WSL 特殊坑，见 `docs/wsl2-troubleshooting.md` 坑 3/5）
5. 生成 `.env.local` 并检查 API key
6. 检查 GPT-SoVITS 三件套是否齐（第 3 步），缺失给指引

只检查不安装：`bash scripts/setup_wsl.sh check`。

## 5. 配置 API key（.env.local）

```bash
cp .env.example .env.local   # setup 脚本也会自动生成
# 编辑 .env.local 填 key
```

| 变量 | 用途 | 申请 |
|---|---|---|
| `DEEPSEEK_API_KEY` | LLM 对话（必填） | platform.deepseek.com |
| `DASHSCOPE_API_KEY` | ASR 语音识别（必填，stt 走阿里云百炼） | bailian.console.aliyun.com |
| `MEMORY_EMBEDDER_API_KEY` | 记忆 embedding（memory 用） | 见 configs/assistant.yaml memory 段 |

> memory 不想配：`configs/assistant.yaml` 里 `memory.enabled: false`。

## 6. 启动

```bash
bash scripts/start_assistant_wsl.sh
```

脚本依次启动 GPT-SoVITS（:8899）→ 语音管线（:8765）→ orchestrator（:8000），
等待各自就绪。浏览器打开：

```text
http://localhost:8000
```

- 停止：`bash scripts/start_assistant_wsl.sh stop`
- 状态：`bash scripts/start_assistant_wsl.sh status`
- 日志：`logs/{gptsovits_v2proplus,pipeline,orchestrator}.log`

## 7. 排障速查

- 启动顺序/就绪日志问题：`docs/wsl2-troubleshooting.md`（WSL 全套坑，重点坑 1/3/4/5/10）
- 上游升级回归：`docs/upgrade-regression.md`
- 权重缺失/放置：`docs/MODEL_DOWNLOAD.md`
- 前端改动不生效（改了 web/ 但雪乃没变化）：坑 11——改的是仓库前端，WSL 运行副本
  要同步（`//wsl.localhost/Ubuntu-22.04/home/<user>/VoxEMW/web/`）并升 `index.html` 的 `?v=`。

## 8. 可覆盖路径（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `GPT_SOVITS_PY` | `~/miniconda3/envs/myenv/bin/python` | GPT-SoVITS 环境 python |
| `GPT_SOVITS_ROOT` | `~/GPT-SoVITS` | GPT-SoVITS 代码根 |
| `VOXEMW_CONFIG` | `configs/assistant.yaml` | 主配置 |
| `PIP_INDEX_URL` | 阿里云镜像 | pip 源 |
| `SETUP_BASE_PY` | `python3` | 建 venv 的 python（setup 脚本用） |

## 相关链接

- GPT-SoVITS V2ProPlus 代码：https://github.com/jdc4429/GPT-SoVITS-V2ProPlus-Windows
- 雪乃 V2ProPlus 权重：见 `docs/MODEL_DOWNLOAD.md`
