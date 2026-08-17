# 本地 WSL2 部署踩坑记录

2026-08-09 在本地 Windows + WSL2（Ubuntu，仓库根即 VoxEMW 项目源码）+ RTX 5080
部署 2dlive 数字人时踩过的坑与解法。环境前提：开发副本在 Windows 桌面 git 仓库，
运行副本（无 git）在 WSL2，`scripts/sync_to_wsl.sh` 同步（排除 .venv/.git/data/logs/.env.local）。

## 坑 1：wsl.exe 会话退出会杀掉 nohup 后台进程

**现象**：`wsl.exe -- bash -c 'nohup python -m ... &'` 返回后，服务进程随之消失
（nohup 挡不住 WSL 会话结束时的整树清理）。start_assistant.sh 在 WSL 里无法直接用
`wsl.exe` 触发——必须用 setsid 脱离会话。

**解法**（唯一可靠方式）：

```bash
setsid nohup .venv/bin/python -m voxemw.pipeline.launch --config configs/assistant.yaml \
    >> logs/pipeline.log 2>&1 < /dev/null & disown
```

即 `setsid`（新会话）+ `nohup`（防挂断）+ 重定向 + `disown`（防 shell 清理）。
三条都要。`systemd-run` 在 WSL 非 root 下会因 polkit 交互认证被拒，不可用。

## 坑 2：TTS `optimize: true` 崩溃——WSL 无 nvcc

**现象**：VoxCPM2 模型加载完成后，pipeline 崩在 torch.compile（Inductor）：
`PermissionError: [Errno 13] Permission denied: 'nvcc'`。
pip 装的 torch 只带 CUDA 运行库，**不带 CUDA toolkit（nvcc）**；AutoDL 服务器
有完整 toolkit 所以没问题，本地 WSL2 没有。

**解法**：`configs/assistant.yaml` 里 `tts.optimize: false`。
代码注释原文：「torch.compile：启动慢、首次合成快；RTF 0.3 不编译也够用」。
代价仅是首句合成略慢，无 nvcc 时是唯一选择。（若想保留 compile，
需 apt 装 nvidia-cuda-toolkit 或复用 pixi env 的 nvcc，但版本必须匹配 torch。）

## 坑 3：speech-to-speech 0.2.11 缺 silero-vad 依赖

**现象**：pipeline 启动崩：`ModuleNotFoundError: No module named 'silero_vad.utils_vad'`
（VADHandler 里 `torch.hub.load` 加载本地缓存 hub 时，hubconf.py 导入 silero_vad 包失败）。

**解法**：`pip install -i https://mirrors.aliyun.com/pypi/simple/ silero-vad`。
requirements.txt 没列它（上游 hub 加载方式隐式依赖），重装环境后需手动补。

## 坑 4：requirements.txt 的 git+ 依赖国内装不了

**现象**：`speech-to-speech @ git+https://github.com/huggingface/speech-to-speech.git@5b443c8`
——GitHub 被墙，pip 直连失败。

**解法**：阿里云镜像有 PyPI 版 `speech-to-speech==0.2.11`（回滚锚点）：
`pip install -i https://mirrors.aliyun.com/pypi/simple/ speech-to-speech==0.2.11`。
注意 0.2.11 是上游 main @5b443c8 之前的老版（缺 #391 投机话轮修复），
行为差异见 `docs/upgrade-regression.md`；本地够用，需要修复时再想办法走代理。

## 坑 5：wxPython 无 wheel 编译失败（tha3 连带）

**现象**：`pip install tha3` 连带解析出 wxPython，编译源码失败：
`ERROR: Failed building wheel for wxpython`（该包在 Linux 上无预编译 wheel，
需要 GTK 全套开发库）。tha3 本体是纯 PyTorch 推理，**运行不需要 wxPython**。

**解法**：`pip install tha3 --no-deps`，手动补它真正需要的依赖
（torch/numpy 等，项目里已有）。装 wxPython 会拉一整套 GUI 依赖，勿碰。

## 坑 6：Git Bash 编辑过的 shell 脚本变 CRLF，WSL 报语法错误

**现象**：用 Edit 工具（或 Windows 侧编辑器）改过 `start_assistant.sh` 后，
WSL 里 `bash -n` 报：`syntax error near unexpected token 'elif'`——
文件行尾被转成 CRLF，bash 把 `\r` 当命令内容。

**解法**：git 仓库内文件保持 LF：
`python -c "open('scripts/start_assistant.sh','wb').write(open('scripts/start_assistant.sh','rb').read().replace(b'\r\n',b'\n'))"`
改完 shell 脚本后养成检查习惯：`file scripts/*.sh | grep CRLF`。

## 坑 7：Windows 本地 miniconda 的 aiohttp SSL 初始化损坏

**现象**：Windows 侧直接跑 orchestrator：
`ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]`（conda OpenSSL 与 Windows 证书库不兼容，
`load_default_certs` 崩）。**与项目代码无关**——本项目在 WSL2 跑，Windows 侧
只做开发编辑。

**解法**：不需要修。验证/运行一律进 WSL2。Windows 侧如需快速验证 API 逻辑，
用 `python -m pytest`（纯逻辑单测不 import aiohttp）。

## 坑 8：wsl.exe 复合命令输出被吞 / 引号地狱

**现象**：长命令（heredoc、嵌套引号、`&` 后台）经 `wsl.exe -- bash -c '...'`
传入时：输出可能整段消失、`$HOME` 展开成空、`/tmp` 被 Git Bash 解析成 Windows 路径。

**解法**：两步走——先在 WSL 里用简单命令写脚本文件，再执行脚本：

```bash
wsl.exe -d Ubuntu -- bash -c 'cat > /tmp/start.sh <<"EOF"
...复杂逻辑...
EOF'
wsl.exe -d Ubuntu -- bash -c 'bash /tmp/start.sh'
```

避免在一条 wsl 调用里塞 heredoc/嵌套引号。进程查询用 `pgrep -af voxemw`，
端口用 `ss -tlnp | grep 8765`。

## 坑 10：TTS 崩在 ffmpeg（PermissionError）——脚本漏 PATH

**现象**：对话时 LLM 正常回复（日志有 output_tokens）、但**无音频下发**——
浏览器没声音、嘴不动。pipeline.log：
`Error during VoxCPM generation: [Errno 13] Permission denied: 'ffmpeg'`
（tts_voxcpm.py `_AtempoStretcher`，rate=0.886 语速补偿每句调 ffmpeg atempo 管道）。

**根因**：ffmpeg 是软链 `~/.local/bin/ffmpeg`（pixi env 自带），**不在默认 PATH**。
原版 `start_assistant.sh` 有 `export PATH="$HOME/.local/bin:$PATH"`，新写的
`start_assistant_wsl.sh` 漏了这行 → TTS 语速补偿崩 → 整条回复无音频。

**解法**：启动脚本必须带 `export PATH="$HOME/.local/bin:$PATH"`。
排查时先 `which ffmpeg`（空 = PATH 问题）。

## 坑 9：test_load_fillers 在 Windows 失败（预存）

**现象**：`tests/test_orchestrator.py::test_load_fillers` 断言失败——fillers 台词的
相对路径键是 `positive/b.wav`（POSIX），Windows 上 `Path.relative_to` 生成
`positive\b.wav`（反斜杠），texts.json 查不到键 → 台词变空串。**WSL2 下通过**，
与任何业务改动无关，属 Windows 本地环境差异，忽略即可。

## 环境速查

- 启动（WSL2 内执行，进程全部 setsid 脱离会话）：
  ```bash
  cd ~/VoxEMW
  pkill -f "voxemw.pipeline.launch|voxemw.avatar.orchestrator" 2>/dev/null; sleep 1
  setsid nohup .venv/bin/python -m voxemw.pipeline.launch --config configs/assistant.yaml >> logs/pipeline.log 2>&1 < /dev/null & disown
  setsid nohup .venv/bin/python -m voxemw.avatar.orchestrator --config configs/assistant.yaml >> logs/orchestrator.log 2>&1 < /dev/null & disown
  ```
- 等待就绪：pipeline 打 `Uvicorn running on http://127.0.0.1:8765`（约 3-4 分钟，
  VoxCPM 4.7G 模型加载）；orchestrator :8000 就绪后浏览器开 `http://localhost:8000`
- 一键启动脚本：`bash scripts/start_assistant_wsl.sh`（2026-08-09 新增，见同目录）
- 相关文档：`docs/upgrade-regression.md`（上游升级回归）
