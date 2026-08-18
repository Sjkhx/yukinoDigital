# 模型/大文件下载说明

以下目录因体积过大**未上传**到本仓库：

- `GPTV2proplus/`（雪之下雪乃 GPT-SoVITS V2ProPlus 权重、参考音频等，约 325MB）
- 如你本地还有 `yukino/` 或 `Yukino/` 大目录（GPT-SoVITS 权重资产），同样未上传

## 下载地址

请到以下页面下载对应的模型包：

https://www.ai-hobbyist.com/forum.php?mod=viewthread&tid=159980&page=1&mobile=no

## 放置位置

> **重要**：推理服务 `scripts/gptsovits_v2proplus_server.py` 实际读取的路径是
> `$GPT_SOVITS_ROOT/GPT_SoVITS/pretrained_models/yukino/`（`GPT_SOVITS_ROOT` 默认 `~/GPT-SoVITS`），
> **不是**下载包的 `GPTV2proplus/` 目录。下载解压后需按下面步骤放置，服务才能加载到权重。

下载解压后得到 `GPTV2proplus/雪之下雪乃/` 目录（含 `yukino-e15.ckpt`、`yukino_e8_s1744.pth`、`reference_audios/`）。

**WSL2 推理环境（默认）**，把权重放到 GPT-SoVITS 安装目录下：

```text
~/GPT-SoVITS/GPT_SoVITS/pretrained_models/yukino/
  ├── yukino-e15.ckpt
  ├── yukino_e8_s1744.pth
  └── reference_audios/日语/emotions/   # 3 条情绪参考音频
```

即把下载包里的两个权重文件拷到 `pretrained_models/yukino/`，`reference_audios/` 一并拷入。

若 GPT-SoVITS 装在别处，用环境变量 `GPT_SOVITS_ROOT` 覆盖，对应路径变为 `$GPT_SOVITS_ROOT/GPT_SoVITS/pretrained_models/yukino/`。
