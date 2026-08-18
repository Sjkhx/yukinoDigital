# 模型/大文件下载说明

以下目录因体积过大**未上传**到本仓库：

- `GPTV2proplus/`（雪之下雪乃 GPT-SoVITS V2ProPlus 权重、参考音频等，约 325MB）
- 如你本地还有 `yukino/` 或 `Yukino/` 大目录（GPT-SoVITS 权重资产），同样未上传

## 下载地址

请到以下页面下载对应的模型包：

https://www.ai-hobbyist.com/forum.php?mod=viewthread&tid=159980&page=1&mobile=no

## 放置位置

下载解压后，按原来的目录结构放回仓库根目录即可，例如：

```text
GPTV2proplus/
  └── 雪之下雪乃/
      ├── yukino-e15.ckpt
      ├── yukino_e8_s1744.pth
      └── reference_audios/...
```

如果你使用的是 WSL 侧推理环境，请把权重放到 WSL 仓库对应路径（如 `~/VoxEMW/GPTV2proplus/`）或按 `scripts/start_assistant_wsl.sh` 里配置的路径放置。
