# 构建环境备忘（本机机器相关，非设计文档）

> 记录本机构建 kevcpp 所需的环境事实与已知坑。内容随时间/机器变化，仅作备忘。

## 编译器与构建工具
- VS2022 Community，MSBuild 位于：
  `d:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe`
- CMake 生成器：`Visual Studio 17 2022`（x64）
- 构建时若 MSBuild 文件跟踪器被沙箱阻断（`CL.exe exit code 2` / `TRK0002`），追加：
  `-p:TrackFileAccess=false`
- **注意**：改源码后若 MSBuild 因 obj 时间戳比源旧而跳过重编，需把源文件 touch 到当前时间：
  `(Get-Item src\xxx.cpp).LastWriteTime = Get-Date`，或用 `-p:TrackFileAccess=false` 重建。

## Vulkan SDK（重编 kev-backend-vulkan.dll 用）
- 本机 Vulkan SDK 路径：`D:\SDK\Vulkan\1.4.363.0`
- 用途：开启 `GGML_VULKAN` / `KEV_BUILD_BACKEND_VULKAN` 重编 `kev-backend-vulkan.dll`。
- 背景：logits 判别通道的 Vulkan 后端源码已实现 `read_logits/vocab_n_tokens/n_batch`
  （`src/kev_backend_vulkan.cpp`）并注册进 ops 表；但现有 `kev-backend-vulkan.dll`（旧，
  早于 ops 表新增 `read_logits` 等字段）加载会报 “plugin ABI older than host”，
  需用此 SDK 重新编译该 dll 才能让 `--backend vulkan` 跑 logits 通道。

## 模型文件（本机）
- `models\Qwen3.5-0.8B-Q8_0.gguf`：原版 0.8B 基座（logits 通道默认，零微调）。
- `models\kev-merged-q8.gguf` + `models\head.bin`：原 kev（LoRA 折叠 + PointerHead）路径旧模型。

## 其他
- 模型权重 / build / logs 已按 `.gitignore` 忽略，不进 git。
- 本机 logits 通道相关实验 fixtures 曾置于 `config/ab/`、`p0reg/`（已并入仓库）。