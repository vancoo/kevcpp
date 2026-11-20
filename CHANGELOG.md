# CHANGELOG

本文件记录 kevcpp 各版本的显著变化。格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [SemVer](https://semver.org/lang/zh-CN/)。

格式约定：
- **Added** — 新增能力 / 后端 / 文件
- **Changed** — 行为 / 产物 / 接口变化
- **Fixed** — 修复
- **Removed** — 移除

---

## [0.7.1] — 2026-11-20

### Fixed

- **修复 Windows 包 Vulkan 后端静默回退 CPU 的严重 bug**（v0.7.0 回归）：
  - 根因：v0.7.0 的 CPU 变体 KAT 探测在 GPU 插件初始化前通过 `ggml_backend_load()`
    预注册了 `ggml-cpu-*.dll`，使后端 registry 非空；而 `llama_backend_init()` 只在
    registry 为空时才调用 `ggml_backend_load_all()`（`third_party/llama.cpp/src/llama.cpp`），
    导致随包分发的 `ggml-vulkan.dll` 永不被发现、`vulkan-1.dll` 永不加载，模型全部落在
    CPU 上且无任何告警（实测 GPU 引擎零活动）。
  - 修复：新增 `src/kev_backend_dynload.h`，在 Vulkan/CUDA 插件 `llama_backend_init()`
    之后若无已注册 GPU 设备，则显式 `LoadLibrary`/`dlopen` GPU 库、解析 reg getter
    （`ggml_backend_vk_reg` / `ggml_backend_cuda_reg`）并 `ggml_backend_register()` 注册
    （本构建布局下 GPU 库导出 reg getter 而非 `ggml_backend_init`，故不能用
    `ggml_backend_load()`）；已注册 GPU 设备时为 no-op，失败时给出明确告警而非静默降级。
  - `kev_backend_vulkan.cpp` / `kev_backend_cuda.cpp` 接入该 helper；未来 Windows CUDA
    构建同样免疫。
  - 实测（AMD Radeon 8060S iGPU，Gemma E2B IQ4_XS，JevBench 231 题）：
    GPU offload 恢复生效，P50 1.003s → 0.204s（~5×），P95 11.91s → 2.27s，
    正确率 145/231 → 146/231（数值路径差异）。

---

## [未发布]（build-portable-backends 分支，HEAD `d961b1c`）

> 本版本围绕**可插拔后端**落地：把 HTTP 服务端产物统一命名为 `kevcpp.exe`，并新增 Vulkan 后端插件，
> 使本机（AMD iGPU gfx1151）无需 ROCm/HIP 即可把整个模型 offload 到 Vulkan 设备上推理。

### Changed

- **服务端产物改名 `main_server.exe` → `kevcpp.exe`**
  - `CMakeLists.txt`：为 `main_server` 目标设置 `OUTPUT_NAME "kevcpp"`（目标名保持 `main_server`，
    不破坏 `scripts/build.ps1` 的 `.vcxproj` 引用）。
  - 产物路径：`build/bin/Release/kevcpp.exe`。
  - 用法/接口、`--backend cpu` 默认路径均不变。

### Added

- **Vulkan 后端插件 `kev-backend-vulkan.dll`**
  - 新增 `src/kev_backend_vulkan.cpp`：`kev_backend.h` C-ABI 实现，把**整个模型 offload 到 Vulkan 设备**
    （`n_gpu_layers=-1`）；纯 MSVC 编译，需在 **GGML_VULKAN 独立构建目录**（`KEV_BUILD_BACKEND_VULKAN=ON`）
    中构建。
  - `CMakeLists.txt`：新增 `KEV_BUILD_BACKEND_VULKAN` 选项（保持默认 OFF，不污染默认 CPU 构建目录）；
    新增 `kev-backend-vulkan` SHARED 目标（链接同一份 `llama`/`ggml`，single-llama rule）。
  - `src/kev_backend_factory.cpp`：`BackendFactory_load("vulkan")` 现在加载 `kev-backend-vulkan.dll`
    （宿主 `KEV_BACKEND_USE_DLL` 时），不再直接报「未构建」。
  - 复用已有 CPU 后端基座，tokenizer/seq-memory/尺寸接口与 CPU 后端一致（B-1/B-2 委托 tokenizer）。
  - 构建脚本：`build/vulkan-exp/build_vulkan.ps1`（独立目录 + 单节点 msbuild）。
  - 快速使用说明写入 `README.md` §7。

- **文档**
  - 新增 `CHANGELOG.md`（本文件）。
  - `README.md`：新增「Vulkan 快速使用」章节、更新架构说明（宿主 `kevcpp.exe` + 可插拔 dll 两栈）、
    更新当前状态 / CLI / 限制章节以反映 Vulkan 后端与产物改名。

- **`.gitignore`**：新增对临时/实验产物的忽略规则（`logs/`、`microbench/`、`repro_*.json`、
  `tests/tmp_*`、`kab_*`、`*.docx` 等），保持仓库干净。

### Fixed

- `README.md` 旧文案中 `main_server.exe`、以及「Vulkan 为未构建扩展点 / ARCHIVED」等过时描述已更正。

### Removed / 未纳入

- 未纳入版本管理：实验/复现中间产物（`logs/`、`microbench/`、`repro_*.json`、`tests/tmp_*` 等）。
- HIP/ROCm GPU（`--backend gpu/cuda`）仍为扩展点，未在本机默认启用（ROCm 会话在 gfx1151 上 HIP/HSA blit
  初始化崩溃 `0xC0000005`）；本机 GPU 路径以 Vulkan 为主。

---

## 变更记录（历史，供定位）

> 以下为 `master` 稳定主线与此前便携后端分支的脉络速记，完整说明见 `docs/` 与 `git log`。

### 固定 / 基线

- `dd51ed7` `fix(server)`：默认 `--ctx -1`（落到 8192）+ 超长 state 清晰报错。
- `267300f` `perf`：砍未使用 tied lm_head（Rank-1）。
- `2607590` `perf`：`ggml concat_f32` threaded + vectorized。
- 稳态延迟：短请求 median ~327 ms，输出与参考逐位一致（bit-identical）。

### 可插拔后端脉络

- `71d2c64` `feat`：可插拔 `kev::Backend` C-ABI + `--backend` + `GGML_CPU_ALL_VARIANTS` 便携 SIMD 方案
  （VNNI dll 必须剪除，见 `docs/F.24_...`）。
- `ac394b2` `feat(gpu)`：进程内 GGML_HIP GPU 后端雏形 + `--backend gpu` 接线。
- `d961b1c` `feat(server)`：服务端产物改名 `kevcpp.exe`，新增 Vulkan 后端（本文件主条目）。

---

## 说明

- 分发便携 CPU 时**必须剪除 VNNI dll**（`ggml-cpu-{cascadelake,icelake,alderlake}.dll`），否则
  AVX512_VNNI `vpdpbusd` Q8×Q8 内核在本模型上返回错误概率（见 `docs/portable_backend_deployment.md` §4）。
- 所有许可证以各自上游（vendored llama.cpp、kev）为准。