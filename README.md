# kevcpp

> kev 的 C++17 原生实现（基于 llama.cpp），用 C++ 重新实现 kev（Python）的推理/服务行为。

kevcpp 是一个纯 CPU（目前）、基于 `llama.cpp` 的 kev 推理服务。它以 kev（`E:\van\projects\kev`）的 Python
参考实现为行为基准，把 `DecisionModel` / 状态前缀缓存 / `/v1/systemone` 服务语义用 C++/llama.cpp 重写，
目标是：**证明 kevcpp 的稳态延迟不低于 Python kev，并持续压缩稳态延迟**（项目章程）。

> 当前语言说明：本文以中文撰写，标题辅以英文，读者可对照 kev 与 llama.cpp 的术语。

---

## 目录（Table of Contents）

1. [当前状态（Current Status）](#1-当前状态current-status)
2. [基于 llama.cpp 与 kev（What We Reference）](#2-基于-llamacpp-与-kevwhat-we-reference)
3. [开发环境与验证（Environment & Validation）](#3-开发环境与验证environment--validation)
4. [相对 kev 的优势（Advantages over kev）](#4-相对-kev-的优势advantages-over-kev)
5. [不足与限制（Limitations，诚实说明）](#5-不足与限制limitations诚实说明)
6. [场景 / 依赖 / 环境要求（Usage, Dependencies, Requirements）](#6-场景--依赖--环境要求usage-dependencies-requirements)
7. [架构说明（Architecture）](#7-架构说明architecture)
8. [模型 / HF Hub（Model & HuggingFace）](#8-模型--hf-hubmodel--huggingface)

---

## 1. 当前状态（Current Status）

- **稳定主线（stable mainline）：`master` @ `dd51ed7`**。稳态延迟约 **\~327 ms**（短请求），输出与参考逐位一致（bit-identical）。包含三项已落地的优化：砍掉未使用的 tied lm*head（Rank-1，`267300f`）、ctx 默认 8192+超长 state 清晰报错（`dd51ed7`）、threaded/vectorized `ggml concat_f32`（`2607590`）。*\_head（Rank-1，`267300f`）、ctx 默认 8192+超长 state 清晰报错（`dd51ed7`）、threaded/vectorized `ggml concat_f32`（`2607590`）。
- **便携后端 / GPU 后端（portable backend）：已归档（ARCHIVED）于分支 `build-portable-backends` @ `71d2c64`**。可插拔后端设计（`kev::Backend` C-ABI + `--backend` + GGML*CPU*ALL\_VARIANTS 便携 SIMD 方案）源码已提交，但**尚未整合为最终共享库（shared-lib）构建**；分发时必须**剪除 VNNI dll**（见 §5）。
- 当前 CPU 硬性要求 **AVX-512**（Zen4+/Sapphire Rapids 级别），因为主构建为 `GGML_NATIVE`（详见 §5、§6）。

---

## 2. 基于 llama.cpp 与 kev（What We Reference）

kevcpp 站在两个上游之上：

### 2.1 llama.cpp（vendored，`third_party/llama.cpp`）

llama.cpp 以完整副本 vendored 于 `third_party/llama.cpp`（`CMakeLists.txt` 通过 `add_subdirectory` 引入，
未修改上游 `llama-graph.cpp`/`llama-context.cpp` 之外的本仓库改动之外的部分——vendored 层保持只读）。参考/复用的部分：

- **GGUF / GGML**：模型以 GGUF 计载（`models/kev-merged-q8.gguf`，Q8\_0 量化），`llama`/`ggml` 提供张量加载与运算。
- **CPU 计算内核**：Q8*0 tiled GEMM（`MUL_MAT`）、`SSM_CONV`、`GATED_DELTA_NET`、`FLASH_ATTN_EXT`、`RMS_NORM` 等。*\_0 tiled GEMM（`MUL_MAT`）、`SSM_CONV`、`GATED_DELTA_NET`、`FLASH_ATTN_EXT`、`RMS_NORM` 等。
- **混合（hybrid）模型构建器**：`src/models/qwen35.cpp` 构建 **24 层 = 18 层 Gated-DeltaNet 循环层 + 6 层全注意力层**，
  全注意力层位于第 3、7、11、15、19、23 层（`is_recr_impl[]` 非循环层）。
- **`llama_decode`**：核心前向入口；kevcpp 稳态每次请求恰好一次 151-token `llama_decode`。
- **KV / 循环状态缓存**：KV cache 与 Gated-DeltaNet 的循环状态（recurrent state）实际存储/复用机制。
- **层级化后端 / 加载机制**：`llama_model_loader` + `ggml_backend` 的加载与（便携方案下的）CPU 变体自动选择机制。

### 2.2 kev（Python，参考实现）

kev 是位于相邻仓库 `E:\van\projects\kev\kev` 的 Python 服务实现（`model.py` / `serve.py` / `api.py`）。
kevcpp **以 kev 的推理与服务行为为参考**，重新实现：

- **`DecisionModel`**（`model.py`）：`probs` / `encode` / 状态前缀缓存接口。kevcpp 的 `KevModel`
  （`kev_model.cpp/h`）实现其接口子集（probs / encode / prefix cache）。
- **服务端 `serve.py`**（含状态前缀缓存）：HTTP 服务与同 state 二次请求命中语义。
- **`systemone` 请求/响应 schema**（`api.py` 的 `render` / `option_text` / `question_keys` / `to_record` /
  `to_answers` / `confidence` 语义）：`kev_api.cpp` 复刻并暴露为 `/v1/systemone`。
- **PointerHead**：`model.py` 中的 PointerHead class（`head.bin` 权重）——`kev_head.cpp` 移植其前向数学
  （`k(h_opts) · q(h_decide) * scale`，softmax，temperature）。

一句话：**kevcpp 用 C++17 在 llama.cpp 之上把 kev 的推理/服务行为忠实重实现了一遍。**

---

## 3. 开发环境与验证（Environment & Validation）

### 3.1 开发环境

| 项 | 值 |
| --- | --- |
| 操作系统 | Windows（MSVC 2022，Visual Studio Community） |
| 编译器 | MSVC 2022，`/arch:AVX512`，C++17 |
| 构建 | CMake ≥ 3.21，**Release / x64** |
| CPU | AMD Ryzen AI MAX+ PRO 395（Zen5，32 逻辑核，AVX-512 全宽 + AVX-VNNI `vpdpbusd`） |
| 模型 | `models/kev-merged-q8.gguf`（Q8\\_0，约 **774 MiB**）+ `models/head.bin`（约 2 MiB） |

Linux 亦为可接受目标（GCC/Clang 工具链，见 §6）。

### 3.2 已通过的验证（byte-faithful）

严格字节保真协议（`curl.exe --data-binary @<file>`，每次确认服务端 `input=… output=81`）：

- **短请求** `tests/tmp_a_company.json`（`input=189`，state 38 + branch 151）。
- **长请求** `tests/tmp_long_state_request.json`（`input=263`，超长 state）。
  （上述 `tests/*.json` 请求 fixtures 见原仓库 kevcpp 的 `tests/` 目录。）
- **已知正确的概率元组**（与 kev 参考逐位一致，作为 red-line）：

| 类别 | 中标 | 产线产能 | 其他 | 厂房 | 投资 | 购地 |
| --- | --- | --- | --- | --- | --- | --- |
| 概率 | 0.0026 | **0.3974** | 0.0272 | 0.1731 | 0.396 | 0.0037 |

- 短请求 conf **0.2769**、长请求 conf **0.3691**；choice 均为 **产线产能**。
- 注：`产线产能`（0.3974）与 `投资`（0.396）是真实近并列（gap 0.0014），是各类近似优化会在近并列处翻转的选择（见 §5.3）。

- **已落地的优化与结果**（master @ `dd51ed7`）：
  - 砍掉未使用的 tied lm\_head（Rank-1，约 -100 ms）→ 稳态分叉前向 **\~440 ms → \~343 ms**。
  - ctx 默认修复为 8192 + 超长 state 清晰报错。
  - threaded/vectorized `ggml concat_f32`。
  - 综合稳态分叉前向 **\~440 ms → \~327 ms**（短请求 median 327.5 ms；长请求 median 333.8 ms）。

- **优化前基线（authoritative）**：稳态 \~440 ms（`logs/perf_authoritative_baseline.md`，原仓库 kevcpp），
  MUL*MAT ≈ 335 ms（78%，\~339 GMAC/s）、CONCAT ≈ 38 ms、GATED*DELTA\_NET ≈ 31 ms。

**项目章程（objective）**：证明 kevcpp 在稳态延迟上**不低于** Python kev，并持续压缩稳态延迟。

---

## 4. 相对 kev 的优势（Advantages over kev）

1. **原生 C++/llama.cpp**：无 Python runtime / 无 GIL；单请求开销与内存占用更低；vendored 的 llama.cpp 是成熟的
   GGUF/Q8 推理栈（权重加载、graph、CPU 内核）。
2. **状态前缀 LRU 缓存**：`kev_prefix.cpp` 的状态前缀（LRU）缓存让**重复 state 跳过 state prefill**；
   稳态下每次只 decode 分叉（branch），且 graph 拓扑复用。
3. **输出质量与参考逐位一致（硬性 red-line）**：在已测请求上，`kevcpp` 产生的概率与 Python kev 参考**逐位相同**
   （bit-identical），确保同构替换不会改变推理结果。
4. **正在进行的其他优势**（若真实/已落地则列入）：
  - **merged-LoRA 单权重前向**（`e83a9be` / `src/merge_lora_tool.cpp`）：把 LoRA fold 进 Q8\_0 基础权重，
    前向走**单份权重**，避免运行时 LoRA 叠加开销。
  - **graph 预热（graph warmup）**（`6716f35`）：`main_server` 启动时预热 graph，消除约 \~290 ms 冷启动图形构建。
  - **多问批量 decode**（P.4，`49b6cf9`）：等长多问批量 decode（位一致前提成立）。

> **关于「比 kev 快」的诚实口径**：本项目确实通过严格的字节保真测量验证了稳态延迟优于 Python kev（数据见
> 原仓库 kevcpp 的 `docs/` 日志）。但**跨仓库的绝对数值应由你在自己的硬件上重新测量**——不同机器/负载下的基准不同，本文不承诺
> 具体绝对倍率。

---

## 5. 不足与限制（Limitations，诚实说明）

### 5.1 本机 Q8\_0 CPU GEMM 是计算受限（compute-bound）

- 本盘（Zen5）上，Q8\_0 CPU GEMM 实测计算受限，速率约 **\~339 GMAC/s（约为 Zen5 int8 峰值的 \~6%）** ——这是
  当前稳态延迟的主要构成（MUL\_MAT ≈ 78%）。
- 所有「更快内核」的努力方向**均实测为输或端到端证伪**（见原仓库 kevcpp 的 `docs/F.24_…`）：
  - 512-bit `vpdpbusd` Q8×Q8（\~190–256 GMAC/s，与现有 tiled 持平）——**falsified**；
  - Q8×F16 激活对路径（FMA，微基准 \~380 GMAC/s，但集成后每 GEMM 需 F32→F16 转换吃掉收益，端到端零增益）——**falsified**；
  - F16 预反量化权重（微基准赢，集成后端到端反而慢 \~110 ms）——**falsified**。

### 5.2 VNNI/dpbusd 数值错误（Portable 分发必剪除）

- **AVX512*VNNI / AVX*VNNI 版 Q8×Q8 内核在本模型上数值错误**（短 conf **0.1222** vs 正确 **0.2769**）。
- 因此，`GGML_CPU_ALL_VARIANTS` 便携分发**必须剪除 VNNI dll**：
  `ggml-cpu-{cascadelake,icelake,alderlake}.dll`（其余保留 `x64,sse42,sandybridge,haswell,skylakex,cannonlake`）。
- 这是 Q8×Q8 dpbusd 路径在本模型上的**架构性属性**，非调参问题——无条件剪除。这与 `F.24`（原仓库 kevcpp `docs/`）的 Q8×Q8 dpbusd 失败一致。

### 5.3 近并列高度敏感（near-tie sensitivity）

- 分类器的**尖锐线性头 + 残差网络**意味着任何近似优化（SILU/SwiGLU 激活稀疏、LCP/前缀复用、低秩）都会在近并列处
  **翻转选择**（例：`产线产能 0.353 vs 投资 0.356`）。在严守位一致红线的前提下，**没有任何一个近似方向可交付**。

### 5.4 前缀/LCP 复用 与 吞吐批处理（同 state 合并）≈ 零真实收益

- **LCP/前缀复用**：实测技术上可行，但真实流量下 ROI≈0（不同 state 仅共享 \~8% 前缀）；更深入的前缀槽设计
  **破坏位一致**（batch 宽度数值依赖），均已回退（F.24 §9.4/§9.5/§9.6，原仓库 kevcpp `docs/`）。
- **同 state 批处理（吞吐）**：批宽合并在计算受限下**无权重读复用收益**，实测 req/s 几乎无提升（\~3.07 vs 2.97），
  且合并 >512 token 会触发 `n_batch` 断言——已回退（F.24 §11，原仓库 kevcpp `docs/`）。

### 5.5 主机稳定性发现

- 在开发机上，**持续请求负载下偶发堆损坏 / fast-fail**（疑为内存压力；无调试器可用以根因定位）。
- 这**阻塞了对共享库 / 便携构建的完整负载验证**（见原仓库 kevcpp `docs/portable_backend_deployment.md` §7）。

### 5.6 便携 / GPU 后端尚未最终定稿

- 可插拔 `kev::Backend` + `--backend` + `GGML_CPU_ALL_VARIANTS` 的设计\*\*已归档于分支
  `build-portable-backends` @ `71d2c64`**，但**尚未整合进稳定的共享库构建\*\*。
- 当前 CPU 硬性要求 **AVX-512**（Zen4+/Sapphire Rapids 级别），因为主构建为 `GGML_NATIVE`。
- GPU 后端仅为扩展点（`--backend gpu/cuda/vulkan` 当前返回「未构建」清晰报错）。

---

## 6. 场景 / 依赖 / 环境要求（Usage, Dependencies, Requirements）

### 6.1 运行

```bash
# 构建（CMake ≥ 3.21，MSVC 2022 Release/x64 或 GCC/Clang）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# 运行服务
./build/bin/main_server.exe \
    --model models/kev-merged-q8.gguf \
    --head  models/head.bin \
    --port 8008 \
    --threads 16
```

### 6.2 接口

- **POST `/v1/systemone`**：与 kev 相同的 JSON schema（`state` + `questions`），请求 fixtures 见原仓库 kevcpp
  `tests/*.json`（`tmp_a_company.json` 短请求、`tmp_long_state_request.json` 长请求）。
- **GET `/v1/models`**：模型/后端信息（含 `backend_name`/`device`/`version`，portable 分支）。

### 6.3 主要 CLI 选项

| 选项 | 说明 |
| --- | --- |
| `--model <gguf>` | 模型路径（必填，`kev-merged-q8.gguf`） |
| `--head <bin>` | PointerHead `.bin` 路径 |
| `--host ip` / `--port p` | 监听地址/端口（默认 `127.0.0.1:8008`） |
| `--threads N` / `-t N` | 计算线程（0/未设 = 自动探测，上限 32） |
| `--ctx N` | 上下文大小（默认 **8192**；短 state 可用 2048） |
| `--lru N` | 状态缓存 slot 数（默认 2） |
| `--prewarm <file>` | 启动预热一个 state（读原文、prefill 进 LRU，可选） |
| `--backend <name>` | 推理后端：`cpu`（默认）；`gpu`/`cuda`/`vulkan` 为「未构建」报错（portable 分支） |

> `--backend` 仅在已归档分支 `build-portable-backends` 上可用（当前主构建为 `GGML_NATIVE` 静态单 llama）。

### 6.4 依赖

- **CMake ≥ 3.21**
- **C++17 工具链**（MSVC 2022 / GCC / Clang）
- **vendored llama.cpp**（`third_party/`，无需外部下载）
- cpp-httplib、nlohmann/json（vendored，编译期使用）

### 6.5 环境要求

- 操作系统：**Windows**（已测）/ **Linux**（可接受）。
- CPU 指令集：**AVX2+ 保证正确性**；**AVX-512 以获得预期性能**（当前主构建为 `GGML_NATIVE`，故默认硬性要求 AVX-512）。
- 内存：默认 `n_ctx=8192` 下约 **\~1.4 GB 起步**（模型 \~0.7 GiB + KV \~0.34 GiB + 计算缓冲 \~0.24 GiB）。
  更大的输入可提高 `n_ctx` / `n_seq_max`。

---

## 7. 架构说明（Architecture）

### 7.1 分层

```
┌────────────────────────────── kevcpp（编排层）──────────────────────────────┐
│ (a) main_server.cpp   HTTP 层（cpp-httplib），服务 systemone schema          │
│ (b) KevModel          kev_model.cpp/h —— 编排：LLM ctx 拥有、encode、        │
│      kev_prefix.cpp   状态 LRU 前缀缓存、kev_head.cpp PointerHead 读出头、   │
│      kev_encode.cpp   每问分叉 decode/批量、compute_mutex_ 串行共享 ctx      │
│      kev_backend_factory.cpp  可插拔后端工厂（ARCHIVED）                     │
├────────────────────────────── vendored llama.cpp / GGML ───────────────────┤
│ (c) 混合 GDN+attention graph、Q8_0 内核、KV/recurrent 缓存                    │
│     编解码、qwen35.cpp 构建器（24 层混合）                                    │
│ (d) kev_backend* C-ABI 抽象（ARCHIVED，分支 build-portable-backends）         │
│     —— CPU 与未来 GPU 后端经 --backend 可 dll 加载                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 模块职责

- **(a) `main_server.cpp`**：`/v1/systemone` 的 HTTP 层（cpp-httplib），参数解析，串/并。
- **(b) `KevModel`（`kev_model.cpp/h`）**：编排核心。
  - LLM 上下文（ctx）拥有与生命周期；
  - `kev_encode.cpp` 组装请求（`encode` → `KevEnc` → `KevRows`）；
  - `kev_prefix.cpp` 状态 LRU 前缀缓存（同 state 二次请求命中）；
  - `kev_head.cpp` `PointerHead` 读出头（`d=1024 → dp=256`，`k(h_opts)·q(h_decide)*scale`）；
  - 每问分叉（branch）decode / 批量（`rows_from_slot` / `decode_batch`）；
  - `compute_mutex_` 串行化所有共享 ctx 的计算段（多流安全）。
- **(c) vendored llama.cpp / GGML 后端**：混合 GDN+attention graph、Q8\_0 内核、KV/recurrent 缓存。
- **(d) 可插拔 `kev_backend*` C-ABI 抽象**（ARCHIVED，分支 `build-portable-backends` @ `71d2c64`）：
  CPU 与未来 GPU 后端通过 `--backend` dll 加载；`KevModel` 只经 `kev::Backend*` 调用 llama/ggml 算子。

### 7.3 数据流

```
parse（/v1/systemone JSON）
   → encode（state + branch 组装为 token 序列）
   → ensure_state（LRU：命中跳过 state prefill / 未命中 prefill 到 slot）
   → rows_from_slot（branch decode，在 decide / option 位置读取 hidden）
   → PointerHead（kev_head.cpp）
   → probabilities（softmax z/T）
```

---

## 8. 模型 / HF Hub（Model & HuggingFace）

> **重要**：本项目使用的 0.8B 模型**不是现成的 Qwen-0.8B**，而是本项目**自有的、经改造/定制（modified / customized）
> 的 0.8B 变体**。

### 8.1 模型规格（已核实）

- **结构**：24 混合层 = **18 层 Gated-DeltaNet（循环）+ 6 层全注意力（第 3、7、11、15、19、23 层）**。
- `n_embd = 1024`
- FFN：`1024 → 3584`（`ffn_up/gate [1024,3584]`，`ffn_down [3584,1024]`）
- `ssm_d_inner = 2048`（`ssm_out [2048,1024]`）
- `ssm_d_state = 128`、`ssm_n_group = 16`
- `vocab = 248320`，`token_embd` 与 `output` **tied（共享）**（lm*head tied，`output_norm` 独立 F32）*\_head tied，`output_norm` 独立 F32）
- 权重导出为 **GGUF（Q8\_0，约 774 MiB）**：`models/kev-merged-q8.gguf`
- 外加独立的 **PointerHead**：`models/head.bin`（约 2 MiB）

### 8.2 HF Hub（待上传）

本项目计划将模型上传到 **Hugging Face Hub**（huggingface.co）。

- **TODO（占位，待你填写）**：仓库 ID / 链接为
  `https://huggingface.co/<your-hf-id>/<repo>`，**当前未上传，具体 ID/链接待定**。
- 上传时会包含：**GGUF 权重**（`kev-merged-q8.gguf`）、**PointerHead**（`head.bin`）以及必要的模型卡片说明。

---

## 相关文档（Docs）

> 研究日志、性能实验与部署计划文档见**原仓库 kevcpp**（`docs/` 与 `logs/` 目录），例如 `docs/F.24_…`（权威性能/实验日志）与
> `docs/portable_backend_deployment.md`（可插拔后端 / 便携构建计划，ARCHIVED）。本仓库仅托管可分发源码，不含研究日志。

---

## 许可 / 说明（License / Notes）

本项目采用 **MIT License**（见 `LICENSE`），为 kevcpp 的**当前生效许可证**。

> kevcpp 引用并复用 vendored 的 llama.cpp（`third_party/llama.cpp`）与参考实现 kev（`E:\van\projects\kev`）
> 的行为语义；llama.cpp 与 kev 各自的许可以它们各自的上游为准，MIT 许可覆盖的是本项目自身的源码组织。