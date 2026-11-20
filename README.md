# kevcpp

> kev 决策引擎的 C++17 原生实现（基于 llama.cpp）：多后端（CPU / CUDA / Vulkan）、跨平台（Windows / Linux）、
> 单次 prefill 直读选项概率的高性能 Jev 式决策服务。

---

## 致谢（Acknowledgements）

kevcpp 站在两个先行项目的肩膀上，向它们致谢：

- [**kev**](https://gitee.com/)**（Python 参考实现）**：本项目的直接前身。kev 定义了 `DecisionModel`、状态前缀缓存、
  `/v1/systemone` 服务 schema 与 PointerHead 读出头。kevcpp 以 kev 的推理与服务行为为忠实基准，
  用 C++/llama.cpp 重新实现其全部语义，并在 JevBench 上保持了更高的准确率（kev 4B 66.2% / 8B 71.4% → kevcpp 4B 76.6%）。
- [**llm2jev**](../llm2jev)**（Python 编排层 + SGLang/MLX 后端）**：同赛道的开源 Jev 复现。其"零解码 prefill-only 打分 +
  前缀缓存 + staged 批量提交"的架构验证了这条技术路线的可行性。kevcpp 与之相比把整条链路（编排 + 推理）收进单一
  C++ 进程，去除了 Python 编排与外部引擎的依赖，并在此之上做出了自己的创新（见 [§5](#5-创新与优势)）。

同时感谢 vendored 的 [llama.cpp](third_party/llama.cpp)（GGUF/GGML 推理栈）与 JevBench 基准。

---

## 1. 性能比较（Performance & Accuracy）

JevBench 231 道公开题（easy/hard/original），letter 决策模式。

### 1.1 准确率（Accuracy）

| 系统 | 模型 | 准确率 |
| --- | --- | ---: |
| Jev 1.13.0（官方，TypeSafe AI） | - | 85.1% |
| **kevcpp** | Qwen3.5-4B Q6\\_K | **76.6%**（177/231） |
| LLM2Jev | Qwen3.5-4B | 76.2% |
| **kevcpp** | Qwen3.5-2B Q8\\_0 | 65.8%（151/231） |
| **kevcpp** | Gemma E2B IQ4\\_XS | 63.2%（146/231） |
| jev-local | Qwen3.5-9B | 74.9% |
| kev（research preview） | 8B | 71.4% |
| kev（research preview） | 4B | 66.2% |
| Open-Jev | 2B | 64.5% |

（除 kevcpp 外的数据引自 LLM2Jev 的 [JevBench 报告](../llm2jev/docs/en/jevbench.md)。）

- **kevcpp 4B 准确率与 LLM2Jev 持平**（76.6% vs 76.2%），显著高于同源的 kev 4B（66.2%）与 kev 8B（71.4%）。
- **跨平台/后端一致**：Windows（Vulkan/CPU）与 Linux（CPU/CUDA）准确率差异 ≤2 题（±1% 内，数值噪声）；
  CUDA 后端完全确定（4B 三次复测逐题 0 差异）。

### 1.2 延迟（Latency）

Qwen3.5-4B，231 题（kevcpp 为 Linux / RTX 5090 CUDA 实测）：

| 系统 | P50 (s) | P95 (s) |
| --- | ---: | ---: |
| Jev 1.13.0（官方） | 0.652 | 0.722 |
| **kevcpp（CUDA，RTX 5090）** | **0.043** | **0.296** |
| LLM2Jev（自报） | 0.048 | 0.346 |

kevcpp 各后端全量数据（correct/231 | P50 | P95，单位秒）：

| 配置 | 0.8B Q8\\_0 | 2B Q8\\_0 | 4B Q6\\_K |
| --- | --- | --- | --- |
| Win / Vulkan（Strix Halo iGPU） | 122 \| 0.089 \| 0.566 | 150 \| 0.153 \| 1.174 | 177 \| 0.477 \| 3.535 |
| Win / CPU（KAT auto） | 121 \| 0.901 \| 8.021 | - | - |
| Linux / CPU（9950X，32 线程） | 119 \| 0.570 \| 5.863 | 150 \| 1.111 \| 12.241 | 177 \| 1.318 \| 11.485 |
| Linux / CUDA（RTX 5090） | 121 \| 0.024 \| 0.157 | 151 \| 0.028 \| 0.183 | 177 \| 0.043 \| 0.296 |

要点：

- kevcpp CUDA 与 LLM2Jev（同为 GPU prefill-only 路线）延迟同量级，**无需 Python/外部推理引擎**即可达到。
- 直接使用GGUF量化模型，准确率不降的情况下更快。
- RTX 5090 CUDA 比 9950X 32 线程 CPU 快 20–30×；CPU 线程扩展良好（4 线程慢 \~2.4×）。
- 诚实口径：绝对数值依赖硬件/驱动/负载，请在自己的环境重新测量。

### 1.3 分题型表现（Per-type Breakdown）

JevBench 231 题按题型分为 **choice**（多选一，139 题）、**noul**（无选项自由决策，74 题）、**score**（数值评分，18 题）。
各题型在不同模型下的准确率与延迟（准确率 = correct/total；延迟为该题型 P50，单位秒）：

**Linux / CUDA（RTX 5090）**

| 题型 | 题数 | 0.8B Q8\\_0 | 2B Q8\\_0 | 4B Q6\\_K |
| --- | ---: | --- | --- | --- |
| choice | 139 | 52.5%（73）\| 0.016 | 66.2%（92）\| 0.018 | 77.7%（108）\| 0.027 |
| noul | 74 | 48.6%（36）\| 0.027 | 63.5%（47）\| 0.031 | 74.3%（55）\| 0.047 |
| score | 18 | 66.7%（12）\| 0.034 | 66.7%（12）\| 0.040 | 77.8%（14）\| 0.060 |

**Win / Vulkan（Strix Halo iGPU）**

| 题型 | 0.8B Q8\\_0 | 2B Q8\\_0 | 4B Q6\\_K |
| --- | --- | --- | --- |
| choice | 53.2%（74）\| 0.059 | 65.5%（91）\| 0.095 | 78.4%（109）\| 0.263 |
| noul | 48.6%（36）\| 0.098 | 63.5%（47）\| 0.167 | 73.0%（54）\| 0.565 |
| score | 66.7%（12）\| 0.136 | 66.7%（12）\| 0.233 | 77.8%（14）\| 0.708 |

**Linux / CPU（9950X，32 线程）**

| 题型 | 0.8B Q8\\_0 | 2B Q8\\_0 | 4B Q6\\_K |
| --- | --- | --- | --- |
| choice | 51.1%（71）\| 0.414 | 65.5%（91）\| 0.901 | 79.1%（110）\| 0.896 |
| noul | 48.6%（36）\| 0.602 | 64.9%（48）\| 1.264 | 71.6%（53）\| 1.324 |
| score | 66.7%（12）\| 0.589 | 61.1%（11）\| 1.118 | 77.8%（14）\| 1.450 |

要点：

- **模型规模是准确率的主导因素**：0.8B→4B 各题型准确率普遍 +20–28pt；题型间 choice/noul 接近，score 样本小（18 题）波动大。
- **各题型跨后端/平台一致**（±1\~2 题数值噪声），延迟与全局 P50 同比例缩放。
- choice 题延迟最低，noul/score 稍高（观察值；与其请求 token 序列更长相符，未单独归因验证）。

### 1.4 跨模型族验证：Gemma E2B（Win / Vulkan）

用 **Gemma E2B it（IQ4\_XS，2.8 GB）** 验证 kevcpp 决策管线对非 Qwen 架构的可用性
（Win / Vulkan，Strix Halo iGPU，v0.7.1）：

| 指标 | Vulkan | 同机 CPU 对照 |
| --- | ---: | ---: |
| 正确 | **146/231（63.2%）** | 145/231（62.8%） |
| P50 | **0.204s** | 1.003s |
| P95 | **2.27s** | 11.91s |

分题型（family）正确数（GPU；括号内为 CPU 对照差异题数）：

| family | 正确/题数 | family | 正确/题数 |
| --- | --- | --- | --- |
| fact | 12/12 | ordinal | 11/12 |
| tool\\_selection | 12/12 | routing | 8/12 |
| routing\\_hard | 5/5 | multi\\_hop | 8/18 |
| intent | 22/24 | long\\_policy | 8/19 |
| extraction | 20/24 | policy | 6/12 |
| adequacy | 9/12 | trap | 6/8 |
| judge\\_hard | 9/17 | adversarial | 4/6 |
| temporal\\_numeric | 1/15 | probability | 2/10 |
| ambiguous | 2/7 | tradeoff | 1/6 |

要点：

- **管线对非 Qwen 架构同样可用**：Gemma E2B 单次决策 63.2%，接近同量级 Qwen 2B（65.8%），
  无需改动任何模型相关代码。
- **GPU/CPU 准确率一致（±1 题）**，延迟差距 \~5×，与 §1.2 各后端规律相符。
- 优势题型（fact/tool_selection/routing_hard 满分）与短板题型（temporal_numeric/probability）
  与 Qwen 系分布不同，属模型能力差异，非管线问题。

---

## 2. 快速使用（Quick Start）

### 2.1 构建

```bash
# 依赖：CMake >= 3.21 + C++17 工具链（MSVC 2022 / GCC 14 / Clang）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
# 产物: build/bin/Release/kevcpp(.exe)
```

GPU 后端在独立构建目录编译（`n_gpu_layers=-1` 整模型 offload）：

```bash
# CUDA（Linux，RTX 5090 / sm_120，CUDA 12.8）
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DKEV_BUILD_BACKEND_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-cuda --config Release -j
# 产物: build-cuda/kev-backend-cuda.so（dlopen 查找 libkev-backend-cuda.so，需符号链接或重命名）
```

```powershell
# Vulkan（Windows，AMD iGPU，无需 ROCm/HIP）
cmake -S . -B build\vulkan-exp -G "Visual Studio 17 2022" -A x64 `
  -DKEV_BUILD_BACKEND_VULKAN=ON -DGGML_NATIVE=ON
cmake --build build\vulkan-exp --config Release
```

### 2.2 运行

```bash
# CPU 后端（默认）
./build/bin/kevcpp --model Qwen3.5-4B-Q6_K.gguf --port 8008

# CUDA 后端（把 kev-backend-cuda.so 置于可搜索路径）
export LD_LIBRARY_PATH=./build-cuda:$LD_LIBRARY_PATH
./build/bin/kevcpp --model Qwen3.5-4B-Q6_K.gguf --port 8008 --backend cuda

# Vulkan 后端（Windows，kev-backend-vulkan.dll 与 exe 同目录）
build\bin\Release\kevcpp.exe --model Qwen3.5-4B-Q6_K.gguf --port 8008 --backend vulkan
```

接口：

- **POST `/v1/systemone`**：kev 相同的 JSON schema（`state` + `questions`），返回 `answers.decision`。
- **GET `/v1/models`**：模型/后端信息（`backend_name`/`device`/`version`）。

### 2.3 可用参数

| 选项 | 说明 |
| --- | --- |
| `--model <gguf>` | 模型路径（必填） |
| `--host ip` / `--port p` | 监听地址/端口（默认 `127.0.0.1:8008`） |
| `--backend <name>` | `cpu`（默认）/ `cuda` / `vulkan`（需对应后端插件） |
| `--threads N` / `-t N` | 计算线程（0/未设 = 自动探测，上限 32） |
| `--ctx N` | 上下文大小（默认 8192；长输入可调大） |
| `--max-concurrency N` | 最大并发（默认 8） |
| `--prefix-capacity N` | 全局前缀 KV 缓存槽数 |
| `--n-batch N` / `--n-ubatch N` | 批处理大小（默认 512 / 256） |
| `--gpu-layers N` | GPU offload 层数（GPU 后端默认 -1 全部） |
| `--decide-config <json>` / `--decide-token <t>` | 决策配置 / 决策 token |
| `--moe-mode <m>` | MoE 专家模式 |
| `--lora <gguf>` / `--lora-scale f` / `--temperature f` | LoRA 叠加 / 采样温度 |
| `--opt-end <tok>` | 可选结束 token |
| `--kev-probe-cpu-variant <dll>` | 内部：KAT 探测子进程模式（勿手动使用） |
| `--quiet` | 静默日志 |

环境变量：

| 变量 | 说明 |
| --- | --- |
| `KEV_CPU_VARIANT` | CPU 变体选择：`auto`（默认，KAT 自动门控）/ `off` / `<tag>`（强制指定，仍经 KAT） |
| `KEV_BINARY_SCORING` / `KEV_BINARY_LETTER` | letter 二分打分模式（基准测试使用） |

---

## 3. 原理简介（How It Works）

```
parse（/v1/systemone JSON）
   → encode（state + question 渲染为 token 序列）
   → 前缀 KV 缓存（全局前缀命中则跳过 state prefill）
   → 一次 prefill 前向
   → selected LM head：只对决策位置读取 A/B/C/D… 有限几个 letter token 的 logits
   → softmax → 概率 → answers.decision
```

- **单次前向决策**：多选题把全部候选渲染进同一 prompt，**一次 prefill** 同时拿到所有选项 letter 的 logits——
  不需要像逐候选打分那样跑 N 次前向。
- **selected LM head 剪枝**：决策只依赖词表中极少几个 letter token 的概率，只计算这些行的 lm\_head 投影
  （`letter scoring via cached A..Z rows`，`09718b0`），跳过全词表（151k×hidden）投影。
- **全局前缀 KV 缓存**：跨请求复用共享 state 前缀的 KV cache（`--prefix-capacity`），命中时只增量 prefill。
- **混合架构模型**：Qwen3.5 系（Gated-DeltaNet 循环层 + 全注意力层混合，`src/models/qwen35.cpp`），
  稳态每请求恰好一次 `llama_decode`。
- **可插拔后端 C-ABI**（`kev_backend.h`）：宿主 `kevcpp` 不链接 llama（llama-free），经版本化 ABI 加载
  CPU/CUDA/Vulkan 后端插件（single-llama rule）。
- **运行时 CPU 变体选择 + KAT 门控**（v0.7.0，`8f27b4e`）：启动时枚举全部 `ggml-cpu-*` 变体，按 cpuid 分数降序
  在子进程跑 GEMM 数值自检（KAT），与 x64 标量参考比对通过者才启用——数值错误的 VNNI 变体被自动拒绝。

---

## 4. 与 kev / llm2jev 的对比定位

| 维度 | kev（Python） | llm2jev（Python + SGLang/MLX） | **kevcpp** |
| --- | --- | --- | --- |
| 语言/形态 | Python 服务 | Python 编排 + 外部推理引擎 | **单一 C++ 进程（编排+推理）** |
| 推理内核 | 自带（Python） | SGLang（CUDA）/ MLX（Metal） | vendored llama.cpp（CPU/CUDA/Vulkan） |
| 多候选打分 | - | 每候选独立 prompt，N 次 prefill（RadixCache 缓解） | **单 prompt 一次 prefill，一次读全部选项 logits** |
| lm\\_head | - | 最后位置全词表投影后 gather | **selected rows：只算 letter 行** |
| 依赖 | Python runtime | Python + SGLang/flashinfer 全家桶 | **零外部运行时依赖** |
| 部署 | - | 需装引擎+GPU 栈 | 单 exe + 后端 so/dll，复制即用 |
| 平台 | - | Linux(NV GPU) / Apple Silicon | **Windows + Linux；CPU/CUDA/Vulkan** |
| JevBench 4B | 66.2%（4B）/ 71.4%（8B） | 76.2% | **76.6%** |

---

## 5. 创新与优势（Innovations & Advantages）

相对 **kev**：

1. **C++ 原生重实现，无 Python runtime/GIL**：单请求开销与内存占用大幅降低；模块化（HTTP 层 / 编排层 / 后端层
   C-ABI 边界）同时保持与 kev 参考的行为/输出一致性。
2. **准确率反超**：kev 4B 66.2% → kevcpp 同模型系 4B **76.6%**（+10.4pt），说明工程化前向（量化/缓存/letter
   打分路径）没有损失、反而稳定复现了模型能力。
3. **状态前缀 LRU + 全局前缀 KV 缓存**：重复 state 跳过 prefill，稳态每请求只一次 decode。

相对 **llm2jev**：

1. **单进程零依赖**：llm2jev 需 Python + SGLang（及其 flashinfer/CUDA graph 全家桶）或 MLX；kevcpp 是一个 exe +
   后端插件，复制即部署，嵌入成本近零。
2. **单次前向多选项**：llm2jev 的 yes/no 逐候选打分对 N 个候选要 N 个 prompt（靠 RadixCache 缓解为增量 prefill）；
   kevcpp 把候选合并进单 prompt，**一次 prefill 同时取得全部选项概率**，算法上更省。
3. **selected LM head 剪枝**：只计算决策所需 letter 行的 lm\_head 投影，跳过 151k 全词表投影与 gather。
4. **运行时 ISA 变体 KAT 门控**（独有）：多 ISA 分发不再依赖手工剪枝——启动时数值自检自动拒绝错误内核
   （VNNI bug 在 Win/Linux 均复现并均被 KAT 拦截），可安全地把全部 CPU 变体一起分发。
5. **跨平台三后端**：同一 ABI 下 CPU/CUDA/Vulkan 可插拔，覆盖 Windows（AMD iGPU Vulkan）与 Linux
   （9950X CPU / RTX CUDA），llm2jev 不支持 Windows/AMD iGPU 路径。
6. **确定性**：CUDA 后端多次复测逐题 0 差异，利于回归测试与结果审计。

---

## 版本与变更（Versions & Changelog）

- 本仓库为**发版仓库**：只包含已确定的发布版本内容；研发过程、实验记录与调研材料保留在 kevcpp 研发仓库
  （`E:\van\projects\kevcpp`），确认发版后才同步到此处。
- 当前版本：`v0.7.0-cpu-runtime-kat`（运行时 CPU 变体选择 + KAT 门控）。
- 历史版本：`v0.6.0-cuda-backend`（CUDA 后端）；`v0.5.1-letter-selected-rows`（letter + selected-head 基线）；
  `v0.5.0-letter-logit`；`v0.4.0-binary-prefix-reuse`；`v0.3.0-logist`。
- `CHANGELOG.md` — 版本变化记录。

---

## 许可 / 说明（License / Notes）

（项目许可证待定。kevcpp 引用的 vendored llama.cpp 许可以其上游为准；kev 与 llm2jev 为独立项目，
本文仅作学术/工程对比引用。）