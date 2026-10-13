# 调研报告：MoE 模型「内存足、显存不足」下按需把专家加载到显存 —— 论文与开源实现

> 调研对象：kevcpp（C++17 / llama.cpp 系）运行 MoE 模型时，在**系统内存(DRAM)充足、GPU 显存(VRAM)不足**的场景下，
> 把权重放内存、按需把被激活的专家加载到显存执行，能否做到、以及如何把性能影响降到最低。
> 资料来自：arXiv 论文精读 + llama.cpp / vLLM / SGLang / KTransformers / TritonLLM 等开源实现的实际抓取验证。
> 调研日期：本报告基于当前可抓取的公开资料整理。

---

## 开头三句话结论（TL;DR）

1. **逐 token 实时把专家搬到 VRAM 在工程上被证明不可行**（vLLM RFC #57794 明确定性）——MoE 路由是 data-dependent 的，
   要到当前层算完 router logits 才知道要哪些专家，**没有可用的 lookahead 预取窗口**，等知道时权重已在关键路径上，
   同步 H2D 传输会直接吃掉解码预算。这是整个方向的硬边界，必须先认清。
2. 社区因此落在**三类可落地的折中方案**：**① 冷尾专家驻留 host + UVA 零拷贝直读**（不搬回 GPU）；**② GPU 端 LRU/LFRU 专家缓存**
   （hot 专家常驻、cold 专家 miss 才 H2D 上传，llama.cpp `--moe-cache-mib` / vLLM `--moe-expert-cache-size`）；
   **③ 冷专家留在 CPU 用 AMX/AVX 量化核执行**（KTransformers，根本不搬）。
3. **对 kevcpp 特别有利**：kevcpp 注于 llamacpp、且宿主是 AMD Ryzen AI MAX+（**统一内存**，CPU↔iGPU 互连带宽高，
   天然满足 UVA/zero-copy 的"快速互连"前提）。kevcpp 最新 vendored llama.cpp **已经内置**了 MoE 全套支撑
   （`build_moe_ffn`、几十种 MoE 架构、`tensor_buft_overrides`），**静态"专家留 CPU、其余留 GPU"** 无需改 llama.cpp 即可做到；
   再往上加"GPU 端 miss-only 专家缓存"即可逼近"仅把最热的专家放显存"的目标。

> ⚠️ 重要前提：**kevcpp 当前模型（0.8B，24 层 Gated-DeltaNet + 6 层 attention）不是 MoE 模型**。本报告讨论的是
> **kevcpp 未来引入 MoE 模型**时的可行性预案。（`grep` 全 src 无 moe/mixture 相关代码；docs 亦无 MoE 规划。）

---

## 一、问题定义与核心矛盾

- **场景**：DRAM 足够、VRAM 不足。想把超大 MoE 模型跑起来：非专家权重（attention/shared/embedding）+ 少量"热专家"
  常驻 VRAM，其余所有专家放 DRAM，被 gate 选中时按需拷入 VRAM 执行。
- **硬约束**：MoE 每 token 只需 top-k 个专家，但**专家选择依赖输入（data-dependent）**，只有第 L 层 router 算完才知道
  要第 L 层的哪些专家 → 若在关键时刻才发起 H2D 拷贝，传输必然落在**关键路径（critical path）**上。
- **量化的性能代价**（HOBBIT 实测）：Mixtral-8x7B 从 CPU 经 PCIe 4.0（32 GB/s）加载**一整层约 80 ms**，而 RTX 4090 算
  同一层仅 ~3 ms；专家加载占解码耗时在 **RTX 4090 上 85.5%、Jetson Orin 上 94.5%**。
- vLLM 的代价模型：1×H100、48 层 ×512 专家/层、每 token 取 10 专家，**单个冷专家 one-target fetch ≈ 63.2 µs =
  单层时间预算的 ~14%**；若每层命中 1 个冷专家，decode 直接掉到 ~40 tok/s。

> 一句话：**"最低性能影响"的唯一靶点是——用「预测 + 预取 + overlap」把 PCIe/统一内存的 DMA 传输藏进上一批专家的计算/attention 里。**
> 只有做到传输与计算重叠，offload 才不至于成为逐 token 的串行瓶颈。

---

## 二、学术论文调研（arXiv / 会议）

> ⚠️ 调研中发现：若干流传的 arXiv id 是错的（`2412.02550` 标称 Flash-MoE 实为粒状材料论文、`2411.17070` 标称 MoE Jetpack
> 实为光学调制论文）。本表均为**重新定位并精读正确论文**后的结果。

### 2.1 精确专家 offload / 细粒度调度（核心参考）

| 论文 | ID / 年 | 核心方法与思想 | 关键指标 | 带宽/加载缓解 |
|---|---|---|---|---|
| **MoE-Lightning** | 2411.11217 / 24 | 高吞吐 + 显存受限。`CGOPipe`：CPU-GPU-I/O **三级流水**用 **paged weights**（GPU 常驻页 + CPU 按需拷页）实现 compute 与 I/O overlap；`HRM`（Hierarchical Roofline）选策略 | Mixtral-8x7B 单 T4(16G) 比最优 offload **吞吐高 10.3x**；用 2-3x 更少 CPU 内存即可达吞吐上界 | 分页 + 三级流水 overlap 藏传输 |
| **HOBBIT** | 2411.01433 / 24 | **最贴近 llama.cpp/kevcpp**。混合精度专家 offload：cache miss 时用 **int4 低精度专家替换低重要性专家**，2-4x 降加载量；三招：① token 级动态加载（用 gate 输出 ‖G(x)‖≈专家重要性，相关 0.99）；② **层间自适应预取**（残差结构使相邻层 gate 输入高度相似，**下一层 top-1 专家预测准 96%**，下 2/3 层仍 ~90%，用 Stacking 一次算所有层 gate）；③ 序列级 `LHU` 缓存（以"高精度 expert miss 代价"为替换目标，比 LFU 少 ~15% 惩罚） | 解码比 SOTA offload 最高 **9.93x**；**jem 在 llama.cpp 上加 ~8000 行 C/C++** | 混合精度 + 层间预取(96%) + LHU 缓存 |
| **FineMoE** | 2502.05370 / 25 | 从 MoE 提**细粒度专家选择模式** + 从 prompt 提**语义提示**，共同引导 prefetch/cache/offload | 推理延迟 **-47%**、专家命中率 +39%（HF Transformers 6 GPU） | 模式/语义引导预取 |

### 2.2 PowerLaw 热冷分离（激活幂律，思想基石）

| 论文 | ID / 年 | 核心方法 | 指标 | 说明 |
|---|---|---|---|---|
| **PowerInfer** | 2312.12456 / SOSP'24 | 神经元激活呈**幂律**：热神经元恒定激活→**常驻 GPU**；冷神经元按输入变→**在 CPU 算**；自适应预测器 + neuron-aware 稀疏算子 | 单 RTX 4090 比 llama.cpp **11.69x**（OPT-175B）；OPT-30B 达 A100 的 82% | dense+sparse 混合，非专家 offload，但"热冷分离常驻/外置"是所有后续 offload 的基石 |
| **PowerInfer-2** | 2406.06282 / 24 | 把矩阵拆成**神经元簇(neuron cluster)**为基本单元；解码 CPU+NPU 混合。存储侧：in-memory **neuron cache**（固定区/热区/冷区各自 LRU）；**cluster 级流水**（一个簇算完立刻算下一个已缓存簇，把随机 I/O 藏进多矩阵计算）；**colum 对齐捆绑**（Gate,Up,Down 同神经元捆绑成 7.5-24KB 一块读）| 比 llama.cpp **24.6x(最高27.8x)**；首个手机跑通 **47B @11.68 tok/s**；7B 省 40% 内存不掉精度 | 移动端方法细节最全 |

### 2.3 权重访存 / DRAM offload（通用权重按需，不限专家）

| 论文 | ID / 年 | 核心方法 | 指标 | 说明 |
|---|---|---|---|---|
| **LLM in a flash** | 2312.11514 / ACL'24 (Apple) | 模型存 flash、按需拉进 DRAM。① **低秩激活预测器**（预测哪个 ReLU 中间神经元被激活，只加载那些行，FFN 稀疏 ~97%）；② **Windowing**（只缓存最近 k=5 token 激活过的行，增量加载）；③ **Row-Column Bundling**（up 列+down 行按同一神经元捆绑成 2x 大块读，摊薄 read latency）；④ DRAM 预分配 + 指针管理避免 realloc；⑤ 32 线程并行读 | 模型做到 DRAM 2x 大小；比 naive 加载快 **4-5x**、GPU(Metal/CUDA) 快 **20-25x**；OPT-6.7B flash→DRAM 传输 13.4GB(naive)→0.2GB | sparsity 是 FFN 神经元粒度（dense 模型），但其**预测+窗口+捆绑+并行读**整套反带宽技术对专家 offload 直接可移植 |

### 2.4 投机式专家选择 / 预取（把 data-dependent 传输藏进计算 —— 最前沿）

| 论文 | ID / 年 | 方法 | 指标 |
|---|---|---|---|
| **Speculating Experts** | 2603.19289 / 26 | 用内部表示**预测未来层专家**使传输与计算 overlap，支持直接执行投机专家 | TPOT **-14%**（跨多 MoE 架构，开源 yalis/offload_prefetch）|
| **SP-MoE** | 2510.10302 / 25 | 首个 **SD 感知(投机解码感知)专家 offload**；cutoff-layer 策略 + 异步预取线程 + 批量 I/O | TPOT 比 SOTA **1.07-3.5x** |
| **MoE-SpeQ** | 2511.14102 / 25 | 片上小 draft 模型预测未来 token 专家序列→预取；Amortization Roofline 动态调投机深度 | Phi-MoE 比 SOTA offload **最多 2.34x** |
| **SpecPrefetch** | 2607.24787 / 26 | 共享轻量 adapter **只预测下一层候选专家用于异步传输，最终执行仍由原生 router 决定**（预测错不改输出语义，只影响传输效率） | Qwen3-VL-30B-A3B，Snapdragon 8 Elite 解码吞吐 +20% |
| **DraftExpert** | 2607.24434 / 26 | 每层训 1 个极轻量 **draft expert**（resident），置信度-规模截断 + target 专家预取，token 仍由 target 精确验证 | DeepSeek-V2-Lite / Moonlight-16B 解码吞吐 **1.45x**，acceptance 84-87% |
| **AcceptMoE** | 2608.02989 / 26 | 投机验证端**按 cache 驻留而非预测自然路由**限定合格专家，减少 host→device 流量 | 全显存 1.29x，offload 2.06x；host→device 流量 **-73.6~77.1%** |
| **SpecMD (least-stale)** | 2602.03921 / 26 | 发现 MoE 专家访问**不满足 LRU/LFU 时间局部性**，提 least-stale 淘汰 | OLMoE 上 0.6GB/5% 显存即 >88% 命中，**TTFT -34.7%** |

### 2.5 其它相关（一网打尽）

- **HybriMoE** 2504.05897（DAC'25）：动态层内 CPU-GPU 调度 + 层间预取 + score 缓存；prefill 1.33x / decode 1.70x（基于 kTransformers）。
- **MoE offloading 缓存/预取深度分析** 2511.05814：LFU 优于 LRU + 投机预取潜力。
- **MoE local routing consistency** 2505.16056：提 SRP/SCH 指标衡量路由局部一致性，测 20 个 MoE，**cache 容量 ≈ 活跃专家 2x** 为最优；**并非所有模型都适合 offload**。
- **DynaExq** 2511.15015：运行时感知混合精度驻留（hot 高精度常驻、冷低精度兜底），bs=32 相对 offload **2.73x**。
- **MoLE** 2503.15798（ICML'25）：训练后把专家重参数化为 **lookup table** 放存储，推理按输入 id 查表，通信近零（彻底绕开重量传输）。
- **ReMoE** 2605.27081：router 微调让路由偏近期专家；llama.cpp 在 Jetson Orin NX 上 TPOT -43.6~49.8%（decode 1.77-1.99x）。
- **CoX-MoE** 2605.17889（DAC'26）：AMX CPU-GPU 协同 + coalesced 执行（基于 kTransformers / HybriMoE 后续）。
- **EStream** 2609.06551、**MoE-CORE** 2610.01950、**RotaryQuant** 2608.08081（28-120B MoE 塞进 16-32GB 的量化+offload+KV 压缩）。

---

## 三、开源工程实现调研（可直接照搬 / 借鉴）

### 3.0 决定性结论：逐 token 实时 swap 不可行

vLLM **RFC #57794 明确验证并否掉了 per-token swap-on-demand**：
- MoE 路由 data-dependent，第 L 层专家集合只有算出第 L 层 router logits 才知道，router 又需要第 L-1 层输出 →
  **无可用的 lookahead 预取窗口**。
- 1×H100，48 层 ×512 专家/层、每 token 10 专家，单冷专家 one-target fetch 63.2µs = 单层预算 14%；每层 1 个冷专家即掉到 ~40 tok/s。

**因此社区的三类真实落地方案**（详见下文），kevcpp 应据此选型：
1. **冷尾驻留 host + UVA zero-copy 直读**（vLLM UVA 后端）
2. **GPU 端 LRU/LFRU 专家缓存**（llama.cpp `--moe-cache-mib` / vLLM `--moe-expert-cache-size`）
3. **CPU 侧用 AMX/AVX 量化核执行冷专家**（KTransformers）

### 3.1 llama.cpp —— 与 kevcpp 血缘最近的参考

> 注意：kevcpp 是 **llama.cpp 系**（vendored 在 `third_party/llama.cpp`），所以下面这些 PR 是**可以直接照搬的最小改造路径**。

| 特性 | PR / 状态 | 机制 | 性能 |
|---|---|---|---|
| **`--n-cpu-moe N` / `--cpu-moe`**（把前 N 层/全部 MoE 权重留 CPU）| **PR #15077**（slaren，合并）| 通过**张量级 tensor override** 把整层专家留 CPU，CPU 上执行 | 粗粒度开关，非 GPU 缓存 |
| **`--moe-cache-mib N`**（GPU 端 LRU 专家缓存）| **PR #29887**（am17an，合并 2026-10-07）| 移植自 qvak-fabric；host 专家情形下在 GPU 用 **LRU 专家缓存**执行，**只有 miss 才 H2D 上传(miss-only)**；**小 batch(≤32 token)走缓存、大 batch 走常规 offload**；不同布局层独立 bank | Qwen3.8-Flash-Next Q4_0 RTX5090：30.8→**67.8 tok/s(2.2x)**，hit 89%；10% 专家总量是常见起点 |
| **`--moe-cache-layers N\|auto`**（汇总整合，未合并）| **PR #30059**（wackMall）| 整合 #29887 + #15077 + #15346(staging 只上传用到的专家) + #20416(fused gate_up) + #26003(page 对齐 MADV_DONTNEED 释放) + #21609/21614(频度加权 LFRU + 依空闲显存定容) + #27861(逐 token 独立 id / eviction 前同步) + #28414(mmap move) | 可当作**实现要素清单** |

**与 `n_gpu_layers` / mmap 的关系（重要坑）**
- llama.cpp **`--split-mode tensor`（张量并行）对 MoE 架构未实现**（Grok/OLMoE/DeepSeek2/GLM-DSA/Nemotron-H/Minimax 等全部 `not implemented`）→ MoE 分片只能走 **layer 模式**，专家层留在 host 与 CPU 一起算。
- 即便 `--n-gpu-layers all`，个别 expert op 若要求 FP32 src 精度被 CUDA `supports_op` 判不支持，会被调度器**静默降到 CPU**（issue #30224）。kevcpp 移植时要留意"全层 offload 后个别 expert op 回落 CPU"。

### 3.2 vLLM —— 生产级权重 offload（UVA 零拷贝 / 分层预取）+ 正在推进的专家粒度

- **已合并 `OffloadConfig`**（`vllm/config/offload.py`）：
  - `UVAOffloadConfig`：`cpu_offload_gb` + `cpu_offload_params`（按**参数名段匹配**挑参数 offload，官方示例 `"mlp.experts.w2_weight"`）。原理 = 从 CPU **pinned** 内存 **UVA zero-copy** 直读，**不搬回 GPU**，要求 fast CPU-GPU interconnect。
  - `PrefetchOffloadConfig`：`offload_group_size` / `offload_num_in_group` / `offload_prefetch_step`（**异步 H2D 预取步数**）/ `offload_params`。用 async prefetch 藏传输延迟。
  - `OffloadBackend` = auto / uva / prefetch。
- **未合并 PR：专家级 LRU/LFRU 缓存**：PR #37190（e1n00r）→ 移植 #50562：`--moe-expert-cache-size`，`CachedWeightProvider`——专家放 CPU pinned，每层一个固定大小 GPU 缓存，**LFRU(频度加权 LRU) 换出**，拿 `expert_map`（全局专家 id→slot，缺席=-1）喂 kernel。**限制：需 `--enforce-eager`（CUDA graph 不兼容）、同步 H2D、仅 BF16+FP8 per-tensor、EP>1 不支持**。RFC #38256。
- **正在推进：专家粒度 residency + 冷尾降级**：RFC #57794 + PR #57943（`uva_bank.py`）：按 EPLB 每专家负载统计，把每层**最冷 n_cold 个专家降级到 host**，device/host 双 bank 各跑一次前向后求和。离线后模型内存 68.18→58.52 GiB，KV cache 144,725→522,878 token；concurrency 1 decode -19%、concurrency 4 端到端更好。
- 其它：PR #49321（AMD RDNA3 WIP）`VLLM_MOE_EXPERT_CACHE_SIZE`，Qwen3.5-122B-A10B 2×24GB 上 35-40 tok/s，实测**跨层专家预测率仅 ~3.4%** → 进一步证明"即用即取"优于"猜预取"。

### 3.3 SGLang —— 通用 offload + 深度绑定 KTransformers 的 CPU-expert 双通道

- 通用层级 expert offload：`cpu_offload_gb` / `offload_group_size` / `offload_num_in_group` / `offload_prefetch_step` / `offload_mode="cpu"`（与 vLLM Prefetch 同源同构）。
- **KTransformers 集成**：`kt_weight_path` / `kt_method="AMXINT4"` / `kt_cpuinfer` / `kt_threadpool_count`（按 NUMA 节点一线程池）/ `kt_num_gpu_experts`（GPU 专家数）/ `kt_max_deferred_experts_per_token` / `kt_gpu_prefill_token_threshold`。即 **hot 专家在 GPU、cold 专家 defer 到 CPU 用 AMX 执行**。

### 3.4 专门项目：KTransformers —— 此方向的标杆

- 定位：CPU/GPU **异构**跑超大 MoE，官方"**hot experts on GPU, cold experts on CPU**"。
- CPU 侧用 **Intel AMX / AVX512 / AVX2 的 INT4/INT8 量化核**；MoE 优化含 NUMA-aware 内存管理；支持 GGUF（mm 布局）、FP8、GPU-CPU-Disk 三级前缀缓存；原生接入 SGLang。
- 论文：*KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models*（2025）。
- 案例：DeepSeek-R1-0528 FP8 → 8×L20 GPU + Xeon 6454S，**227.85 tok/s**（8 路并发）。
- **与 kevcpp 关系**：它本质是 llama.cpp 衍生（用 GGUF/llama.cpp 张量布局），CPU 专家用量化核执行而非搬回 GPU → 是 kevcpp 若想做成"hybrid"的最完整产品化参照。

### 3.5 TensorRT-LLM / DeepSpeed / accelerate —— 现状说明

- **TensorRT-LLM**：**没有通用的"专家放 host RAM、按需搬 GPU"路径**；MoE 走多卡 **expert-parallel(EP)**，把专家分到多 GPU，而非放系统内存。唯一相近是 **PLE（Partial-Linear Embedding）host offload**，仅适用于 Qwen3.8-Flash-Next 特定模型的 embedding/索引表，不是 expert。
- **DeepSpeed-MoE / ZeRO++**：expert CPU/disk offload 是**训练侧**特性（ZeRO-Infinity/ZeRO++ 动量/通信 offload），服务推理不作为一等能力；官方文档未确认到推理 expert-offload 参数 → **不建议作为 kevcpp 生产参考**。
- **HuggingFace accelerate / transformers `device_map`**：按层/模块/整参数张量粒度 offload 到 CPU/disk，hooks 在 CPU 上执行被 offload 的模块；**对"单根 [专家×维度] 张量里挑几行"的 expert 粒度无内建手段**。

---

## 四、kevcpp 应用可行性评估（重点）

### 4.1 kevcpp 现状与相关事实核对

| 项 | 事实 | 对可行性的影响 |
|---|---|---|
| 当前模型 | 0.8B，24 层 Gated-DeltaNet(18)+attention(6)，**非 MoE**，GGUF Q8_0 ~774 MiB | 本报告是"未来跑 MoE"的预案 |
| GPU/后端 | `kev-backend-vulkan.dll`（GGML_VULKAN，`n_gpu_layers=-1` 整模型 offload）、CPU 后端 | MoE offload 需落在这些后端上 |
| 宿主硬件 | **AMD Ryzen AI MAX+ PRO 395（Zen5）+ AMD iGPU gfx1151，统一内存** | **统一内存 = 高带宽 CPU↔GPU 互连 → UVA/zero-copy 前提成立**，这是最大利好 |
| vendored llama.cpp | 近期版本，**已内置** MoE 全套：`build_moe_ffn`、`ggml_ffn_moe`、几十种 MoE 架构（qwen3moe/deepseek2/glm4-moe/olmoe/...）、`n_expert_shared`、per-layer `n_expert_used`、`tensor_buft_overrides` | **无需改 llama.cpp 即可静态把专家放 CPU** |
| 后端 C-ABI | `kev_backend.h`：`llama_model_params` 透传（`n_gpu_layers`、`tensor_buft_overrides`），单 llama 规则 | 静态 offload 可经 C-ABI 参数直接实现，不动 ABI 结构 |
| 计算受限风险 | kevcpp 报告 CPU GEMM 计算受限（compute-bound），VNNI/dpbusd 数值错误需剪除 | CPU 侧执行冷专家时需用**已验证正确**的 CPU 核（避开 VNNI dpbusd 坑） |

### 4.2 kevcpp 可行，但"逐 token swap 到 VRAM"不可行 —— 三层落点

结合 4.1 与第 3 章，kevcpp 引入 MoE offload 有**三条可行路径，按成本/收益排序**：

**路径 A（成本最低、收益确定）——静态把专家层留在 CPU + 其余留 GPU（llama.cpp 原生能力）**
- 用 `kev_backend_params` 新增一两个字段（如 `n_cpu_moe_layers`），在后端 `op_init` 里对 `llama_model_params` 设置
  `n_gpu_layers`（attention/shared 上 GPU）+ **`tensor_buft_overrides`**（正则匹配 `blk.*.ffn_*_exps.weight` 留 CPU）——
  **等价于 llama.cpp `--n-cpu-moe`（PR #15077），无需改 llama.cpp 源码，只动 kevcpp 后端的 params 装配**。
- 效果：MoE 专家权重（体积大头）不进 VRAM；attention/shared/KV 在 GPU。**零精度损失**（CPU 核与 GPU 核数值不同，需验证/量化——见 4.3）。
- 局限：每 token 专家都被 CPU 算，若模型热专家比例高，CPU 侧会成瓶颈；不像"缓存 grad 专家在 GPU"那样利用局部性。

**路径 B（把 hot 专家放 GPU、cold 专家留 host —— 命中率驱动的 GPU 端专家缓存）**
- 对应 llama.cpp `--moe-cache-mib`（PR #29887）、vLLM `--moe-expert-cache-size`（PR #37190/#50562）的 GPU 端 LRU/LFRU 缓存：
  - 全部专家常驻 **CPU pinned** 内存；GPU 只留**最热的 N 个专家**的小缓存。
  - 前向时 miss → **仅 H2D 上传该专家（miss-only）**；hit → 直接用。
  - **小 batch（≤32 token）走缓存、大 batch 退回到常规 offload 路径**（避免传输放大）。
  - `expert_map`（全局专家 id→GPU slot，缺席=-1）喂 GGML 内核。
- 这是"**按需把专家加载到显存**"最接近字面目标的实现——但注意它把**热专家**常驻显存、**冷专家**留在内存按缺即取，而非应答式逐个 swap。
- 落到 kevcpp：这是**改动量最大**的一条，需在 vendored llama.cpp 的 MoE 执行路径（`build_moe_ffn` / `MUL_MAT_ID`）上做缓存逻辑，或移植 #29887/#30059。对 kevcpp 的 C-ABI，可通过新增后端 ops（如 `set_expert_cache_capacity` / `expert_cache_stats`）暴露控制面。
- 局部性越大（routing-consistency 高、活跃专家少）收益越高；用 kevcpp 的 0.8B 梯队小 MoE（如 Qwen3-MoE 0.6B A3B 类）实测 hit rate 是最佳验证。

**路径 C（追求最低性能影响 —— on-host 冷专家 CPU 量化核执行，KTransformers 途径）**
- hot 专家在 GPU，冷专家留在 CPU 用 **AMX/AVX INT4/INT8 核**执行（根本不做 H2D），彻底避免"搬专家"的传输与关键路径问题。
- 这是论文流派"预取藏传输"与工程流派"CPU 量化执行"的合流，也是 KTransformers 的完整产品化做法。
- 落到 kevcpp：CPU 侧要具备**经过验证**的 INT4/INT8 量化 GEMM 核（注意 kevcpp 已发现 VNNI dpbusd 数值错误，须用已证明多正确性的 CPU 路径），复杂度高，适合作为第二阶段。

### 4.3 「最低性能影响」的关键设计清单（综合论文 + 开源，直接适用于 kevcpp）

1. **冷热分离驻留**：hot 专家常驻 GPU，cold 专家留 pinned host（或 CPU 算）。`cache 容量 ≈ 活跃专家数的 ~2x` 是 routing-consistency 论文推荐的最优点。
2. **预测 + 预取 + overlap**（核心）：用**层间预测**（HOBBIT 下一层 top-1 专家预测 96%）驱动**异步预取**，把 H2D/统一内存 DMA 藏进上一批专家计算/attention。预测不对**只影响传输效率、不改输出**（SpecPrefetch 语义最安全）。注意 vLLM 实测跨层专家预测率仅 ~3.4% —— **预测收益高度依赖模型**，须实测再定是否上。
3. **miss-only 上传 + LRU/LFRU 换出**：只传用到的专家；LFRU（频度加权）防前层霸占缓存；`cache` 淘汰盯"miss 代价"而非命中率（LHU / least-stale）。
4. **混合精度兜底**：低重要性专家 miss 时拉 **int4** 版（HOBBIT/DynaExq 2-4x 降传输、精度几乎不掉）；用 gate 输出做快速重要性。
5. **批量拷贝构造**：同一专家的 (gate,up,down) **捆绑**成一块读；int4 分两段按需；**32 线程并行读**摊薄 first-byte；DRAM 预分配避免 realloc；**page 对齐 + MADV_DONTNEED** 释放防 RSS 常驻。
6. **统一内存红利**：kevcpp 宿主是 AMD Ryzen AI MAX+ 统一内存 → **UVA/zero-copy 或统一内存零拷贝直读**是可优先采用的方式（vLLM UVA 路线），其对"快速互连"的要求在本机天然满足。
7. **CUDA graph 冲突**：运行时改设备端 bank / expert_map 会失效 graph 捕获（vLLM 被迫 `--enforce-eager`）。kevcpp 后端若不启用 graph 则无此问题；若启用需注意。
8. **数值一致性红线**：kevcpp 有**逐位一致(bit-identical) 的 hard requirement**。offload 走 GPU 时，GPU(FP32/量化) 与 CPU 核数值天然不同，**过半/近并列选项可能翻转**（kevcpp §5.3 已记录近并列敏感）。因此：
   - 方案的选择必须显式声明**是否追求逐位一致**；
   - 若要求一致，只能把**同一条执行路径**的权重以**相同量化/精度**offload（如 Q8 GPU+CPU 同核），或用 **UVA 直读不重算**（读的是同一批权重、同核 → 可保持一致）；
   - 建议新增 `--moe-offload` + `--tolerance` 开关，让用户选择"精度优先"或"吞吐优先"。
9. **GPU 显存判定**：初始化时用 `ggml_backend` 探测可用 VRAM，按**满 VRAM 放注意力/共享 + 剩余放热专家**的方式自动分配 N 个 GPU 专家缓存位。

### 4.4 kevcpp 落地建议（分阶段）

- **阶段 0（决策）**：明确 kevcpp 是否引入 MoE 模型、以及**是否维持逐位一致红线**。技术可行，但"是否值得"取决于真实模型与流量（kevcpp 现状0.8B 模型远不需要 offload）。
- **阶段 1（静态 offload，凭证）**：后端新增 `n_cpu_moe_layers` 参数 → `tensor_buft_overrides` 把 MoE 专家留 CPU、其余上 GPU。**改动集中在 `kev_backend_{vulkan,gpu}.cpp` 的 params 装配，不动 llama.cpp、不动 C-ABI 结构（仅在结构体末尾追加字段）。**
- **阶段 2（GPU 端专家缓存）**：移植 llama.cpp `--moe-cache-mib`(PR #29887) 的 miss-only LRU 缓存；控制面经 C-ABI 新 op（`expert_cache_config` / `expert_cache_stats`）暴露给 KevModel。
- **阶段 3（预取 / CPU 混合，性能最优）**：做层间预取或 KTransformers 式 CPU 冷专家执行；此时才谈"最低性能影响"。
- 每阶段用 kevcpp 现有的**字节保真 / tok/s 基线**做回归（复用 `bench_bf16_cpu.py`、`logs/perf_authoritative_baseline.md` 方法论）。

---

## 五、关键数字速查

| 项 | 数值 | 来源 |
|---|---|---|
| Mixtral-8x7B 一层 CPU→GPU 加载 | ~80 ms（PCIe4 32GB/s） | HOBBIT |
| 同层 RTX 4090 计算 | ~3 ms | HOBBIT |
| 专家加载占解码耗时 | RTX4090 85.5% / Jetson 94.5% | HOBBIT |
| 单冷专家 one-target fetch | 63.2µs = 单层预算 14% (1×H100) | vLLM RFC #57794 |
| hoBBIT 层间 top-1 专家预测 | 96% | HOBBIT |
| llama.cpp `--moe-cache-mib` | Qwen3.8 Q4_0 RTX5090：30.8→67.8 tok/s，hit 89% | llama.cpp PR #29887 |
| vLLM 专家缓存 | host→device 流量 -73.6~77.1%（AcceptMoE）| 论文 |
| cache 容量最优 | ≈ 活跃专家 2x | routing-consistency 论文 |
| DeepSeek-R1 FP8 (KTransformers) | 227.85 tok/s，8×L20+Xeon | KTransformers |

---

## 六、权威来源（链接）

**论文**（arXiv，可点开精读）：
- MoE-Lightning https://arxiv.org/abs/2411.11217 · HOBBIT https://arxiv.org/abs/2411.01433 · PowerInfer https://arxiv.org/abs/2312.12456 · PowerInfer-2 https://arxiv.org/abs/2406.06282 · LLM-in-a-flash https://arxiv.org/abs/2312.11514 · Speculating Experts https://arxiv.org/abs/2603.19289 · MoE-SpeQ https://arxiv.org/abs/2511.14102 · SP-MoE https://arxiv.org/abs/2510.10302 · AcceptMoE https://arxiv.org/abs/2608.02989 · SpecMD https://arxiv.org/abs/2602.03921 · SpecPrefetch https://arxiv.org/abs/2607.24787 · DraftExpert https://arxiv.org/abs/2607.24434 · FineMoE https://arxiv.org/abs/2502.05370 · HybriMoE https://arxiv.org/abs/2504.05897 · routing-consistency https://arxiv.org/abs/2505.16056

**开源工程**：
- llama.cpp `--n-cpu-moe` https://github.com/ggml-org/llama.cpp/pull/15077
- llama.cpp `--moe-cache-mib` https://github.com/ggml-org/llama.cpp/pull/29887
- llama.cpp 汇总(实现要素清单) https://github.com/ggml-org/llama.cpp/pull/30059
- llama.cpp multi-gpu(MoE split 限制) https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- llama.cpp expert-op 回落 CPU https://github.com/ggml-org/llama.cpp/issues/30224
- vLLM OffloadConfig https://github.com/vllm-project/vllm/blob/main/vllm/config/offload.py
- vLLM MOE expert cache (LFRU) https://github.com/vllm-project/vllm/pull/37190 / https://github.com/vllm-project/vllm/pull/50562
- vLLM RFC #57794 (per-token swap 不可行 + 冷尾降级) https://github.com/vllm-project/vllm/issues/57794
- SGLang offload https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/arg_groups/fields/exec_.py
- KTransformers https://github.com/kvcache-ai/ktransformers · CPU-GPU Expert Scheduling https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/experts-sched-Tutorial.md

---

## 附：一句话指导意见

> **可行，且 kevcpp 有独特的统一内存优势；但不要做"逐 token 把专家 swap 进显存"——那已被证明会击穿关键路径。**
> 最优路径是 **A(静态把专家留 CPU) → B(GPU 端 miss-only 热专家缓存) → C(冷专家 CPU AMX/INT4 执行 + 层间预取)** 三级演进，
> 全部可用 vendored llama.cpp 现成能力 + 后端 C-ABI 扩展实现，且多数无需改动 llama.cpp 源码。
> "最低性能影响"的钥匙是**预取 + overlap 把统一内存/PCIe 传输藏进计算**，并严格守住 kevcpp 的数值一致性红线。