# kevcpp 性能优化综合评估（管理会话汇总）

> 本文件是**只读调研 + 分派管理**会话的综合结论，由多个子代理分别调研、本会话负责分派与汇总，
> 再由主会话补齐关键纠偏（SIMD 结论）。**本文件不含任何代码改动。**
> 前置必读：`docs/PERF_ANALYSIS.md`（已有大量实测基线）与 `docs/LAYER_PRUNING_ANALYSIS.md`。
>
> 📌 **开始新优化前先看 §6「已探索方法档案」**——里面是**已经走通/证伪/实测无收益**的所有方法，
> 避免重复踩坑（尤其不要再试"开 AVX-512/VNNI"或"P.1 合成单次 decode"）。

---

## 0. 现状基线（引用 PERF_ANALYSIS 实测，threads=32，1 问 6 选项，0.8B Q8_0）

| 段 | 耗时 |
|---|---|
| state prefill（38 tok）| ~323ms |
| branch decode（89 tok）| ~290ms |
| **cold 总** | **~660ms** |
| warm（state 命中，只跑 branch）| ~282ms |
| head/softmax | ~1-2ms |

主瓶颈：读 Q8_0 权重的内存带宽（每 token **506.12 MiB**，隐含 ~183 GB/s，见 §8）+ llama.cpp 每次 decode 的图构建/调度固定开销。

---

## 1. ⚠️ 已证伪 / 已被推翻的优化（重要：避免走弯路）

### 1.1 ❌ P.1「state+branch 合成单次 127-token decode」
附录 B 实测定论：Gated DeltaNet / hybrid recurrent 在"整行一次前向" vs "截断(38)+续接(89)两次前向"
之间**不逐位一致**（|Δp|≈0.0013），不满足任务硬约束。**不可行。**

### 1.2 ⚠️「SIMD 被关、Q8_0 跑标量、开回 AVX2 能回收 150-250ms」——**结论有误，已纠偏**
- 表面证据：`build/CMakeCache.txt` 里 `GGML_AVX/AVX2/AVX512:BOOL=OFF`。
- 真相：根 `CMakeLists.txt:17` `GGML_NATIVE=ON(FORCE)` → `ggml/CMakeLists.txt:141-142` 把 `INS_ENB` 置 OFF →
  AVX 系列 option **默认值是 OFF（缓存即显示 OFF）**；随后 MSVC+`GGML_NATIVE` 触发
  `ggml/src/ggml-cpu/CMakeLists.txt:288-291` `include(FindSIMD.cmake)`，后者**用非缓存 `set()`
  在 ggml-cpu 子目录作用域里把 `GGML_AVX2/AVX512=VON`**（`FindSIMD.cmake:92`、`:98-99`）。
  - 决定性佐证：`build/CMakeCache.txt` 中 `AVX2_FOUND=TRUE`、`AVX512_FOUND=TRUE`、`FMA_FOUND=TRUE`。
- **结论：当前 Release 构建实际已经用 AVX2（乃至 AVX512）编译了 ggml-cpu 内核，并非标量。**
  无需、也无益于"重新开 SIMD"。Cache 里的 OFF 只是 `INS_ENB=OFF` 造成的阴影，不反映编译结果。

> 依据：子代理 B 只看 cache 得出"标量内核"的误判；追踪子代理读 `FindSIMD.cmake` 后推翻。管理会话再次核实 `FindSIMD.cmake` 与 `*_FOUND=TRUE` 佐证。

---

## 2. 真正值得做的优化（按确定性与性价比排序，均保证不破坏概率逐位一致）

### 🔧 P0a — 增加「state 前缀预热」入口（最推荐，纯 src，零 vendored，零一致风险）
- 现状：`PrefixLRU`（`kev_prefix.h:15-42`）只有 `lookup/register_/invalidate`，唯一 prefill 入口是请求内触发的
  `ensure_state`（`kev_model.cpp:134`），`init`（`:32-91`）与 `main_server` **无任何预热接口**。
- 机会：把**常见 system prompt + state** 在启动/空闲时 prefill 进 LRU slot，之后同类请求直接从 cold→warm
  （~660ms→~282ms）。命中率高时是**最大单点收益**。
- 改动面：`kev_model` 暴露一个按给定 state token-id `prefill_state(ids)` 方法（复用 `ensure_state`/`decode_seq` 纯 prefill 路径），`main_server` 启动时调用。
- 风险/收益：零一致风险（纯 prefill，与现有 warm 一致）；收益视业务前缀稳定性，命中率高则近减半延迟。

### 🔧 P0b — 给同一 KevModel 的并发 compute 加互斥（正确性加固，src）
- 现状：httplib 默认**已是线程池**（`httplib.cpp:8136-8140` 默认 `ThreadPool(max(8, hardware_concurrency-1))`；
  `httplib.h:183-187`），但 `main_server.cpp:114` 所有 handler 捕获**同一个 KevModel**，且 `KevModel`
  内 `ctx_/lru_/scratch_seq_` **无任何 mutex**（`kev_model.h:89-98`）——多并发请求会**竞态**（互相覆盖
  KV/scratch/LRU），并非安全串行。
- 机会：给 `probs()`/`systemone_json` 加一把 `std::mutex`，把竞态降为"严格串行"，保证正确性/计时可信。
- 量级：非性能提升，而是修复隐患，是后续并发优化的前提。

### 🟦 P.2b — llama.cpp 图持久缓存（vendored，最后压延的锤）
- `process_ubatch`（`llama-context.cpp:1393-1466`）只在 `res->can_reuse(gparams)` 时复用图，且**现有缓存只有
  2 个槽** `gf_res_prev[n_outputs>0]`（`:2420`）；kevcpp 的 state(38, n_outputs=0) 与 branch(89, n_outputs>0)
  分居两槽，每次请求 branch 图都要重建。
- **复用的真正拦路虎（recurrent 特有）**：`llm_graph_input_rs::can_reuse`/`mem_hybrid::can_reuse`
  （`llama-graph.cpp:345-361` / `:1116-1137`）还硬判 `head`/`rs_z`/`s_copy`，而 recurrent `find_slot`
  （`llama-memory-recurrent.cpp:505-714`）每次都重排 cell、改 head → **即使形状相同也常重建**。
- 最小改法：`llama-context.cpp` 把 `gf_res_prev[2]` 升级为按 `(n_tokens,n_seqs,n_seq_tokens,n_outputs,equal_seqs)`
  分桶的 map + 放宽 rs/mem_hybrid 的 `head`/`rs_z` can_reuse（`set_input` 本就每次重填 head，图不把 head 烘焙进拓扑）。
- 一致性保障：缓存图必须与全新 build 的算子序列/累加次序完全等价（`s_copy`/`head` 填充仍逐位复刻基线 set_input）。
- 内存代价：每桶一份 compute buffer（n_ubatch=256 时 ~245MB）；(38,·,0) 与 (89,·,k) 两桶 → 内存翻倍 ~490MB。
- 预期：冷再降 ~50-150ms。**需维护 vendored patch，是收益中、代价中的活。**

### 🟦 P.4 — 多问一次 decode（多问业务才受益；单问=0）
- 新事实：llamacpp **原生已支持**多 seq 共享前缀一次 forward——`llama_memory_recurrent::find_slot`
  本支持 `n_seqs>1` 且 `GGML_ASSERT(equal_seqs())`（`:520/:589-650`），DeltaNet state 按 `n_seqs` 维独立
  （`delta-net-base.cpp:27`），**无需改 vendored 的 mem_hybrid 索引**。
- 缺的只是 kevcpp `src/` 驱动层：`rows_from_slot`（`kev_model.cpp:157-195`）逐 row `seq_cp`+`decode_seq`。
  需把多问合成**等长（pad 到 max）**的 ubatch + 多 seq_id。
- 收益：多问 N 时把固定开销/图 build 摊成 1/N；单问=0。
- 风险：pad 到等长可能改变 chunked 段边界 → **有逐位一致风险**，需对拍确认"pad 是否改变非 pad 输出"。

### 🔧 小项（确定性、低风险，但量级小）
1. **消除 `systemone_json` 双 encode + 三遍 tokenize**：`kev_api.cpp:190` 先 `model.encode`，`probs()`（`:194`，
   `kev_model.cpp:200-202`）内部又 `encode`+`rows_of` 一遍；`:203` 还对 answers 再 count_tokens 一遍。合计 ~1-3ms。纯 src 可消。
2. **修 `kev_prefix.cpp` `register_` 潜在重复项/重复赋值**（`kev_prefix.cpp:36-39`，当前被 lookup erase 后不可达，latent）。
3. **scratch 序列定期 `seq_rm` 回收**：`rows_from_slot` 每 branch 只 `seq_cp` 重指尾（`:169`），长分支残留 cell 无法回收 → 长跑内存增长。

### 👎 明确不做
- head 向量化 / 改 fp16：`kev_head.h:29` Linear 已是 `std::vector<float>`（fp32），本来无 fp16 换算浪费；head 规模 ~2MB、<1ms，性价比≈0。
- `llama_memory_seq_cp` 优化：recurrent 已 O(1) 别名，免费。
- 开启 `GGML_LLAMAFILE`：面向大 K 大 batch GEMM，对 0.8B 单行 GEMV 收益有限，且有累加次序差异（一致风险）。不推荐。
- `GGML_CPU_REPACK`：需对 mmap 只读模型额外拷一份重排权重，0.8B 单行 GEMV 收益 ~15-30% 但回内存，且根 CMake 已 FORCE OFF（`CMakeLists.txt:24`），若要开需改根 CMake。次可选。
- LCP 增量前缀缓存：recurrent 上"截断共享+只补差异"与"整体重 prefill"能否逐位一致未验证（PERF 附录 B 已证 recurrent 拆两次不另一致方向），值低风险高，不优先。

---

## 3. 推荐推进顺序（保证逐位一致前提）

1. **P0a 前缀预热**：纯 src、零风险、命中率高时近减半延迟（cold→warm）。**先做。**
2. **P0b 加锁**：修复并发竞态，给吞吐优化打正确性地基（src）。
3. **P.4 多问 batch**（**仅当业务多问**）：src 驱动层合成等长多 seq，需对拍 pad 一致性。
4. **P.2b 图持久缓存**（vendored，需 patch 维护）：冷延的最后 ~50-150ms，且需连带放宽 recurrent head 校验。
5. 小项（双 tokenize 消除、scratch 回收、register_ 修 bug）随手做。

---

## 3.5 「参考其他项目」的落地方式（离线替代）

> 本会话无网络（web_search/web_fetch provider 未注册，多轮实测不可用），故"参考其他项目/论文"采用
> **离线的本地参考项目**替代，即使无网也让"对比参考"落到实处。待后续在可联网会话中补齐在线论文调研。

### 3.5 结论（子代理已交付，离线"参考其他项目"完成）

**原版 Python `kev`（`E:\van\projects\kev\kev\`）：**
- `shared_prefix.py`：是**训练**路径（含反向/gradient checkpointing 的 `forward_rows_batch(shared_prefix=True)`），其 fp32 reference 核下与 row form 也只对到 ~1e-5、Triton 核到 TF32 精度——**非逐位一致**；serving 一律走 `prefix/probs_with_prefix`。因此"state+branch 合并前向"在 kevcpp 的逐位一致硬约束下**不可移植**（与 P.1 已证伪一致）。它也没有比 kevcpp `PrefixLRU` 更强的 LCP 增量缓存（同为"完整 state 指纹精确命中"）。→ 无更强者，无需移植。
- `cuda_graphs.py`：CUDA 图"捕获一次 + 按形状桶 LRU 复用多次"（`_replay:206`/`capture_pending:224`，GRAPHS_KEPT=256），文档明说结果等于 eager 到浮点重结合、**非 bit-for-bit**。机制上与 **P.2b（图持久缓存）同构**，只取其"首次 eager 跑→按形状桶复用"的调度思想。CPU 无 kernel-launch replay 收益。
- `fused_qwen35.py`：推理专用 kernel 融合（6 投影拼 GEMM、chunked gated delta 单核、RMSNorm/SwiGLU/softplus 融合）。CPU ggml 上近似对应批矩阵合并维（P.4 多 branch batch 已有此意）。
- `model.py`/`serve.py`：`_rows_hidden`/`_readout_many`/`PointerHead.many`（多 row 一次 forward、readout 批化）+ `PrefixCache.plan/store`（批内按 state 去重、miss 一次 forward 同时产 prefix）—— 对应 **P.4 / P0a**。

**上游 `llama.cpp`（`E:\van\projects\llama.cpp`）——决定性发现：**
- 递归 hash 对比，上游 HEAD 与 vendored `third_party/llama.cpp` **787 个 C/C++/cmake 核心文件 0 处不同**（ggml.c、ggml-cpu.cpp、quants.c、llamafile-sgemm、repack.cpp、llama-graph/context/hybrid/memory、delta/recurrent 等全部逐字节一致）。**vendored 已是当前最新上游**，`git log -80` 中与 CPU/Q8_0/recurrent 相关提交（x86 tiled flash attn、AVX512-FP16 f16 点积等）**均已含在 vendored 内**，其余提交落在 CUDA/HIP/Vulkan/Metal/sycl/server 等非 kevcpp 路径。
- **结论：任务二无可上——上游没有比 vendored 更新的 CPU 内核可白捡；kevcpp 现有 CPU 路径已是最新上游。**

**综合两路参考，值得后续采用且不违背逐位一致的 Top-5（来源）：**
1. **多 branch 单次 batch decode（P.4）**：`model.py:_rows_hidden/_readout_many/PointerHead.many`(:312/:471/:197) + `cuda_graphs.py:283-299`(按长分组/右对齐)。多问摊薄固定开销；单问=0；一致性需对拍 pad 边界。
2. **批内 state 去重 + 前缀预热（P0a）**：`serve.py:PrefixCache.plan/store`(:49-67) + `probs_and_prefix`(:406)。一致性：无（纯 prefill）。
3. **"先 eager 跑→按形状桶 LRU 复用"调度（P.2b 同构）**：`cuda_graphs.py:_replay/capture_pending`(:206/:224)。一致性：中（需保证缓存图 == 全新 build 算子序列）。
4. **readout 批化**：`model.py:_readout_many`(:471)/`PointerHead.many`(:197)——多问把 decide/opt hidden 集中取一次；kevcpp head 已近零但多问时优于逐问 `hidden_at_output`（`kev_model.cpp:180-188`）。一致性：无。
5. **（默认不做）shared_prefix 合并前向**：已证 P.1 违一致，仅存档，若未来引入"非逐位一致"配置可参考 `shared_prefix.py:30-141`。

---

## 4. 一句话给 owner

- 当前冷 ~660ms 已接近该 CPU/模型的内存带宽下界（~400ms）+ 一次首访建图+(第 2 次图 build)固定开销。
- **不要再纠结"开 SIMD"**（实际已开 AVX2/AVX512）；**不要再做 P.1 合并 decode**（recurrent 数值不一致）。
- 剩余最确定的三条：**① 前缀预热（cold→warm 近减半）、② 多问打包 batch、③ 图持久缓存（需 vendored patch，代价中）**，
  外加**并发加锁**修复竞态。

---

## 5. 实施结果（本轮新开两个子代理落地 + 统一构建/对拍验证）

> 本轮按 owner 决定**不开网**，把 §2 里"纯 src + P.2b"落地，P.4 暂不做（有 pad 逐位一致风险，留待业务具多问时再上）。两个子代理分别负责 `src/` 与 vendored，**write-scope 不相交**。

### 5.1 落地内容
**子代理 A（纯 src，5 项）** — `git status` 确认落 5 个文件：
1. P0a 前缀预热：`KevModel::prewarm_state`（`kev_model.{h,cpp}`）+ `main_server --prewarm <file>` 启动参数（`main_server.cpp`）。
2. P0b 并发加锁：`KevModel` 增 `std::mutex compute_mutex_`，`probs` 计算段（ensure_state+rows_from_slot）持锁（`kev_model.cpp`）。
3. 消除 systemone 双 encode：新增 `probs(const KevEnc&)` 重载，`systemone_json` 一次 `encode` 后复用（`kev_api.cpp:195`）。
4. 修 `PrefixLRU::register_` 潜在重复项（`kev_prefix.cpp`）。
5. scratch 每问 `llama_memory_seq_rm` 回收（`kev_model.cpp` rows_from_slot）。

**子代理 B（P.2b 图持久缓存，vendored）** — `third_party/llama.cpp/src/llama-context.{h,cpp}`：
- 把"两槽 last-used"`gf_res_prev[2]` 升级为按 `graph_shape_key` 分桶的 `std::map` 持久缓存（按 context 实例）。
- 命中且跨桶时 re-bind（sched_reset+alloc，**跳过 `model.build_graph()`**）。
- **对 head/rs_z 采取保守策略（未放宽）**：已证 head 被烘焙进常量写目标偏移（`build_rs` `llama-graph.cpp:3518`），放宽会导致数值漂移；故首以 `can_reuse` 为最终权威，任何桶都不绕过它 ⇒ "缓存图 ≡ 全新 build"前提成立，绝无假阳性复用。
- ⚠️ 子代理 B 引入 1 处编译错误已由 lead 修复：`llama-context.h:266` 在 `struct graph_shape_key`（:388）定义前用了它做返回类型（C2059/C3646），已加前向声明 `struct graph_shape_key;`。

### 5.2 构建与逐位一致验证（全部通过）
- **构建**：`scripts/build.ps1`（MSVC Release）exit 0，产出 kev.exe / headbench.exe / m7_bench 等。
- **m7_bench（端到端，`req_m7.json` + `--lora kev-lora-qwen35.gguf`）**：cold 与 warm 概率 **逐位一致** golden `0.0125 0.6794 0.0697 0.0265 0.2043 0.0075`（state_len=161 与 golden 相同）。⚠️ 注意 golden m7 日志用 `--lora` 生成；不带 lora 的概率不同属预期，非回归。
- **`tests/parity.py`**：headbench + readout `max|dp|=0.000e+00, argmax_all_match=True`。
- **`tests/parity_encode.py`**：ALL PASS。
- **服务器端到端**：`--prewarm` 把 15-token state 预填进 LRU；`POST /v1/systemone`（curl --data-binary 传 UTF-8）返回 `input_tokens=127`、答案概率 `中标:0.0125 产线产能:0.6794 其他:0.0697 厂房:0.0265 投资:0.2043 购地:0.0075` 与 golden **完全一致**，确认 `probs(KevEnc)` 双 encode 消除路径逐位正确。（用 PowerShell `Invoke-WebRequest` 发送中文会乱码/错参数，是测试侧编码问题，非代码问题。）

### 5.3 诚实说明：P.2b 的实际收益可能有限
- 子代理 B 保守处理（不放宽 head/rs_z）保证正确性，但 recurrent `find_slot` 的 head 单调前移下，"同形状 + head/rs_z 恰好复现"的命中概率低 → P.2b 在 recurrent 上可能只给出**有限甚至为 0 的提速**，属 recurrent 模型固有上限，非 bug。
- m7 cold 848.9ms / warm rows 414ms 比 golden（655/282）慢，主因是本机当时负载噪声（PERF 附录 B 已指出本环境跨时间点计时不可比），**不能据此断言回归**；逐位一致是硬保证且已证。
- 后续若想真提速：受控 A/B（同一负载窗内 best-of-3）实测；或业务多问时启用 P.4。

---

## 6. 已探索方法档案（F.O.M.，避免后续重复工作）

> **目的**：把所有已经走过的优化方法集中列成一张"去过/没去过"的表，标注**结论 / 证据 / 收益 / 是否破坏逐位一致 / 是否可复用**。
> 后续任何人做优化前，先看这张表，**命中「❌/≈0」的方法不要再试**；命中「✅」的方法直接看对应落地/接口。

### 6.0 硬约束（贯穿所有方法的红线）
- **概率必须逐位一致**（与 `tests/golden` / `parity.py` / `parity_encode.py` 对拍）。
- 任何改累加次序、反量化方式、token 偏移、前向算子序列的方法 = 大概率破坏该约束，做了等于白做。
- 只改 vendored `third_party/llama.cpp` 需打 patch 并长期维护；`E:\van\projects\llama.cpp` 上游**不可改**。

### 6.1 已确认「不可行 / 无收益」的方法（不要再试）

| # | 方法 | 判定 | 证据 / 原因 | 收益 | 破坏逐位一致? |
|---|---|---|---|---|---|
| F.01 | **P.1：state+branch 合成单次 127-token decode** | ❌ 不可行 | PERF 附录 B 实测：recurrent（Gated DeltaNet）"一次整行" vs "截断+续接两次" **|Δp|≈0.0013** | — | 是（必然） |
| F.02 | **"SIMD 被关、Q8_0 跑标量、开回 AVX2 回收 150-250ms"** | ❌ 误判（实际已开） | `GGML_NATIVE=ON`→`FindSIMD.cmake` 非缓存 `set()` 置 AVX2/AVX512 ON；`AVX2/AVX512/FMA_FOUND=TRUE` | — | 不适用 |
| F.03 | **强制开 `GGML_AVX512_VNNI`+`GGML_AVX_VNNI`** | ≈0 无收益 | 本轮实测 A/B（见 §6.2 前）：VNNI OFF warm 336–381ms vs ON 321–390ms，无差别 | ≈0 | 低（非主因） |
| F.04 | `GGML_LLAMAFILE`（llamafile sgemm） | ❌ 不推荐 | 面向大 K/大 batch GEMM；0.8B 单 token GEMV 带宽 bound；Q8_0 只有 AVX 级 `tinyBLAS_Q0_AVX`；有舍入差异 | ≈0–小 | 是 |
| F.05 | `GGML_CPU_REPACK` | ❌ 对 Q8_0 无用 | repack 的 Q8 阻塞内核是 **Q4_0/Q4_K 权重专用**，Q8_0 权重模型零收益；需拷贝 mmap 只读权重 | 对 Q8_0=0 | 是（如开） |
| F.06 | **LCP 增量前缀缓存**（仅共享前 k token） | ❌ 不优先 | recurrent 上"截断共享+只补差异"与"整段重 prefill"逐位一致性未验证，风险高；`PrefixLRU` 精确匹配已够 | 高但赌一致性 | 高风险未证 |
| F.07 | head 向量化 / 改 fp16 | ❌ 不需 | `kev_head.h:29` Linear 已是 fp32，无 fp16→fp32 浪费；head ~2MB、<1ms | ≈0 | 无 |
| F.08 | `llama_memory_seq_cp` 优化 | ❌ 已最优 | recurrent 已 O(1) 指针别名，不做字节 memcpy | 0 | 无 |
| F.09 | `shared_prefix.py` 合并前向 | ❌ 不可移植 | 训练路径，非 serving；fp32 只对到 ~1e-5，非逐位 | — | 是 |
| F.10 | Intel AMX | ❌ 不适用 | 本机 **AMD Zen5**，无 AMX | — | n/a |

### 6.2 ✅ 本轮已落地并通过逐位一致验证（可直接复用）

| # | 方法 | 落地位置 | 验证 |
|---|---|---|---|
| F.11 | **P0a 前缀预热** | `KevModel::prewarm_state` + `main_server --prewarm <file>` | 启动预填 state → 后续同 state 请求走 warm |
| F.12 | **P0b 并发加锁** | `KevModel` 加 `std::mutex compute_mutex_`，`probs` 计算段持锁 | 修竞态→严格串行；无数字变化 |
| F.13 | **消除 systemone 双 encode** | `probs(const KevEnc&)` 重载 + `systemone_json` 复用 enc | /v1/systemone 返回概率与 golden 逐位一致 |
| F.14 | **修 `PrefixLRU::register_` 重复项** | `kev_prefix.cpp` 同 seq 原位刷新 | 防御性改进 |
| F.15 | **scratch 每问 `seq_rm` 回收** | `kev_model.cpp` rows_from_slot | 防长跑内存增长；m7 逐位一致 |
| F.16 | **P.2b 图持久缓存（vendored）** | `llama-context.{h,cpp}` 分桶持久缓存，保守 head/rs_z | 编译成功、m7/parity 逐位一致；**收益已被受控测试重估为最多 ~70ms**（§7.2，非 PERF 估的 250ms），且 recurrent 命中率低实际可能更少 |

### 6.3 ⌛ 已识别、尚未尝试 / 待业务触发的方法

| # | 方法 | 触发条件 | 风险 |
|---|---|---|---|
| F.17 | **P.4 多问一次 batch decode** | 业务变多问（N≥2）时启用 | pad 到等长可能改动 chunked 段边界，有逐位一致风险，需对拍 |
| F.18 | 换 **Q4_0/Q4_K 量化权重**吃 Zen5 VNNI | 愿意为提速牺牲逐位一致（需另加对拍开关） | 破坏逐位一致 |
| F.19 | P.2b「放宽 recurrent head/rs_z can_reuse」 | 若确认 `set_input` 能重填 head 写偏移且保逐位一致 | 大概率数值漂移，已被 B 否决 |
| F.20 | 受控计时基线（供未来 A/B） | 需要可靠性能数字时 | 本机负载噪声大，须同窗 best-of-3 |

### 6.4 关键经验 / 坑（避免重复踩）

1. **不要只看 `build/CMakeCache.txt` 判"AVX 没开"**——`GGML_NATIVE=ON` 会隐蔽地用 FindSIMD 置 ON（影子 OFF）。要看 **`*.vcxproj` 的 `<EnableEnhancedInstructionSet>` + `PreprocessorDefinitions`**，或运行 `ggml_cpu_has_avx*`。
2. **Q8_0 是内存带宽 bound、且上游没给 Q8_0 dot 写 AVX-512/VNNI 内核** ⇒ 换 SIMD 指令集对 Q8_0 无意义；要吃 VNNI 得换低比特权重。
3. **golden m7 日志带 `--lora` 生成**；不带 lora 跑 m7 概率会不同，**别误判成回归**。
4. **PowerShell 发含中文的 HTTP body 会乱码/错参数**（`Invoke-WebRequest`）；测 server 用 `curl --data-binary @file`（UTF-8）。
5. **本机计时噪声大**：跨时间点比较会得出错误"倒退/提升"，必须用同一负载窗内 A/B best-of-3。
6. **改 vendored = 长期 patch 负担**：P.2b 这类只在你确实想要 last ~50-150ms 且能接受维护时才做；否则优先纯 src。
7. **子代理落地编译错误要先修**（例：P.2b 曾把 `graph_shape_key` 在定义前用作返回类型 → 加前向声明修复），确认编译通过再进对拍。

---

## 7. 受控性能测试：cold 到底耗在哪（实测分解）

> 使用新增的受控探针 `src/cold_profile.cpp`（独立 bench，`--n 161 --n2 89` 用**真实** req_m7 state 长度复现 cold）。
> 数据为同负载窗 threads=32 采集。结论**修正**了 PERF_ANALYSIS 对"图构建固定开销 ~250ms"的**高估**。

### 7.1 实测数据
**真实 cold 形态（state 161 tok → branch 89 tok，同 ctx，threads=32）：**
```
seg_state  n=161 first_ms=486.5  best_ms=452.9   (首访建图 vs 重复同形)
seg_branch n=89  after_ms=274.0  best_ms=264.8
cold_est  state_1st + branch_after = 760.6 ms   ← 与 m7 cold ~650-800ms 吻合
graph_overhead_first_vs_best      = 42.8 ms      ← 首次建图额外开销(很小)
```

**变长 n 曲线（`--list`，`best_ms` 同形状重复）：**
```
n=1:15.5  n=2:18.8  n=8:29.7  n=16:50.8  n=32:95.8  n=64:182.6
n=89:250.8  n=128:381.4  n=192:554.7  n=256:750.2
→ 每 token 前向带宽 ≈ 2.7–3.1ms（线性，截距 ≈ 12–15ms）
```

### 7.2 结论（修正 PERF 估计）
1. **cold 时间几乎全花在「读权重的内存带宽」**（~2.9ms/token 线性前向），不是图构建。
   - state 161 ≈ 470-486ms，branch 89 ≈ 264-274ms，合计 ~760ms。
2. **固定图/调度开销总共只有 ~70-90ms**（每次 `llama_decode` ~12-15ms + 首访建图 ~40ms），**远小于 PERF 估的 ~250ms**。
   - 因此 **P.2b 图缓存即便完美命中，最多省 ~70ms**（且受 recurrent head 命中率限制可能更少）→ **P.2b 价值被 PERF 高估，不优先**。
3. **带宽是 0.8B Q8_0 的硬下界**，无法靠 SIMD/图缓存突破。精确值（来源：`models/gguf_tensor_dump.py` + `bw_probe` 重测 + `cold_profile`，见 §8）：
   - **每 token 实际流式读 ≈ 506.12 MiB**（Q8_0 trunk 24 层权重；token_embd 257.66 MiB 因 embeddings 模式不被每 token 读）；隐含前向带宽 ≈ **~183 GB/s**，纯读带宽参考 ~246 GB/s（见 §8 表）。
   - 要实质降 cold：① **换 Q4_0/Q4_K 量化**（权重字节减半 → 每 token ~253 MiB → 纯带宽下界 ~1.0ms，cold 可望 ~380-450ms，但破坏逐位一致）；② **减小输入长度**（按 ~2.9ms/token 线性减）；③ **提高 warm 命中**（P0a 预热，同 state 后续请求冷→暖 ~260-290ms，是唯一不牺牲一致性的显著提速）。

### 7.3 附：探针用法（复用，避免再造）
```
build\bin\Release\cold_profile.exe --model D:\share-models\Qwen3.5-0.8B-Q8_0.gguf --threads 32 [--n 161] [--n2 89] [--list] [--reps 5]
```
- `--list` 打变长曲线；默认 `--n 38 --n2 89`（改传 161/89 对标真实 req_m7）；`threads` 默认 32。
- 判读：`best(n)` 斜率=每 token 带宽；`first−best`≈建图固定；`cold_est` 与 m7 cold 对照。

---

## 8. 带宽矛盾已解决：2.9ms/token 是带宽受限（收尾定论）

> 用 **llama/kevcpp 自身计时**（`cold_profile`）做带宽占用交叉验证，加上精确权重字节（`gguf_tensor_dump`）与重测真机带宽（`bw_probe`），把交接 §4 的"带宽矛盾"彻底解开。

### 8.1 交叉验证：llama 自身计时隐含的前向带宽

同一负载窗 `threads=32`、`best-of-N` 复测 cold_profile（n=161 state / n2=89 branch，用真实 req_m7 长度）：

| 段 | best_ms | token 数 | 每 token | 隐含前向带宽（/ 506.12 MiB） |
|---|---|---|---|---|
| state | 446.2–462.4 | 161 | 2.77–2.87 ms | 185–191 GB/s |
| branch | 259.2–268.3 | 89 | 2.91–3.01 ms | 176–182 GB/s |
| **合计/均** | — | — | **≈2.8–3.0 ms** | **≈176–191 GB/s（中位 ~183）** |

- **隐含前向带宽 ≈ 183 GB/s**，与事实2 的解析估计（506.12 MiB / 2.9ms = 183 GB/s）严格吻合 → **GGUF 权重字节解析与 llama 实际运行相互印证**。
- 对照 `bw_probe` 纯读（16T best-of-7，同窗负载噪声复测 247.8 GB/s；参考 240–255 GB/s）：**利用率 ≈ 183/246 ≈ 74%**（区间 71–78%），是正常 DRAM 流式读效率区间。
- **带宽纯下界时间** = 506.12 MiB / 246 GB/s ≈ **2.06–2.16 ms/token**。实测 ~2.9–3.0 ms → **实测是纯带宽下界的 ~1.4 倍**，其中约 **0.8ms（~28%）** 是软件/计算/调度开销。主因是内存带宽 —— **结论：带宽受限**。

### 8.2 旧数据 vs 实测真值（全部来源可复核）

| 项 | 旧（前序估计/旧探针） | 实测真值 | 来源 |
|---|---|---|---|
| 真机纯读带宽 | 33–75 GB/s（旧探针严重低估） | **240–255 GB/s**（≈256 峰值 96%） | `bw_probe` 重测（AVX512 流式读，2GB best-of-7） |
| 每 token 读字节 | ~390–490MB（估） | **506.12 MiB**（Q8_0 trunk，token_embd 不被每 token 读） | `models/gguf_tensor_dump.py` |
| 隐含前向带宽 | ~169 GB/s（由低估字节反推） | **~183 GB/s** | 上述两者相除 = cold_profile 交叉验证 |
| 冷态 | cold 760ms / branch ~260-274ms | 同（本表不改变，仅数字精度） | `cold_profile --n 161 --n2 89` |

### 8.3 为什么"~169GB/s 反推超 256 峰值"的矛盾是假的

- **权重字节被低估**：旧估计 ~390–490MB，实际 trunk Q8_0 权重 = **506.12 MiB**（此前漏算部分 norm/bias 及对 Q8_0 精确字节的估计错误）。
- **把不流式读的 token_embd 算进去了**：token_embd 257.66 MiB 在 kevcpp embeddings 模式下每 token **不读**（input 走 get_rows 只 gather ~1KB；LM head 的 MUL_MAT 不执行）。若把 ~774MB（含 token_embd）/ 2.9ms 当"每 token 带宽"，才得到>256 的伪超峰。修正后 506.12/2.9 ≈ 183 GB/s，**在 256 峰值之内且只为流式读的合理 ~74%**，矛盾消失。

### 8.4 优化推理（保持逐位一致硬约束 F.18 结论不改）

1. **量化才有效**：换 Q4_0/Q4_K（字节减半 → 每 token ~253 MiB）→ 纯带宽下界 ~1.0ms → cold 有望 760→~400-450ms，**但破坏逐位一致**（F.18，需另加对拍开关，不改默认）。
2. **不换量化则带宽不可突破**：纯带宽下界 ~2.1ms 已占实测 2.9ms 的 ~72%，SIMD/图缓存（P.2b）至多收 ~0.8ms 的软件开销里的一小部分（已实测最大 ~70ms），**无法突破带宽**。
3. **唯一不牺牲一致性的实质提速 = 提高 warm 命中**（P0a 前缀预热，把 161-token state 变成 warm 后只跑 branch ~260ms），或减短输入长度。

### 8.5 给 owner 的一句话结论

> **Qwen3.5-0.8B Q8_0 单 token 前向 2.9ms 是内存带宽受限（每 token 流式读 506.12 MiB，隐含 ~183 GB/s，为 246 GB/s 纯读的 ~74%，非计算瓶颈）；此前"超 256 峰值"的矛盾是把 token_embd 误计 + 权重字节低估造成的假象，已解决。要实质提速只能是换低比特量化（牺牲逐位一致）或 P0a 前缀预热走 warm。**