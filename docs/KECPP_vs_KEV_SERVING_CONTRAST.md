# kevcpp 完整服务路径 vs kev(kev) — 结构性对比（本轮只读溯源）

> 目标：回答「kev(bf16, 字节更多) 同机 curl 多次 <500ms，而 kevcpp(Q8_0, 字节更少) ~650ms」。
> 只读，未改任何文件。全部结论来自对两端源码的逐行核对 + 复用已归档实测(docs/PERF_ANALYSIS.md,
> PERF_OPTIMIZATION_SYNTHESIS.md, BF16_BENCH_RESULT.txt)。「代码确认」指直接从源码读出的执行路径；
> 「实测」指引用已归档的探针结果；「推断」指无法在本会话实测、需谨慎的部分。

---

## 0. 直接答案（TL;DR）

kevcpp 那 650ms 不是「这套负载整体的带宽墙」，而是**它自己的执行形态**：每问一次 `llama_decode`、
state 每问都被接着读、recurrent/DeltaNet 路径下 trunk 权重**按 token 流式读取**，所以 506.12 MiB(每 token)
× 实际 token 数 = 总量巨大。kev 却把**所有 question 的 branch 合成一次 batched forward**（`[B,Lb,d]`），
GEMM 里权重只读一次并在整批 token 间复用；它有 950 MiB 的字节/次，但**次数和总 token 摊薄到极少**，
净读字节远小于 kevcpp。

一句话：**kevcpp 用「更少字节/次 × 更多次(每问题/每token)」；kev 用「更多字节/次 × 更少次(合并批)」。**
用户直觉「设备带宽不是唯一瓶颈、还有优化空间」——**成立**。但前提是：kevcpp 逐问/逐 token 的路径自己
就是把「内存墙」和一些固定开销叠了好几遍；这些是可竞争的，不是铁墙。

> 诚实边界：kevcpp 那个「~183 GB/s、达到纯读 74%」的带宽结论（PERF §8）是**在它自己的逐 token 流式假设
> 下**成立的「该路径下限」，不等于「任何做法的带宽墙」。它只能证明 kevcpp 单行 Q8 GEMV 前向是带宽受限，
> 不能证明 kev 同样受限——而实测恰好证明 kev 不是（kev 隐含每字节吞吐低得多，是 GEMM/overhead 形态）。

---

## 1. 一次 curl 请求里 kev 到底做了什么（同口径：state+branch+head）

### 1.0 前提（代码确认）
- 后端：`kev.serve` 在 CPU 上走 `DecisionModel`(torch, eager for hybrid)。`rows_form()` 对 hybrid(Qwen3.5
  Gated DeltaNet) 永远为 True（`model.py:306-310`）→ **一定走 row form**。
- 精度：`serve.py:282` 只在非 CPU 置 bf16；CPU 下 dtype 由 checkpoint `weights_dtype` 决定。本模型为
  bf16 训练（premise，块 trunk=950.1 MiB 已由 bench 测），故 CPU 服务为 bf16（checkpoint.py `_adapted_torch`
  `if weights_dtype=="bf16"` 路径，`checkpoint.py:277-281`）。→ **推断（需核对运行目录 meta），与 premise 一致**。
- graph=kev.cuda_graphs / fused=kev.fused_qwen35 在 CPU 均**不启用**（serve.py:283-285 / fused 只在 cuda；
  fused_qwen35 需 fla CUDA，`fused_available()` 需 fla，文档明说 serving 的 GPU 专用）。→ CPU 上**无图、无 fused**，
  纯 eager 参考层。**代码确认**。

### 1.1 调用链（serve → model，代码确认）
```
POST /v1/systemone → answer_async → submit(rec)
  submit: enc = model.encode(tok, rec, max_state=65536, max_branch=…)   # 打包 state+每问 branch
  → queue → model 线程 _work → _run(encs)  # CPU 上 graphs=None
  _run → self.prefix_cache.plan(encs) → model.probs_batch(encs, cached, keep)
    probs_batch: graphs is None → 每条走 probs_one(self, enc, prefix, keep)
      probs_one: prefix命中 → probs_with_prefix(enc,prefix)   # 只跑 branch
                 未命中 → (keep) probs_and_prefix(enc)         # state一次 + branch，同时产 prefix
                    probs_and_prefix: rows_form真 → self.prefix(enc)[state前向, 存KV+DeltaNet] 
                                        → _branch_rows_from_prefix(enc, cache)
                   (不keep) probs(enc)
    _branch_rows_from_prefix: rows_of(enc) → _rows_hidden(所有branch rows, cache=cache, prefix_len=Ls)
      _rows_hidden: rows_per_pass 切片 → _pad_rows(批量rows) → 一次 self.lm([B,Lb], …)。#★
      → per question: PointerHead / softmax
  _run 末尾: store() 把新 prefix 写回 cache（供下次命中）
```

### 1.2 分环节账（一次**新 state** 请求，Q 个 question，state=Ls, branch 平均 Lb，Q6 选项）
| 环节 | 发生什么（代码确认） | 前向次数 | 读权重字节 |
|---|---|---|---|
| **state** | 每请求**首次冷 miss** 时 `prefix()` 跑一次 state 前向，把 KV+DeltaNet recurrent 存进 PrefixCache。`prefix_min_tokens=0`(hybrid, model.py:265) → 无条件缓存。 | 1 | ~1×trunk |
| **branch** | `_rows_hidden` 把全部 Q 个 branch pad 成 `[B≈Q, Lb]`，**一个 forward 全算**（`rows_per_pass=16384//(Ls+Lb)` 对小 state 极大 → 所有 Q 一次）。state 前缀经**浅复制 cache**(copy.copy + Tensor句柄)，不逐问拷 KV 数据。 | **1** | **1×trunk**（整批权重只读一次，跨 Q·Lb 行 GEMM 复用） |
| **head** | `PointerHead.many` / `_readout_many` 一次算全部 question 的 logits，一个 sync。 | 0 | 忽略(~2MB) |
| **固定开销** | torch eager 图已在内存，**无逐次重构图**；个别 per-question 元素级 op 是 Python/torch 分发。 | – | – |

→ kev 一次冷请求 ≈ **2 次 trunk 读**（state 1 + branch 合并 1）；同 state 后续命中 **≈ 1 次 trunk 读**（只 branch）。
→ 关键：branch 的 trunk **只读一次**，与 Q、与 Lb 里的具体 token 数关系很小（权重在 `[B·Lb, d]@[d,out]`
  GEMM 里跨行复用，L2 里单层权重可缓，不逐 token 重读）。

> 实测锚点（PERF §4.3 / bench）：torch 单 127-token(38+89) row-forward ≈ 400ms；同 state 命中只算 branch ≈
> ~260-340ms 量级 → 与「kev <500ms」吻合（依赖请求 state 命中或 Q 并入单 forward）。

---

## 2. 一次 curl 请求里 kevcpp 到底做了什么（同口径）

### 2.0 前提（代码确认）
- `main_server.cpp:169` POST → `systemone_json`(kev_api.cpp:185) → `model.probs(enc,err)`(kev_model.cpp:206)。
- `probs`: `rows_of` → `ensure_state`(kev_model.cpp:134) → `rows_from_slot`(kev_model.cpp:157)。
- 线程：`init` 自动探测到 32（`kev_model.cpp:48`）。`n_batch=512, n_ubatch=256`（32-59）。

### 2.1 分环节账（同请求，新 state，Q question）
| 环节 | 发生什么（代码确认） | 前向次数 | 读权重字节 |
|---|---|---|---|
| **state** | `ensure_state`: LRU lookup；未命中 `decode_seq(state_ids, 0, slot, {}, pos=∅)`（kev_model.cpp:148）**一次纯 prefill**（n_outputs=0 只存 KV+recurrent）。命中则 0。 | 1(冷)/0(热) | ~1×506.12 MiB |
| **branch** | `rows_from_slot` **逐 question 循环**（kev_model.cpp:167）：每问 `llama_memory_seq_cp(slot→scratch)` + `decode_seq(r.ids, Ls, scratch_seq_, out_pos)`（kev_model.cpp:176，**一次解码整条 branch**）+ `head_.probs`。→ **Q 次独立的 `llama_decode`**，每次形状按该问 branch 长，**逐次重建 ggml 图**。 | **Q** | **Q × 506.12 MiB**（q8 trunk；且 recurrent 下限 + 每次图构建/调度固定成本） |
| **head** | `kev_head.cpp` 逐问 `probs(hd,optp)`（Q 次，标量，~<2ms 总） | 0 | 忽略 |
| **固定开销** | 每次 `llama_decode` ~12-15ms 固定 + 首访建图 ~40ms（cold_profile/perf）。state+branch 两/多形状不同 → `can_reuse` 不满足 → **每形状重构图**（llama-context.cpp，PERF §1.3）。 | – | – |

→ kevcpp 一次冷请求 ≈ **1 + Q 次 trunk 读**；同 state 热（命中 LRU）≈ **Q 次**（仍逐问）。
→ 关键：trunk **每问读一次，且每 token 流式读**（Q8 path：`[1,·]` 或塞进 n_ubatch 的单行/少行，recurrent
  DeltaNet 顺序推进迫使逐 token 流式读权重，测得 506.12 MiB/token，~183 GB/s）。所以总字节 ≈ Q × Lb × 506,
  远大于 1×trunk。

> 实测锚点：m7 (1问6选项, state38+branch89≈127tok) 冷 ≈ 323(state)+290(branch) ≈ **~650ms**；热(命中, 只 branch)
> ≈ **~282ms**；不同英文多问可线性放大（PERF §2.3 / SUBAGENT §1）。

---

## 3. 对比表（逐环节，两大栏）

| 环节 | **kev (torch bf16 eager)** | **kevcpp (llama.cpp Q8_0)** | 差异 |
|---|---|---|---|
| **state 段** | 冷 miss：`prefix()` 1 次 state 前向；命中缓存返 prefix（PrefixCache 浅存 KV/DeltaNet）。无条件缓存(hybrid)。前向=1(冷)/0(热)。读≈1×950MiB。 | 冷 miss：`ensure_state` 1 次纯 prefill(n_outputs=0)；热=LRU 命中 0。读≈1×506MiB。 | 结构同（都是 state 一次）。读字节 kev 大但只 1 次。 |
| **branch 段 前向次数** | **1 次**（所有 Q 问 `_pad_rows` 合并进 `[B,Lb]` 一个 forward） | **Q 次**（每问独立 `decode_seq`/`llama_decode`） | **核心差异**：kev 1 vs cpp Q |
| **branch 读权重字节** | ≈ **1×950 MiB**（权重整批 GEMM 复用） | ≈ **Q × 506 MiB**（每问每 token 流式） | kev 净字节小得多（Q≥2 时） |
| **batch 方式** | 多行 `[B,Lb,d]` batch GEMM，权重跨 Q·Lb 行复用（`_rows_hidden`/`_pad_rows`） | 单问单行 GEMV，`seq_cp`(别名,免费) + 单 row decode，逐问串行 | kev 摊薄、权重复用；cpp 逐问 |
| **state KV 复制** | **浅拷贝** Tensor 句柄(shared_prefix/`_rows_hidden` copy.copy)，不逐问拷 KV 数据 | `llama_memory_seq_cp` recurrent = **O(1) 别名指针**，不 memcpy | 都「免费」，non-issue |
| **head** | `PointerHead.many` 一次全算 + 1 sync | `kev_head` 逐问标量(共~1ms) | 都可忽略 |
| **图构建/固定开销** | eager 图在内存，**无逐次重建** | 每 `llama_decode` 重构图(P.2b 缓解 ≤70ms)；state+branch 形状不同→各自建 | kev 固定开销≈0；cpp 每问 ~12-15ms+ |
| **每 token 带宽形态** | 非带宽受限：`[B·Lb]` GEMM 权重缓存复用，隐含每字节吞吐低(~几 GB/s) | **带宽受限**：绕不过 trunk/每 token 流式，~183 GB/s(74% 峰值) | kev 用「少读」绕过墙；cpp 硬扛墙 |
| **固定读 trunk 次数(冷/热)** | 冷≈2 / 热≈1 | 冷≈1+Q / 热≈Q | kev 恒定小而 cpp 随 Q 线性 |

结论行：**同一次请求，kev 总读字节 ≈ (1 或 2)×950 MiB，与 Q 无关；kevcpp ≈ (1+Q)×506 MiB，随 Q 线性。**
在 m7 单问场景 Q=1 时：kev≈1-2×950MiB vs cpp≈2×506MiB，二者同量级 → 差距主要来自 kev 的 eager「一次 127 tok
forward」vs cpp「两次图构建(38+89)」→ ~400 vs ~650ms。**Q≥2 时差距被放大**：kev 仍 ~2 次读，cpp 达 3+ 次读。

---

## 4. kev 快于 kevcpp 的最可能主因（排序，标注依据）

1. **[代码确认] batch 合并：kev 把所有 question 的 branch 合成一个 forward，trunk 只读一次；
   kevcpp 每问一次独立 `llama_decode`，trunk 每问每 token 流式重读。** → 净读字节 kev 与 Q 无关、cpp 与 Q 线性。
   这是**最大结构性优势**，且是用户「内存墙不该是唯一瓶颈」的正确答案：**墙真存在，但只压在 kevcpp 自己
   的逐问/逐 token 形态上；kev 用 batch GEMM 权重复用把它绕开了。**
2. **[代码确认] 固定图开销：kev(eager) 零逐次重构图；kevcpp 每 `llama_decode` 建图/调度(~12-15ms)，
   且 state 与 branch、各问 branch 形状不同 → 各自重建。** 单问冷请求里这是 ~400 vs ~650 里可分离的部分
   （PERF §7.2 已实测图开销仅 ~70-90ms，主要是带宽差）。
3. **[代码确认] state 前缀缓存深度：kev 的 PrefixCache 命中后整批只算 branch（一次合并 forward）；
   kevcpp LRU 命中也只是「逐问 decode」不再逐问读 state，但每问仍独立 forward。** 二者都做了前缀复用，
   但 kev 的「命中后在合并 forward 上快」更强。kev 在 <500ms 里很可能多数请求是**命中态**（同 state 反复 curl）。
   → 需要诚实：kev 若每次新 state(冷)，也未必 <500ms；用户测得稳定 <500 说明请求大概率复用了 state(=warm)。
4. **[推断] kev 的 batch 还顺带摊掉了 parent/forward 里与 Q 无关的 Python 分发与元素级 op 固定成本**
   （torch eager 虽每字节慢，但总数少、次数少，净墙钟更低）。量化方向未验证，量级不明。
5. **[排除] fused_qwen35 / cuda_graphs 与本机无关**：CPU 上都不启用。不要归因到它们。

（注：首轮「单 token decode 56ms」是新口径，与完整服务链路无关，已作废。）

---

## 5. kevcpp 针对性 1-3 条优化（改哪、预期收益、是否破坏逐位一致）

> 前提约束：任务标的**概率逐位一致**（对拍 golden/parity）。以下按此红线把关。

### 优化 1（推荐，唯一结构性大项）— 多问一次 batch decode（等价 kev 的 `_rows_hidden`）
- **改哪**：`kev_model.cpp::rows_from_slot`（157-199）。把全部 question 的 branch pad 成等长 ubatch，
  用**多个 seq_id 共享同一 state 前缀**（`llama_memory_recurrent` 已支持 `n_seqs>1`、共享 cell；PERF §3 / §5 P.4，
  无需改 vendored 的 mem_hybrid 索引），一次 `llama_decode` 算所有 branch，输出只取各 branch 的 decide/opt。
- **预期收益**：多问下把 trunk 读从「Q×Lb×506」摊到一次「约max(Lb)」的批量读，固定图也摊成 1 次。
  参考 kev 同型：多问可减 ~40-70%（PERF §3/§5 估算）；单问=0。命中 kev 结构优势的关键。
- **是否破坏逐位一致**：**有风险，须对拍**——pad 到等长会改变 chunked DeltaNet 段边界，可能改变非 pad
  输出的逐位值（PERF §5 / F.17 已标注；`SUBAGENT §1` 亦证"多问打包需等长 pad"）。**若业务多问且 RF 允许，可加
  `--batch-rows` 开关，默认关，开时单独对拍；不破坏默认路径。**
- **诚实**：这是回应「还有优化空间」最直接的一条，但**绕不开 recurrent 的一致性红线**，是「值大、但要验证」。

### 优化 2（纯 src，零一致性风险）— 前缀预热做深：把 state+branch 也当「可命中单元」/ 常见 state 预热
- **改哪**：`main_server.cpp --prewarm`（已有，`main_server.cpp:130-150`，P0a 落地）+ `kev_model.cpp`。
- **收益**：让用户反复 curl 的同 state 走 warm（只 branch，~260-340ms），冷→热近减半。**与 kev 的直读差对齐**
  ——kev <500ms 大概率是命中态；kevcpp 若能热，单问也落到 ~282ms。零一致风险（纯 prefill）。
- **是否破坏逐位一致**：否（纯 prefill，与现有 warm 同）。
- **说明**：优化 2 不能弥合「多问」摊薄差（那靠优化1），但它直接回应「kev 快在命中缓存」。

### 优化 3（纯 src，低风险，顺手）— 收窄/复刻 kev 的固定开销与重复工作
- **改哪**：`kev_api.cpp::systemone_json` 已做 once-encode(probs(KevEnc) 重载，落地)；可再确认 `count_tokens` 不再
  重复 tokenize。并顺带清理 `llama_batch_init` 每次新建/释放（`kev_model.cpp:109-122`）与每次 `seq_cp` 的前置
  `seq_rm`（已落地）。量级小(~1-3ms)。
- **预期收益**：每次请求 ~1-3ms 的确定性小项；非主项。
- **是否破坏逐位一致**：否。

（明确**不做**：P.1 state+branch 合成单次 decode —— 已证 recurrent 不逐位一致 |Δp|≈0.0013（PERF 附录B/F.01），
死路；SSE/AVX 已开（F.02 已纠偏）；GGML_LLAMAFILE/REPACK 对 Q8_0 无收益（F.04/F.05）。）

---

## 6. 诚实标注：源码确认 vs 推断

**代码全文确认**（本会话逐行读取）：
- kev：`serve.py`(Server.submit/_work/_run、PrefixCache.plan/store)、`model.py`(encode/rows_of/rows_form/_rows_hidden/
  _pad_rows/rows_per_pass/prefix/probs_and_prefix/probs_with_prefix/_branch_rows_from_prefix/probs_one/probs_batch/
  probs_with_prefix、PointerHead.many、_readout_many)、`checkpoint.py`(_adapted_torch dtype 规则)、`device.py`、
  `shared_prefix.py`(训练共享前缀，C 路径经 `_rows_hidden` 浅拷贝 cache)、`fused_qwen35.py`(CUDA-only)、
  `scripts/serving_bench.py`(请求形态/多问场景)。
- kevcpp：`kev_model.cpp`(ensure_state/rows_from_slot/decode_seq/probs)、`kev_model.h`、`kev_prefix.cpp/h`(LRU 精确匹配)、
  `kev_head.cpp/h`(逐问标量 head)、`kev_api.cpp`(systemone_json once-encode)、`main_server.cpp`(--prewarm/线程/锁)、
  `kev_encode.cpp/h`(encode/rows_of 与 kev 对齐，parity 已证)、`bench_bf16_cpu.py`。
- 「kev branch 合并为单 forward」「kevcpp 逐问 decode」「state 前缀缓存两段语义」「fused/图在 CPU 不启用」：**均代码确认**。

**推断/需谨慎（本会话无法实测）**：
- kev 在 CPU 真正服务的 dtype 是否为 bf16：由 checkpoint `weights_dtype` 决定，本会话 workspace 无运行目录 meta
  可核对；按 premise(kev bf16 trunk=950MiB) 与 bench(以 bf16 载 base) 一致处理。**风险点**：若该权重为 fp32，
  则 kev 字节=1900MiB 而非 950，有关结论的绝对值要×2（但「kev 次数/合并」相对结论不变）。
- 「kev 各请求多数为命中态/热」：由「稳定 <500ms」反推，属推断；无法在本会话起服务复测。
- 「torch GEMM 权重跨批量复用、不逐 token 重读」：由 bench `[c]`(B 增→每 token 摊薄) 与 eager 特性推断，
  未在本会话 instrument。
- kevcpp 的带宽数字(183 GB/s、506 MiB/token)：引用先前已归档正确结论，非本会话新测。
- 优化 1(pad 批) 的逐位影响：推断存在风险，未实测对拍。

**未美化任何一侧**：kev 的快，确实依赖「多问合并 + 前缀命中 + eager 少图」三把刀；若每次都是全新长 state 且 Q 少，
kev 的冷请求也会接近甚至超过 kevcpp 的冷成本区（读 950MiB×2）。kevcpp 的 wall 也在很大程度上是「逐问逐 token
形态」自己堆出来的，并非不可回避的铁墙。