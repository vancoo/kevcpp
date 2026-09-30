# kevcpp 性能分析报告（冷启动到 torch 目标）

> 分析对象：`E:\van\projects\kevcpp`（master，纯 C++ CPU 版 Kev 决策服务），复用 vendored
> `third_party/llama.cpp`，`Qwen3.5-0.8B-Q8_0.gguf` + `kev_head.cpp`(PointerHead)。
> 目标请求：`tests/golden/req_m7.json`（1 choice 问、6 选项）。**本报告为只读分析，不改任何代码。**

---

## 0. 结论速览（TL;DR）

- **现状分段**（threads=32, cold）：`state prefill 38 tok ≈ 323-385ms`、`branch decode 89 tok ≈ 290-305ms`，
  HTTP/cold 总 ~676ms（`m7_bench` ~650ms、`m7_after_thr` ~655ms）。
- **最大耗时源**：① **每条 `llama_decode` 的固定图构建/调度开销**（cold 约 200-260ms 是"不可回收字节量"之外的纯固定成本），
  ② 前向实算的**内存带宽**（Q8_0 0.8B 读权重的硬件下限 ~360-400ms）。两者叠加 ≈ 676ms。
  ⚠️ 第三个常被误判的因素——`llama_memory_seq_cp` 在 recurrent（DeltaNet）后端是 **O(1) 指针别名**，**不做整块 KV 拷贝**，因此"行形式复用状态"在本模型上几乎免费。
- **与 torch 对齐判定**：**把 cold 676ms 压到 ~400ms 是可行且现实可达的**。因为 400ms 恰好 ≈ 该负载的
  "内存带宽下界"（~127 token × 有效权重字节）。路径就是**砍掉 ~250ms 固定开销 + 把 state+branch 合并为单条 decode**，
  把两次图构建收敛为一次。见 §4。
- 最该先做的两件事（零上游改动风险都尽量低的优先）：
  1. **将 state+branch 合并为单次 127-token decode**（老 back 前向一次算完，省掉第二次图构建/调度）——**不碰 vendored llama.cpp 也能做一部分**（见 §4 P.1）。
  2. **预热水表/固定 `n_batch`** 让图只建一次并复用（改 vendored 或驱动参数）——P.2。

---

## 1. 现状分段瓶颈分析（有日志与源码行号背书）

### 1.1 复现的分段计时（`tests/golden/m7_*.log`，threads=32，state=38 tok / branch=89 tok / 1 问 6 选项）

| 日志 | n_ctx | state(38tok) | rows(branch 89tok) | cold 总 | warm 总 |
|---|---|---|---|---|---|
| `m7_baseline.log` (0-thr, 旧基线) | 8192 | 494.0 | 904.9 | **1399.4** | 900.8 |
| `m7_after_thr.log` | 8192 | 350.8 | 304.5 | **655.4** | 282.5 |
| `m7_t32.log` | 8192 | 323.0 | 289.6 | **612.7** | – |
| `m7_ctx2048.log` | 2048 | 384.4 | 307.5 | **692.0** | – |
| `m7_ctx1024.log` | 1024 | 345.1 | 304.6 | **649.8** | – |

线程伸缩（ctx8192）：rows = t1 2499.7 → t4 746.1 → t8 477.6 → t16 355.0 → t32 289.6；
state = t1 1275.1 → t4 397.0 → t8 276.1 → t16 325.2 → t32 323.0。

**n_ctx 证据链**：ctx1024/2048/8192 三项 state 与 rows 几乎相同（~310/300 与 ~350/305），
说明解码耗时**不受 `n_ctx`（8k→2k→1k）影响**；真正决定单次 decode 规模的是 `n_batch/n_ubatch`（都=512）与每次喂的 token 数。
`kev_model.cpp:53-55` 把 `n_batch=n_ubatch=512`，`llama-context` 在 init 时 `graph_reserve(n_tokens=512)`（`m7_after_thr.log:1581`）。

### 1.2 每 decode 的耗时成分拆解

关键反直觉点：**state(38 tok) 反而比 branch(89 tok) 慢**（323 vs 289）。若纯按 token 线性，89tok 应是 38tok 的 ~2.3×。
这说明两段各含一个**与 token 数无关的固定开销**，而**第一次 decode（state prefill）额外吃着"建图+调度"的一次性成本**。

带宽下界估算（0.8B Q8_0，`m7_after_thr.log:52` file_size=763.78MiB，8.52 BPW）：

- 前向每 token 需读的有效权重 ≈ 总权重字节 − `token_embd`(GET_ROWS 是索引取行，非整行 matmul)。
  `token_embd` = 248320×1024×~1.06B ≈ 270MB（`m7_after_thr.log:183/186` tied，仅 gather）。
  剩余 ~24 层 attn+DeltaNet+FFN 权重 ≈ 763.78 − 270 ≈ **~490MB/token**（含 scale 等小 tensor，实际 matmul 主体重 ~370-400MB）。
- 127 token（38+89）→ 带宽读取 ≈ 127 × ~390MB ≈ **~50GB**。
- 观测到的 branch-字节分量 ≈ 289ms−固定 ≈ 240-260ms → 有效读带宽 ~**130-160 GB/s**（DDR 多通道合理）。

由此反推各段成分：

```
state prefill  323ms ≈ 固定(建图+调度, 首次) ~200ms  +  字节(38tok) ~120ms
branch decode  289ms ≈ 固定(二次建图, 形状不同) ~50-90ms +  字节(89tok) ~200-240ms
head/softmax           ~1-2ms（dp=256×6，kev_head.cpp:39-55，可忽略）
-------------------------------------------------------------------------------
cold 总             ≈ 固定 ~250-280ms + 字节 ~360-400ms = 612-676ms ✓（与日志吻合）
```

**结论**：fixed/不可回收字节占比约 **38-42%**，这是当前优化空间的主矿。前向实算是带宽下界，占 ~58%。

### 1.3 源码层面的固定开销来源（`src/kev_model.cpp`）

- `probs()` (`kev_model.cpp:193-205`) 顺序调用：`ensure_state()`（38tok prefill, 首次建图）→ `rows_from_slot()`（89tok decode, 形状不同 → 再次重建图）。
- `decode_seq()` (`kev_model.cpp:97-121`) 每次 `llama_decode(ctx, batch)`。llama.cpp `process_ubatch`（`llama-context.cpp:1393-1466`）在 `gf_res_prev_active==res && res->can_reuse(gparams)` 不成立时执行 `res->reset(); ggml_backend_sched_reset; model.build_graph(gparams); ggml_backend_sched_alloc_graph`。
  - `graph_params`（`llama-context.cpp:2543-2565`）含整个 `.ubatch`，因此 38 与 89 token 的两次 decode **形状不同 → can_reuse=false → 各自重建图**。
- **A2 已把输出降到最小**：`kev_model.cpp:57` `n_outputs_max_per_seq=128`，`decode_seq` 只对 decide/opt 位置设 `batch.logits=1`（`kev_model.cpp:113-115`）。故本段**输出端无浪费**，非瓶颈（`m7_t32.log` 图里 `out_ids` GET_ROWS 只取请求位）。

### 1.4 `llama_memory_seq_cp`：在 recurrent 后端其实是"免费"的

`rows_from_slot()` 调 `llama_memory_seq_cp(mem, slot, scratch_seq_, 0, -1)`（`kev_model.cpp:165`）看似"复制状态前缀"。
日志 `kv_unified=false`（`m7_after_thr.log:912`）+ hybrid DeltaNet → 走 `llama_memory_recurrent`：
- `seq_cp`（`third_party/llama.cpp/src/llama-memory-recurrent.cpp:249-284`）只做 `tail_dst.tail = tail_src.tail` + 往共享 cell 的 seq_id 集合插入 dst——**不 memcpy 任何 KV/state 字节**。
- 因此**"复制"几乎不耗时**。真正值得担心的"把 38 个 state 的 KV/DeltaNet state 复制给每个 branch"在本模型里是假的（别名共享）。
  - 唯一副作用是因果：branch 在共享 cell 后**追加**新 cell，不改 state 源 slot → slot 可跨请求复用（LRU 语义成立，`kev_prefix.cpp`）。

> 顺带说明：这并不等于"完全零成本"。共享 cell 使 state 的 KV/state 会随 branch decode 被"读到"（attention 需读 state 的 K/V 与 DeltaNet 的 recurrent state），但那是**必然的字节读取**（属于 §1.2 的字节分量），不是额外 memcpy 复制。

### 1.5 各优化在"这个 0.8B CPU"上的实际可回收量级

| 项 | 可回收量级判断 | 依据 |
|---|---|---|
| 多线程（已完成 A3） | 已回收 ~660ms（1399→676，threads 0→32） | baseline vs after_thr |
| 免去每次请求的重复图构建 | **潜在 ~120-250ms** | `build_graph`+`sched_alloc_graph`（§1.2 固定部分） |
| state+branch 合并单次 decode | **潜在 ~90-200ms**（省第二次固定+调度） | §1.3 |
| 减小 `n_batch/n_ubatch` 512→128/256 | 潜在 ~30-80ms（缩小预留图/工作区与调度） | `graph_reserve` 512 |
| 端侧/解码器双 tokenize | ~0-2ms | `kev_api`/`kev_model` 每次 encode 多次 tokenize，量很小 |
| 量化/换模型 | 无此需求（目标不改模型） | – |
| `llama_memory_seq_cp` 优化 | **≈0**（recurrent 已别名） | §1.4 |
| head 计算 | ≈0（<2ms） | §1.2 |

---

## 2. 与 Python 原版 `kev/model.py` 的实现/思路对比

### 2.1 计算路径/公式一致性（对拍结论：**一致**）

| 环节 | kev/model.py | kevcpp | 一致? |
|---|---|---|---|
| encode/分词 | `encode()` L82-119、`user_tokens()` L73-76 | `kev_encode.cpp:41-117`、`kev_encode.cpp:34-39` | ✅ 同为 `<\|x\|>→<¦x¦>` 重写 + `add_special_tokens=False` |
| rows 拆分 | `rows_of()` L165-179 | `kev_encode.cpp:119-144` | ✅ 同 `<decide>` 末位、`</opt>` 偏移 |
| 隐藏向量 | `AutoModelForCausalLM(...).model` 读 `last_hidden_state` L234 | `embeddings=true, pooling=NONE` 读 `t_embd`（output_norm 后） | ✅ 均**跳过 LM head**（torch 用 `.model`；llama.cpp 用 embeddings 路径） |
| PointerHead | L182-195：`z=(k(h_opts) @ q(h_decide))*scale` | `kev_head.cpp:25-37`：`apply(k)+apply(q)` 点积 × scale | ✅ 公式同（`kev_head.h:11-14`） |
| scale | `1/sqrt(dp)`，dp=256 → 1/16 | `kev_head.cpp:98` `1/sqrt(256)` | ✅ |
| 温度 | eval 时 `z/temperature`（L195） | `probs()` 内 `invT=1/T` 再做 softmax `kev_head.cpp:43-54` | ✅ head.bin 温度 2.351 |
| softmax | `F.softmax(z,-1)` | 数值稳定 softmax | ✅ |

结论：**计算路径逐位一致**（已有 `parity4.log`/m5_* 对拍支撑）。不存在公式层面的差异可省。

### 2.2 关键思路差异：#1 —— Python 复用"整行一个 forward"，C++ 拆成两次 decode

- **torch 侧（`forward_rows_batch` L341-360 / `_rows_hidden` L312-339）**：单个 1 问请求在 row form 下是一条 `(state 38 + branch 89 = 127)` token 的 **causal row**，`chunk = len(rows) = 1`，**一次 forward 全算完**。torch 的 eager 图已静态存在，无"每次重构图"的固定开销。`probs_with_prefix`/`prefix`（L394-439）缓存 state prefix，命中时只算 branch。
- **C++ 侧（`kev_model.cpp:200-202`）**：先 `ensure_state`（38tok decode）再 `rows_from_slot`（每 branch 一次 decode），**同一请求内产生了两次形状不同、各自重建 ggml 图的 `llama_decode`**。
- **量化差异（对本 1 问请求）**：torch 冷启动 ≈ 一次 127tok forward + 轻微图已存在；C++ 冷启动 ≈ 两次（38 + 89）llama_decode + 两次图构建。**这就是 C++ cold ~650ms vs torch ~400ms 的主要来源之一**（差距 ~250ms 固定开销，§1.5）。
  - 但注意：C++ 之所以拆，是为了把 state 单独 cache 供**跨请求复用**（LRU slot）。单问请求里这个复用用不上却付出了两次图构建。**多问请求时**每次 requests 仍需独立（state 相同才命中），见 §3.6。

### 2.3 关键思路差异：#2 —— Python 的 batch/一次算多条 row；C++ 一条条 decode

- torch `_rows_hidden`（L312-339）把多个 question/多个请求的 rows 按 `rows_per_pass`（默认 16384 token 预算）**pad 成 `[N, Lmax, d]` 一批 forward**（`_pad_rows` L280-291）。多问/多请求时一次 forward 算多条 row → 权重读一次、多条共享，**摊薄带宽与固定开销**。
- C++ `rows_from_slot`（`kev_model.cpp:153-191`）**每个 question 单独一次 `decode_seq`**。1 问 6 选项时只有 1 次（89tok），无浪费；但**多问（如 5/10 问）时 = 每问一次 89tok decode，状态前缀被每个 branch 重复读到，且每次都重建/调度图**。这正是 torch hybrid 侧注释（`model.py` L261-264）说"否则每个问重新算 state"的原因。
  - 对单问请求：多问 batch 差异**不适用**；对多问请求：C++ 一次只喂一个 branch 的 89 tok，而 torch 会把 N 个 branch（share 同一 state prefix）打包进一次 forward，权重读 N 次但固定开销摊成 1/N。**这是多问场景 C++ 的明确损失**，且随 N 线性放大固定开销。

### 2.4 Python 的 prefix 复用 vs C++ 的 LRU（都实现了，语义对齐）

- torch：`prefix()`（L394-403）把 state 前向存 KV + DeltaNet state +（attention-only 才存）state hidden；`probs_with_prefix`（L424-439）命中只跑 branch，用完 `cache.crop` 复位。hybrid `prefix_min_tokens=0`（L265）。
- C++：`PrefixLRU`（`kev_prefix.cpp`）+ `ensure_state`（`kev_model.cpp:130-151`），同 state 命中 `state=0.0ms`（`m7_after_thr.log:1624` warm）。
- **等价**：warm 只算 branch，`~282ms` 与 torch"命中只跑 branch"同一思路。✅
  - 唯一盲点：**C++ 的 LRU 只缓存"完全相等"的 state_ids**（`kev_prefix.cpp:17` 精确匹配），torch 也一样（前缀相等才命中）。都没有"最长公共前缀"增量缓存——两者都不需要为此改造（差异为 0，非优化点）。

### 2.5 有无"可移植"的 torch 优化思路尚未在 C++ 落地？

- ✅ 已落：跳过 LM head、前缀缓存、只取 decide/opt 隐藏、多线程、降 n_ctx。
- ❌ 未落 / 落不彻底：**"一次 forward 算完整 row"（state+branch 合并）** 与 **"多 branch batch"**（diff #1/#2）。这是 C++ 相对 torch 的**结构性差异**，也是 Table §1.5 里主要可回收项。
- torch 的 LoRA 融合（peft `get_peft_model` L252，LoraConfig target=all 投影）——C++ 用 `llama_adapter_lora_init` 原生 LoRA（`kev_model.cpp:66-73`），**融合正确**（有 `m5_lora*.log` 对拍），无额外优化空间。

---

## 3. 未来可选优化清单（按性价比排序）

> 标注：🔧=只改 `src/`（无上游），🟦=需改 vendored `third_party/llama.cpp`（可打补丁，见风险），⛔=不能动上游。
> "单问"="1 问请求"（本基准），"多问"="N 问请求"。

### 🔧 P.1 状态+单分支合并为一次 decode（**首选，单问收益最大**）
- **改哪**：`kev_model.cpp::probs()/ensure_state/rows_from_slot`。
  冷 miss 时，若只有 1 个 question，直接把 `state_ids + row.ids`（127 tok）作为一条因果 row 喂**一次** `decode_seq`（position 从 0 连续），再从同一 output 取 decide/opts 的 hidden。
- **预期收益**：省掉第二次图构建+调度。单问 cold 预估 676 → **~450-500ms**（§1.5 的 ~90-200ms）。多问次之。
- **前提/风险**：仅当该请求的 state 命中**且只服务此一个 question** 才能省 state slot 的独立缓存。若还要给后续请求留 state 缓存，可"先 packed 算完出答案，再额外把 state 单独 prefill 进 LRU"（双花但首次仍省）；或为单问做"packed 快路径"。需保持 `rows_of` 的 decide/opts 绝对偏移换算。**不碰 llama.cpp**；形态等价已由 torch row form 证明。
- **零上游改动，风险低**，是达成 400ms 的**主干**。

### 🟦 P.2 图只建一次并跨请求复用 / 预热
- **改哪**：让 llama.cpp 复用同一形状的图（冷启动后在 init 或首个请求预热一次）。触发方式二选一：
  - (a) 驱动层：若每个请求的两段形状固定（state 38 / branch 89），**预取同样 38→89 形状的 ubatch 建好图**，后续 can_reuse 命中。但不同请求 state 长度会变 → 形状漂移仍需重建。**根治须改 llama.cpp 的 graph cache 策略**（⛔ 若依赖 `E:\van\projects\llama.cpp`）。
  - (b) **改 vendored**：`llama-context.cpp` 引入"按 `(n_tokens_ceil, n_outputs)` 分桶持久缓存图 + 复写 batched 输入"，避免每次 `build_graph`。这是 llama.cpp 的合理演进方向，但改动较深。
- **预期收益**：单问 cold 再降 ~120-250ms；多问/多请求都有益（跨请求，只要形状稳定）。
- **风险**：🟦 改 vendored 副本要长期维护 .patch（vendored 打补丁风险见 §3.6）。收益大、代价中。

### 🟦 P.3 把 `n_batch/n_ubatch` 从 512 收到贴近真实行宽（128/256）
- **改哪**：`kev_model.cpp:54-55`（这是 src，🔧 即可）改成按最大 row（state max + branch max，如 384+1024 的容量上限会回来；但**默认实时行宽 127** 远小于 512）。
  - 注意：`decode_seq` 每次 `llama_batch_init(n, ...)`（`kev_model.cpp:105`）只申请当前 n，`n_batch` 是**容量**。把它降下来使 `graph_reserve` 的投影/工作区（当前 ~491MB，`m7_t32.log:1616`）缩小，`build_graph` 节点数、`sched_alloc_graph` 更轻。
- **预期收益**：预估 ~30-80ms（固定图构建/调度随预留减）。
- **风险**：即便允许上下文大，N 过大时 `n_batch` 需 ≥ 单次 ubatch 的 token 数；若一次要喂 state+branch 128+则 128 可能不够，设 256 更安全。**低风险**。

### 🟦 P.4 多 branch（多问）一次 batch decode — 多问收益大、单问无收益
- **改哪**：`kev_model.cpp::rows_from_slot`——把所有 question 的 branch 用一个 `llama_batch`（不同 seq_id，共享 state slot/alias）一次性喂入，`llama_decode` 一次算所有 branch，输出只取各 branch 的 decide/opts。
- **预期收益**：多问请求把固定开销摊成 1/N、权重/state 读取摊薄（torch `_rows_hidden` 批处理思路）。**单问=0**；多问如 10 问可省 ~40-70%。
- **前提/风险**：需多 seq 共享同一前缀（recurrent 内存的 cell 别名天然支持多 seq 共享前缀，见 §1.4），但 llama.cpp 的 `llama_memory_recurrent::seq_cp` 别名后**多 seq 追加的 tail 独立** → 一次多 seq decode 在 recurrent 后端是否允许共享 cell 前向需要验证（固有问题：DeltaNet 每 seq 独立 state，多 seq 并发 forward 的 `set_inputs` 需按 seq 索引 separate state）。**这是真正要动 vendored 的风险点**（改 graph input 的 mem_hybrid 索引）。低于 P.2 收益确定性。

### 🔧 P.5 端侧/解码器消除重复 tokenize / 预热
- **改哪**：`kev_api.cpp::systemone_json` 对答案 `count_tokens(answers_str)`（`kev_api.cpp:203`）做一次额外的 GGUF 分词；`encode` 内对 state/instr/每个 option 各 tokenize 一次（`kev_encode.cpp:45,71,74`）。
- **预期收益**：合计 ~1-3ms，**可忽略**。仅作备注，不值得为它动结构。

### 🟦 P.6 mmap 已启用、无需再动
- 日志 `load_mode = mmap`（`m7_after_thr.log:157`），模型只映射不拷贝。此项已达最优。若内存压力大可 `no_mmap` 折衷，但会变慢，非优化。

### 🔧 P.7 LoRA 融合验证（结论：已正确，无需改）
- `llama_adapter_lora_init` + `llama_set_adapters_lora`（`kev_model.cpp:66-73`），`m5_lora*.log` 对拍一致。LoRA 权重在 `build_lora_mm` 内融合进 matmul（`llama-graph.cpp:1515`）。运行期 LoRA 额外读权重很小（r=16），非瓶颈。

### 👎 已排除 / 不建议
- 🧊 LLM head / 层剪枝：`docs/LAYER_PRUNING_ANALYSIS.md` 已结论"不剪"（LM head 已跳过、24 层剪=精度换速不划算）。
- 🧊 `llama_memory_seq_cp` 优化：recurrent 已 O(1) 别名，无可优化（§1.4）。
- 🧊 head/softmax：<2ms，忽略。

### 🟦 / ⛔ 上游区分小结
- **必改 vendored `third_party/llama.cpp`**：P.2(b)（图缓存策略）、P.4（多 seq 共享 recurrent 前向的真正支撑）。
- **可只改 `src/` 实现（不碰 vendored）**：P.1（合并为单行 decode）、P.3（降 n_batch，src 参数）、P.5、P.7。
- **不可动 `E:\van\projects\llama.cpp` 上游**（外部仓库）；vendored 副本理论上可打补丁，但需：
  - 每次 upstream 合入都要 rebase .patch；
  - 与官方行为分叉后的长期维护成本；
  - 只做**向后兼容、增量**的改动（不开新后端、不改公共 tensor 布局），把风险控制在 llama_decode 内部。

---

## 4. 目标对齐：纯 CPU、不改模型/量化，能否从 ~676ms → ~400ms？

### 4.1 判断：**可行，且是最现实的收敛路径。** 依据：
- 400ms ≈ 该负载的"内存带宽下界"（§1.2：~127 token × ~390MB ≈ 50GB ÷ ~130GB/s ≈ ~360-400ms）。
- 现 cold 676ms − 下界 ~400ms ≈ **~276ms 是固定/二次图开销**，属于"可以移除的、非硬件下界的部分"。
- 因此**只要把固定开销几乎清零，就停在 ~400-450ms**——恰好落到 torch 目标的同一数量级（torch 也受同一带宽下界约束，只是几乎无图重建固定开销）。

### 4.2 最可能成功的 1-2 条路径（建议按序推进）

**路径 A（首选，组合：P.1 + P.3，只改 src）**
1. **P.1**：单问冷请求把 `state+branch` 合成**一次** 127-token `llama_decode`（省第二次图构建）；多问再切回分段或 P.4。
2. **P.3**：`n_batch/n_ubatch` 512 → 256（贴近实时行宽，缩预留图与调度）。
- 预期：676 → **~430-470ms**（再叠加 warm/命中保持现有 282ms 不变）。
- 风险：低，全程不碰 vendored；只需把 `rows_of` 偏移换算对、及"packed 出答案后再补 state slot 缓存"的取舍。

**路径 B（锦上添花,🟦 但要动 vendored：P.2(b) 图持久缓存）**
- 在 `third_party/llama.cpp` 加"按 (n_tokens_n=512 整块 / n_outputs 桶) 持久缓存图并复用"，跨请求免重建。
- 预期：cold 再砍 ~120-250ms → **~350-410ms**（接近/略优于 torch）。
- 风险：中（需 patch 维护），但这是把精确命中 torch 的最后一块。若接受"~450ms 已达标"，可推迟。

**路径 C（多问者启用，🟦：P.4 多 seq batch）** —— 用于业务为多为的场景，单问不适用。

### 4.3 预期的最终形态
- 单问 cold：约 **~400-450ms**（路径 A 即可达 ~430，加 B 到 ~400）。
- warm（同 state 命中）：保持 ~282ms。
- 多问：走 P.4 后从"每问一次 decode"变为"每问一批"，按 N 摊薄固定开销（严重多问时接近"一次 127×N token 大前向"）。

---

## 附录：一句话给 owner
- 现在的"慢"**不在** `llama_memory_seq_cp`（recurrent 下免费），**也不在**词法/head，**而在**"两次各自重建 ggml 图 + 首访建图"的 ~250ms 固定开销（§1.2）。
- 先做 **P.1（state+branch 合成一次 decode）+ P.3（降 n_batch）**，纯 src、无 vendored，走到 ~430ms；确实要贴住 400ms 再加 **P.2(b)（图持久复用）**。多问再考虑 **P.4**。这些都无需改模型/量化。

---

## 附录 B：实测验证结论（分支 perf-single-decode）

> 本节报告对「路径 A（P.1+P.3）」的**实际实测结果**。§4 的乐观估计（676→~430ms）基于"~250ms 固定图开销可移除"，但对 **Gated DeltaNet / recurrent** 模型不成立。实测分两个独立发现：

**① 测量环境注意（重要）**：本机当时负载不稳定（多 chrome 进程占用 CPU），解码耗时在所测期间从"空闲 ~668ms"漂到"加载 ~760-820ms"。因此**跨时间点比较（尤其拿空闲的 baseline 对比加载期的优化后）会得出错误"倒退/提升"结论**。以下结论全部在**同一负载窗口内**做 A/B。

| 变体 | n_batch | cold 总（同负载窗口 best-of-3） | compute buffer | probs 一致性 | 判定 |
|---|---|---|---|---|---|
| 现状（A1+A2+A3） | 512 | ~769ms | 491MB | golden 逐位一致 | 基线 |
| **P.1 打包**（state+branch 单次 decode） | 256 | 802ms(单次 rows) | – | **不一致（\|dp\|≈0.0013）** | ❌ 正确性否决 |
| **P.3 降 n_ubatch**（512→**256**，n_batch 保持 512） | 512 | ~766ms | **245MB（减半）** | golden 逐位一致 | ✅ 保留（省内存、latency 中性） |

- **P.1（打包，被否决——正确性硬伤）**：把 state(38)+branch(89) 合成一条 causal row 单次 `llama_decode`（位置 0..126）。
  - 实测单次 127-token decode ≈ 802ms（并行时间点，仅作规模参考）。
  - **硬伤是概率数值偏差**（`0.0129 0.6807 0.0688 0.0258 0.2043 0.0076` vs golden `0.0125 0.6794 0.0697 0.0265 0.2043 0.0075`，|Δp|≈0.0013）。Gated DeltaNet 的 recurrent state 在"一次整行前向"与"截断(38)+续接(89)两次 decode"之间**不逐位一致**，不满足任务"概率逐位一致"的硬约束。**recurrent 后端不可行**（attention 模型也许可，本案不适用）。
- **P.3（n_ubatch 512→256，n_batch 保持 512，保留）**：compute buffer 491→**245MB（省一半内存，确定收益）**，且 n_batch=512 容量上限不变（state<=384/长 branch 通用场景不缩水）；同负载窗口内 best-of-3 cold **~766ms vs 512 的 ~769ms，latency 中性（无倒退）**，probs 逐位一致。→ 作为**纯内存优化**提交保留（无性能损耗、正确性不变）。
- **结论**：① P.1 在 recurrent 模型上因数值一致性被否决；② P.3 作为内存优化保留；③ 当前 A1+A2+A3+**P.3(n_ubatch=256)** 双 decode+LRU 形态，对单问冷请求接近该 CPU 该模型的实用下限（空闲 ~660ms，≈内存带宽下界 + 一次首访建图；负载下相应变慢属环境噪声）。报告 §4 的"砍 ~250ms 固定开销"在 recurrent 模型上未兑现——多出的开销是 recurrent 前向/带宽本征成本，非可移除的图构建固定成本。
- **若要继续压**：唯一高确定性杠杆是 **P.2(b) 图持久缓存（需改 vendored third_party/llama.cpp）**，且必须**保住两次 decode 形态**（recurrent 数值一致性依赖），只优化"同一形状 decode 免重建图"。多问另见 P.4。