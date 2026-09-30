# 代码对比增量报告（kevcpp 只读）

> 与 `docs/PERF_ANALYSIS.md` 已覆盖项（P.1–P.7、§1–§4）**区分开**，本报告只记新增。
> 所有标"新增"点均为本子代理在源码/黄金样例上**新核实**的观察。

---

## 1. 多问请求：是否逐问独立 decode？【新增确认】

- **现状**：`probs()`（`kev_model.cpp:197-209`）→ `rows_from_slot()`（`kev_model.cpp:157-195`）外层 `for (const auto & r : rows.rows)`，**每个 question 一个 `KevRow`**。每次迭代独立执行：
  `llama_memory_seq_cp(slot→scratch)`（`kev_model.cpp:169`）→ `decode_seq(r.ids, Ls, scratch_seq_)`（`kev_model.cpp:176`，内部 `kev_model.cpp:121 llama_decode`）→ `head_.probs`（`kev_model.cpp:188`）。
  → **逐问一次独立 decode，确认**。N 问 = N 次 `llama_decode`，每次图形状由该问 branch 长度决定。
- **能否打包多问到一次 decode**：可，但需多个不同 `seq_id` 共享同一 state 前缀同时前向（一次 `llama_decode` 内多 seq）。问题：所有 branch 目前**共用 `scratch_seq_` 这一个 seq**（`kev_model.cpp:44`），要并行则须各自 seq；且 Gated DeltaNet 需按 seq 索引独立 recurrent state（PERF §3 P.4 已列为 🟦）。这里确认**改 vendored 是硬前提**（还款 heap 后为🟦）。注意：不同问 branch 长度不同→图形状不同，即使打包也只是"一次调度多个分支"，固定图开销仍按其最大 n 一次摊薄。
- **判断**：多问到一次 decode = 🟦（需改 vendored 的多 seq recurrent 前向）。

## 2. HTTP 并发【新增】

- **现状**：`main_server.cpp:127` `httplib::Server svr`，各 handler 直接捕获共享 `kev::KevModel model`（`main_server.cpp:114`），**无互斥锁**、不设置 `new_task_queue`。
  → **纠偏（管理会话核验）**：cpp-httplib 默认 `new_task_queue` 实际是 **ThreadPool**（`httplib.cpp:8136-8140` 默认 `ThreadPool(max(8, hardware_concurrency-1))`，`httplib.h:183-187`），并非单线程同步 accept。因此并发请求会进入多线程处理，但因共享的 `KevModel` 无锁，实为**竞态**（互相覆盖 KV/scratch/LRU），而非"安全串行"。**body 解析/序列化可优化点**（次量级）：
  - 每请求都 `json::parse`（`main_server.cpp:152`）+ `res.set_content(out.dump())`（`main_server.cpp:164`）——nlohmann dump 重复字符串化，量小可忽略。
  - `run_test_api` `packed_equals_separate` 已证明语义等价，但打包/拆包重复 `out["usage"]`、`answers.dump()`（`kev_api.cpp:203`）多次字符串化。
- **判断**：并发压测收益目前**为 0（本就串行）**；若要并发，须配合 online 锁 + 只读 KevModel 复制（重 init 一次）。规划时先确认业务是否多并发。真改 = 🔧（src 加线程池注册）+ 🟦（共享 ctx 线程安全需谨慎）。

## 3. head：fp32 成本与多 branch 向量化【新增核实】

- **现状**：`kev_head.h:29` `Linear::W/std::vector<float>`、`kev_head.cpp:12-23` `apply()` 为**标量双重循环**；权重从 .bin **按 fp32 读入**（`kev_head.cpp:93-96`）。**全程无 fp16**。
  → 问题问"是否 half"的答案是：**本来就是 fp32**，torch 的 `nn.Linear` 亦 fp32，**二者已对齐，无任何 half→fp32 换算浪费**。"改 fp32 无损"在此无意义（已是 fp32）。
- **多 branch 向量化**：`apply` 对标量无 SIMD（`kev_head.cpp:14-21`）；`q` 对 h_decide、`k` 对每个 opt 各自 `apply`。可预投影 `k(h_decide)`（每问一次）+ 对全部 branch 一次性算 `k(W_k^T h_opt)` 合并——但 head 规模仅 2×(256×1024)≈2MB，耗时**<2ms（PERF §1.5 已证）**，向量化收益可忽略。**📢 注意**：PERF 说 head 约 1-2ms，本子代理实测从源码看其实远小于 1ms（dp=256 标量点积 6 次 ≈ 微秒级），**该 1-2ms 也偏保守**。
- **判断**：🔧 可改（SIMD/批处理）但**不值得**，性价比≈0。

## 4. 编码/解码器冗余【新增关键发现：不止 P.5】

- **现状（比 PERF P.5 更大的隐藏量）**：`systemone_json`（`kev_api.cpp:190`）先 `model.encode(ps.rec)`，随后 `model.probs(ps.rec)`（`kev_api.cpp:194`）内部**再次调用 `encode(req)` + `rows_of`**（`kev_model.cpp:200-202`）。
  → **同一请求的 state+全部 instr/option 被 tokenize + `<|x|>→<¦x¦>` 正则重写两遍**，且 `kev_api.cpp:203` 又对 `answers.dump()` 做第三次 tokenize（`count_tokens`）。
- **量级判断**：state 38 tok / branch 89 tok，正则重写+2…3 次分词合计仍远 <1ms（PERF 说 1-3ms 属实）。**但结构性冗余明确、零风险可消**：`systemone_json` 应复用第一次 `encode` 结果，或直接复用一个"encode 一次 + probs 收 KevEnc"的 API。这是**纯 src、无上游**的确定性小优化。
- **判断**：🔧 可改，收益 ~1-3ms（确认 P.5 量级，非更大）。

## 5. LRU / 前缀缓存【新增确认 + 边界】

- **现状**：只精确匹配（`kev_prefix.cpp:17` `e.ids == ids`），**无 LCP 增量**。跨请求命中**已实测生效**（M3 README pass2/pass4：state 210ms→0ms）。
- **LCP 机会**：对**共享前 k 个 token、后段不同**的两个 state，当前都未命中（各自全 prefill）。理论上可让较长 state 复用短 state slot 中**位置对齐的共享 cell**（llama_memory 支持同 cell 挂多 seq_id + recurrent O(1) 别名，见 §1.4）。但 Gated DeltaNet 的 recurrent state 是**顺序推进**的，截断共享前缀后是否仍能只补差异段、与"整体重 prefill"逐位一致**未验证且有数值一致性风险**（PERF 附录 B 已证明"拆两次 vs 整行"在 recurrent 上不逐位一致）。→ **值低，风险高**。
- **判断**：🟦（依赖 recurrent 共享 cell 前向语义），但不建议优先。

## 6. 正确性 / 内存风险【新增，只报不改】

1. **`PrefixLRU::register_` 冗余/潜在重复**（`kev_prefix.cpp:36-39`)：`e.ids=ids, e.last_use=++clock_, e.ids=ids` 双重赋值；若 entries_ 恰有相同 seq（当前被 `lookup` erase 后不可达），会**再 push 一条重复项**→ 后续 evict/查 find 紊乱。**当前不可达**（latent），建议清理重复小于隐蔽 bug。
2. **scratch 复用不 `seq_rm`**：`rows_from_slot` 每 branch 只 `seq_cp` 重指尾（`kev_model.cpp:169`），**从不 `llama_memory_seq_rm(scratch,0,-1)`**。长分支之后残留 cell 仍在共享 cell store，"短分支重指尾"不会回收——**内存增长（非正确性）**，长跑服务可能随请求多样累积。低危，建议周期回收。
3. **batch pos/seq 复用**：state prefill `pos` 从 0（`kev_model.cpp:148`）、branch 从 `Ls` 续（`kev_model.cpp:176`），各 `decode_seq` 都 `llama_batch_init` 新申请（`kev_model.cpp:109`），n_seq_id=1、seq_id[0]=seq；同 `scratch_seq_` 反复用，但每 branch `seq_cp` 重指尾后再 decode，**位置无碰撞、无概率漂移**。✅ 未发现问题。
4. **LRU evict stale**：满时 `lookup` 先 erase 旧项返回其 seq，`ensure_state` 再 `llama_memory_seq_rm(evict,0,-1)`（`kev_model.cpp:146`）清空整段再 prefill → **无 stale KV 泄漏**。✅

---

## 按性价比建议（增量口径，排除 PERF 已列）

1. **🔧 消除 `systemone_json` 双 encode + 三遍 tokenize**（§4）：确定性 ~1-3ms、零风险，纯 src。
2. **🔧 修 `register_` 重复赋值/潜在重复项**（§6.1）：纯防御、零成本。
3. **🔧 给 scratch 加周期 `seq_rm` 回收**（§6.2）：长跑防内存增长。
4. **🟦 多问打包一次 decode**（§1/PERF P.4）：多问场景唯一结构性大项，但需改 vendored 多 seq recurrent 前向，且 branch 长度异→图形状异。
5. **（不做）head 向量化 / LCP 增量缓存**：head 微秒级、LCP 有 recurrent 一致性风险，性价比≈0。