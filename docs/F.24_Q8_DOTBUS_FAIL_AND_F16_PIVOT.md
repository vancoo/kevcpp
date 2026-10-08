# kevcpp — branch(151) 前向根因结论 + Q8_0×Q8_0 dpbusd 内核原型实证(证伪) + 转向 (a) F16 激活路径 决策

> 分支: perf-gemv-q8tiled @ 2607590 (当前工作区, 提交干净, 仅 untracked: logs/, microbench/ 新增, tests/ fixture)
> 机器: AMD Ryzen AI MAX+ PRO 395 (Zen5, 32 逻辑核, AVX-512 全宽 + AVX-VNNI `vpdpbusd`)

---

## 0. 结论速览

1. **用户核心质疑"800M 读一遍为何几百 ms"已彻底闭合**, 逐 op 有据 (`logs/branch_151_forward_budget.md`):
   时间不在"读权重", 而在 **Q8_0 GEMM 内核只跑到 ~200 GMAC/s (~4-5% 的 Zen5 vpdpbusd 峰值 5.1 TMAC/s)**。
2. **审计建议的"重写 Q8_0×Q8_0 内环为 `_mm512_dpbusd` tiled 内核 → MUL_MAT 271ms→sub-100ms" 被实证证伪**:
   完整 512-bit 原型在干净实现 + 向量化 dequant 下仍被 **~190-256 GMAC/s** 封顶, 与现 ggml tiled 内核持平。
   根因是 Q8_0 的 **per-32-subblock dequant + 每输出 `-128*bsum` 修正** 的数值开销, 不是 SIMD 宽度。
3. **破 500ms 的正确方向 = 换数值路径 (a) 激活侧去 Q8 用 F16/F32 累积**, 而非修改内核微结构。

---

## 1. 逐 op 时间预算 (来自审计, 已核实; 详表见 logs/branch_151_forward_budget.md)

来源: `logs/m7_sqtrace.txt` [`KEVOPS`], decode(151)=429.5ms (隔离 cold_profile), 服务端稳态 probs ≈ 525-535ms (加 seq_cp + head + 调度/噪声 ~95-105ms)。

| Op | 累积 us | 次数 | % |
|---|---|---|---|
| **MUL_MAT** | 271 319 | 187 | **63.2%** |
| **CONCAT** | 49 144 | 18 | 11.4% |
| **GATED_DELTA_NET** | 30 812 | 18 | 7.2% |
| FLASH_ATTN_EXT | 5 529 | 6 | 1.3% |
| SSM_CONV | 2 355 | 18 | 0.5% |
| RMS_NORM | 2 236 | 36 | 0.5% |
| GET_ROWS | 1 176 | 74 | 0.3% |
| SCALE/ADD/MUL/GLU/ROPE/CPY/SET | ~6 000 | 多 | ~1.4% |

- 187 MUL_MAT = 144 (GDN, 8/层) + 42 (全 attn, 7/层) + 1 tied lm_head (已 slim M=151→6)。
- 模型 = Qwen3.5 HYBRID: 24 块 = 18 Gated-DeltaNet + 6 全 attn (3,7,11,15,19,23)。n_embd=1024, ffn=3584, ssm_d_inner=2048, ssm_d_state=128。构建器 `third_party/llama.cpp/src/models/qwen35.cpp:155-224,470-483`。

### 地板对比 (核心量化)
- DRAM 单读 547MiB: 1.6-2.6 ms (机器纯读峰 353 GB/s)。
- 理想 int8 计算: 77 GMAC / 5.394 TE12 (实测 Zen5 VNNI 峰) ≈ **14.3 ms**。
- 实际 Q8_0 内核: ~200 GMAC/s → **~385 ms**。

### 决定性实验
`microbench/gemm_pattern.exe` (真实 Q8_0 dot, M=151 K=1024 N=3584): 权重重读 151× vs 1× → speedup **0.73-1.0×**。
=> 内核 MAC 管线懒始于 DRAM(不在等待权重), 而非 DRAM 受限。

### 无冗余可消
权重已读一遍(F.23)、GDN 顺序 fp32 递归不可并行、CONCAT 49ms 是每层装配 + scheduler 同步伪影、lm_head 已 slim、6 层全 attn 仅 1.3% 且 flash 融合。前向是**真 int8 内核-MAC 受限 + 顺序 GDN 受限**。

---

## 2. Q8_0×Q8_0 dpbusd 512-bit 原型 — 实证证伪审计建议

### 交付
- `microbench/q8_gemm512.cpp` + `microbench/build_q8_gemm512.bat` (untracked, 未动 tracked 源)。
- 几何: M=151 K=1024 N=3584 (审计 FFN 主形状)。
- 数值: 与 ggml Q8_0 `+128` bias 技巧逐位一致; 用标量 fp 参考核对 **max_err≈0.0002 全过 (OK)**。

### 结构演进与测得吞吐 (16 线程, best-of)
| 结构 | GMAC/s |
|---|---|
| 简单 per-subblock `_mm256` 风格 dpbusd | 193 |
| 16-lane fp 累积 (corr 每 lane 重复 → 已修) | 124 → 修复)
| fmadd 向量 dequant + per-output scalar corr (最终) | **190** |
| 8 / 32 线程 | 256 / 230 |

### 结论
**512-bit 全宽 + 向量化 dequant 无法让 Q8_0×Q8_0 突破 ~250 GMAC/s**, 与现 ggml tiled 内核持平 (~5% 峰)。
DPBUS 的 int8 MAC 便宜 (峰 5 TMAC/s), 但 Q8_0 数值路径 (per-32 块 scale + 128*bsum 修正) 使每个 MAC 的"外包" fp 开销吃掉吞吐。
=> **根因是 Q8_0 去量化, 不是 SIMD 宽度 / 不是 256 vs 512。审计"换 512 即可 5-20×"前提不成立, 被 falsified。**

---

## 3. 转向 (a): 激活侧 F16 数值路径 (决策已定)

### 理由
- argmax 反正已产出 fp 激活; 当前 Q8_0×Q8_0 的优势只在省字节(读一次), 而读一次仅 547MiB≈2ms, 不成瓶颈。
- Q8_0 权重 × **F16 激活** 是 llama.cpp prefill 的标准带宽受限路径: 权重 Q8_0 只反量化一次(src0 侧), 激活保持 f16, 内环转 **FMA**, 可远超 200 GMAC/s。
- 代价: 激活 tensor 从 Q8(34B/32) 变 f16(2B/elem) → 激活侧内存/带宽上升; 但 M=151·K≤6144 激活块很小, 可忽略。

### 待办可行性微基准 (交给子代理)
1. 实测 **Q8_0 权重 × F16 激活 的 FMA 内环** 能否上 1+ TMAC/s (远超 200 GMAC/s)。
2. 若可行 → 评估进 ggml `MUL_MAT` 的路径 (src0 Q8_0 + src1 F16, 门控 env 保持逐位一致) 与预期 MUL_MAT 271ms 的降幅。
3. 产出: 微基准平均/峰值 GMAC/s + 进 ggml 的接入建议 + 是否值得继续。

---

## 关联
- 根因报告(sub代理): `logs/branch_151_forward_budget.md`
- 决策任务: 共享看板 `task-2`「pivot to f16/f32 activation numeric path」
- 现场证据: `logs/q8tiled_server_evidence.md`, `microbench/peak_int8.exe`(峰基线 5.1/8.0 TMAC/s @16/32thr)

---

## 4. (a) 可行性微基准 — 实测结果 (子代理 `46ed4604` + 独立复核)

### 交付物 (untracked, 未动 tracked 源)
- `microbench/q8_f16_gemm.cpp` + `microbench/build_q8_f16_gemm.bat` + `microbench/bin/q8_f16_gemm.exe`
- Q8_0 权重 × F16 激活 → FP32 累积; ggml block_q8_0 布局; AVX-512 FMA (`_mm512_cvtph_ps` + `_mm512_fmadd_ps`); MR=16 行/列, 线程分 N 列。
- 正确性: 对 9664 cells 标量 fp32 参考, **max_err≤0.0004 / mean_err≤8e-5 全过 (纯 fp 舍入)**。

### 实测 GMAC/s (本机独立复核 best-of)
| 形状 | thr=8 | thr=16 | thr=32 |
|---|---|---|---|
| FFN K1024 N3584 | 382(2.91ms) | **331-371** | 363 |
| attn_qkv K1024 N6144 | 414 | **381** | 380 |
| (Q8×Q8 对照 q8_gemm512 K1024 N3584 @16) | | **233** | |

全 K 扫描 @16thr: K1024 241-364, K2048 ~300-310, K3584 **425**, K6144 225。

### 判决
- **破 250 GMAC/s: 是, 决定性** — 主导两形状 ~330-414, 约为 Q8×Q8 的 **~1.4-1.6×** (本机独立复核: f16 381/331 vs q8 233)。
- **上 ~1 TMAC/s: 否** — FMA 实际封顶 ~400-425 GMAC/s (~30% fp32-FMA 理论 ~1.28 TMAC/s; 5.1 TMAC 是 int8 dpbusd 的峰)。
- 为何封顶: **f16 路径是激活读带宽受限, 非计算受限**。每权重列重读整块 M×K 激活 (15KB-1.9MB), load-port + cvtph + FMA 竞争。证明: 预转 f32(去 cvtph) 反而更慢(241 vs 389), 因激活字节翻倍 → 载数/装载是限界, 非转换。寄存器平铺重排激活复用在 acc 寄存器压力下受阻(每输出 cell 需 lo+hi fp32 acc)。

### 投影前向收益
- KT=K=1024, M=151 主导工作: 每 MAC ~1.5×; 投影 **MUL_MAT 271ms → ~175-195ms (~1.4-1.5×)**; 对 ~525-535ms 总前向 ≈ **~15-18% (~75-95ms)**。
- 注意: 真实 FFN 权重不全 K=1024, 大 K(6144) 掉到 ~225 GMAC/s → 实际收益取低端 **~15%**。

### 进 ggml 接入建议 (`KEV_Q8F16=1` 门控)
1. src0(权重): 每图 dequant+repack 一次(权重只读一遍 ~2ms, 非瓶颈)。
2. src1(激活): prefill 计算路径 src1 已是 f16(kevcpp 在 Q8 量化前就是 f16), 跳过 src1 的 Q8_0 量化, 把 f16 buffer 直给 MUL_MAT → 同时去掉 src1 量化 + 其 per-32 dequant + int8 -128*bsum 修正。**激活侧零额外字节**(prefill src1 本就是 f16)。
3. 内环: `_mm512_cvtph_ps` 每 32-fp16 块 → 2×(16-lane) wf, 2 FMA/行入 fp32 acc; MR=16/列, 线程分 N。
4. 数值: fp32 累积; 与 Q8×Q8 比特不同(fp16 激活精度 + 求和序) → **严格 opt-in `KEV_Q8F16=1`**, 与默认 A/B 干净可测。

### 决策
- **采纳进入下一步**: Q8_0×F16 是**真实、低风险 ~1.5× 收益**(MUL_MAT ~271→~180ms, 端到端 ~15%)。
- **但 ~400 GMAC/s 带宽受限, 不是上 ~1 TMAC/s 之路**; 那需 AMX 式 / L1 驻留激活平铺的外积核(k-blocked outer-product 使一行面板激活在 L1 跨列块复用)。
- 下一步建议: 交给子代理在 ggml-cpu `MUL_MAT` 落地 `KEV_Q8F16=1`(改 src1 走 f16), 以端到端 A/B 实测。

---

## 5. 整合实测 = NEGATIVE (子代理 bd29cb8f/d9625d3c, 已提交 perf-q8f16 @ 664f0d9)

### 关键前提修正 (反例, 推翻 F.24 §4 的集成假设)
- **kevcpp 的 src1 激活原生是 F32, 不是 F16** (`qwen35.cpp:512,527` 全部 F32)。
- **src1 的 Q8_0 量化发生在 ggml-cpu tiled MUL_MAT 驱动内部** (`from_float=quantize_row_q8_0`, 因 src0=Q8_0 时 vec_dot_type=Q8_0), 不是图节点、不是 kevcpp scarf 路径。
- 后果: F16 内核必须先**每次 MUL_MAT F32→F16 转换** —— F.24 sketch 假设"src1 已 f16, 激活侧零额外字节"**为假**。这个每 op 转换成本正好吃掉了理论收益。

### 实测 A/B (同机, best-of-N/中位, 16thr, ctx=2048)
- 服务器 `/v1/systemone` `latency_ms` (=model 前向): **KEV_Q8F16=0 中位 162.6 vs =1 162.3 ms —— 端到端零增益** (对照 tiled=0 legacy = 161.4)。
- 隔离 `cold_profile` branch(151) decode: **423.5 → 628.1 ms best, ~+48% 回归** (探针确认 Q8F16 真接入 M(ne1)=151 F32 项)。
- 正确性: argmax 一致(choice=1), probs ≤~1% 相对差(纯 fp 容差), 无 NaN/崩溃, 计算 buffer 尺寸匹配。

### 为何负
微基准 `q8_f16_gemm.exe` 的 ~380 GMAC/s 只在"src1 已是 f16 缓冲"下成立; kevcpp 图需每 GEMM F32→F16 转换, 覆盖理论增益, 且对大量小-N 分支 GEMM(如 1024×16 ssm_alpha/beta)转换+列串行激活重读远超过 GEMM 本身。

### 更重要的附带发现: 旧基线 ~525ms 已过时
当前 HEAD (2607590) 服务器 `latency_ms` ≈ **162ms**, 且 **KEV_Q8_TILED=0 vs =1 几乎无差** ⇒ **Q8_0 GEMM 内核不再主导 ~162ms 的服务器 total**。F.24 的 "MUL_MAT=63% / 271ms of 525ms" 预算对当前 build+此请求已 stale(因 2607590 CONCAT threaded 等后续优化已压掉旧预算大部分; 此请求 decode 也较小)。即使 Q8F16 内核更快, 也不会有 ~15% 端到端增益。

### 结论
**REVERT, 生产不启用 `KEV_Q8F16`**。实验保留于 `perf-q8f16`(默认路径 bit-identical 未动)供审。真正的版本需 src1 在图内保持 F16(转换-free 喂入)且排除小-N GEMM——当前不满足, 该 pivot 对 kevcpp 无效。
- 证据: `logs/q8f16_server_evidence.md` (完整 A/B + 正确性 + diff 说明)。

---

## 6. 权威基线(复盘正确地 440ms, 否掉之前 162ms 假象)

> 之前文档/会话里出现过的 "~162ms 已破 500ms" 是**错误**的, 由 PowerShell `Invoke-RestMethod` + `$body` 变量把 POST body 截断成 `input=70` 造成(残缺请求才 ~162ms)。**已更正作废**。真实请求 `tests/tmp_a_company.json` tokenize 为 **input=189**(state 38 + branch 151)。

### 6.1 权威稳态实测(本次会话, 字节保真, 见 logs/perf_authoritative_baseline.md)
- 协议: `curl.exe --data-binary @tests\tmp_a_company.json`, 每次确认服务端 `input=189 output=81`; 弃前 3 冷; `KEV_TRACE=1`。
- 稳态: **min 428.6 / p25 434.1 / median 442.9 / mean 455.3 / max 504.1 ms**(低噪声平台期 ~435ms; 464-504 尾巴为 Chrome/Postman 干扰); 首冷 ~588ms(state prefill)。
- 三层交叉验证(±10ms 内一致): (a) server `model_ms` ~428-437; (b) 逐-op 墙钟累加 ~426-431; (c) 隔离 cold_profile branch(151) best 433.6 / mean 444.7。
- 稳态 = **恰好一次 151-token llama_decode**(state LRU 缓存), 源码 + `graph_reused` 指标双重确认。

### 6.2 逐-op 预算(实测, ~428ms 前向)
| op | ms | % | 说明 |
|---|---|---|---|
| **MUL_MAT** | **335** | **78%** | 187 个, 全 M=151 tiled; ~339 GMAC/s (~6% Zen5 int8 峰) |
| CONCAT | 38 | 9% | 18 个, 每 GDN 层装配 + barrier |
| GATED_DELTA_NET | 31 | 7% | 18 个, 16-head × 151-token 串行 fp32 递归 |
| FLASH_ATTN_EXT | 5 | 1% | 6 个全 attn 层 |
| SSM_CONV | 5 | 1% | 18 个 |
| RMS_NORM | 4 | 1% | 115 个 |
| ROPE + 其他元素级 | ~5 | ~1% | |
| **合计** | **~426-431** | | = graph-compute 墙钟 |

### 6.3 关键新发现(lead 源码钉死): 未使用的 lm_head 跑 M=151
- tied lm_head(N=248320)在 **M=151** 跑, = **38.4 GMAC**, 占全部 GEMM 的 ~34%(total 113.5 GMAC, 非之前文档的 77 GMAC)。
- 机制: kevcpp `embeddings=true` + `pooling=NONE`(`kev_model.cpp:117-118`) → llama.cpp prefill/encode 路径强制 `n_outputs = n_tokens = 151`(`llama-context.cpp:1593-1597` `output_ids[i]=i; n_outputs=n_tokens`) → `inp_out_ids` 151 项(`llama-graph.cpp:2489-2502`) → `qwen35.cpp:211-219` 的 lm_head 在 M=151 对整个词汇表投影。
- 但应用只 `hidden_at_output`→`llama_get_embeddings_ith` 读 **7 行 hidden**(decide + 6 opts), **从不调用 llama_get_logits**。lm_head 输出全废弃。
- => 去掉/降到 M=7 可省 **~112ms**(38.4→0.9 GMAC), red-line 安全(7 行 hidden 在 lm_head 前已算好)。

---

## 7. 优化方向(权威预算驱动, 按稳态省时排序) — 逐个讨论与解决

> 目标: 稳态 ~440ms 压缩。三个方向都有实测依据; 逐项落地, 各配端到端 A/B(字节保真 curl, best-of-N/中位 + KEV_TRACE)。

| 排名 | 方向 | 预期稳态省时 | Red-line 风险 | 状态 |
|---|---|---|---|---|
| 1 | **砍掉/缩小未用的 M=151 lm_head → M=7(或整删)** | **~112ms → ~325ms** | 无(7 行 hidden 不变) | 🟦 待子代理(下一部) |
| 2 | **更快 Q8_0 tiled 内核**(~339 → 目标 ~1 TMAC/s) | ~150ms → ~180ms | 中(需保 fp 累加序 + 门控) | ⬜ 待评估 |
| 3 | CONCAT 融合 + GDN 串行开销 | ~45-70ms | CONCAT 无; GDN chunking 高 | ⬜ 待评估 |

**可达稳态**: Rank1 → ~325ms; Rank1+2 → ~180ms; Rank1+2+3 → ~130-180ms(噪声机 +30-60ms)。

**Rank2 诚实标注**: 不是 F.24 已证伪的 512-bit dpbusd 微重写 / 不是 Q8×F16 pivot(均已实测无效)。真正候选 = **权重预算反量化到 F16/F32 repack 一次、内环转 FMA**(成本一个 ~547MiB repack buffer, 改变数值需门控), 尚未实测, 需独立微基准验证, 不能保证赢。

---

## 8. 实测落地进度(权威基线起步 ~440ms → 逐项压缩)

> 当前已落地: Rank-1(砍 lm_head) commit `267300f`, ctx-fix commit `dd51ed7`; 已新开 `perf-rank2`。

### 8.1 Rank-1 完成(commit 267300f)
- 砍掉未使用的 M=151 lm_head。稳态实测(字节保真): **~445 → ~343-345ms(-100ms / -22%)**, 短请求 probs **逐位一致**(独立复核: 长款 `input=263` 200 / 短 `input=189` 概率逐位不变)。red-line 安全。
- 机制: `embeddings=true` → encoder 路径强制 `n_outputs=n_tokens=151` → lm_head M=151; 应用只读 7 行 hidden, 从不读 logits。

### 8.2 ctx 默认修复(commit dd51ed7)
- 根因(bug): `main_server.cpp:74` `ctx=2048` shadow 掉文档 8192 默认 → 每流注意力 KV 仅 256 cells → >256 的长请求报 `failed to prepare attention ubatches`。
- 修复: `ctx=2048 → -1` → 未传 `--ctx` 落 8192(`n_ctx_seq=768`), 传了用入参。并加**纯增量预检**: `input > n_ctx_seq` 时 decode 前返回清晰 `state too long for context`, 取代玄学 `could not answer request`。
- 验证(字节保真): 长请求 `input=263` 200(修复前 422)/短请求 `input=189` 概率逐位不变/`--ctx 2048` 长请求清晰 422。red-line 安全。
- ctx 最大值: `n_ctx_train=262144`(256K); 默认 8192 容纳 state ≤617 token。

### 8.3 Rank-2 微基准(未测候选"权重预反量化 → FMA"), microbench/deq_fma_gemm.cpp, 报告 logs/rank2_deqfma_microbench.md
- **结论: 该方向可行, 但只是 ~100ms 的有界收益, 非 ~1 TMAC/s 终极答案。**
- F16-weight variant 每形状都胜: FFN 477 / attn_qkv 502 / ffn_down 579 / ssm_out 449 / deep-K6144 512 GMAC/s(基线 339, 即 1.4x 主导 FFN, 1.5-1.7x 更深 K)。F32-w 更慢(高 K 崩到 254, 权重字节翻倍记忆受限)。
- **上 ~1 TMAC/s: 否**(FMA 实际封顶 ~580 GMAC/s ≈ 45% of 1.28 TMAC/s fp32-FMA 峰; 内存/load-port 受限, 复现 Q8×F16 发现)。
- 一次性 repack 成本: f16 ~50ms / f32 ~70-105ms(加载时, 摊薄 0/请求)。
- **NET**: F16-w 混合估算 → MUL_MAT 335→~227ms, 稳态 ~345→**~239ms**(含额外权重读 +2ms); F32-w → ~293ms(更差)。
- 正确性: SIMD vs scalar OK(F16w max_rel_err≤2.5e-4 / F32w 位精确); **F16 有权重量化误差(非 Q8_0) → 需 parity 门控**。
- **决策点(待 owner)**: 是否把 F16-w pre-dequant 内核落地进 ggml(需 `~+547MiB` repack 内存 + env 门控 + parity 校验, 稳态 ~345→~239ms)。注意: 到真 ~1 TMAC/s 需 register/L1 驻留 k-blocked 外积平铺(AMX 式), 本 FMA 微核达不到。

### 8.4 Rank-2 集成实测 = ALSO FALSIFIED(子代理 99bbc7d6, 按 commit-or-revert 规则已回退, 无 commit)
- 把微基准验证的 F16-w pre-dequant FMA 内核真接入 ggml-cpu(KEV_DEQFMA=1, lazy per-tensor Q8_0→F16 repack, 路由在 mul_mat_tiled, parity 通过 choice 一致 max|Δp|=0.0041)。
- **端到端实测反而慢 ~110ms(31%)**: 同窗默认 Q8_0 tiled median 355ms vs DEQFMA 465.6ms。微基准"能赢"≠ 集成"能赢" —— 与 Q8×F16 pivot 同款教训。
- 根因(子代理实测): 187 个 Q8_0 权重张量 → F16 = **2× Q8_0 字节, 每次前向从冷 DRAM 读**, 且 F32 激活被每线程重读(Q8_0 tiled 先量化成 1B)。→ ~2× 权重 DRAM 流量前向胜过 /*/*FMA 计算收益。
- **结论: F16 预反量化方向在真实集成下被判死, 不采用。** 真 GEMM 杠杆仍在 L1 驻留 k-blocked 外积平铺(微基准 §5), 非这个带宽 2× 路径。
- 证据: logs/rank2_deqfma_integration.md。DEQFMA 相关源码已回退, 分支 perf-rank2 @ dd51ed7 无 commit, tree 干净。

**规划**: 逐个方向(先 Rank1 lm_head → 再 Rank2 内核 → 再 Rank3 CONCAT/GDN), 每项由子代理实现 + 字节保真端到端 A/B + 位一致校验, 达标再 commit 到命名分支(沿本项目惯例)。

---

## 9. 当前真实稳态(所有已完成工作后)

> 权威字节保真基线(current HEAD `perf-rank2` @ `dd51ed7`, 短/长请求): **短 input=189 median 327.5ms(min 301.4/max 357.7); 长 input=263 median 333.8ms(min 307.7/max 365.0)**。

### 9.1 Rank-3a CONCAT = 伪影, 不落地(子代理 c5a703cc, 已回退)
- **"~38ms CONCAT" 是冷启动/contention 伪影**: 'perf_authoritative_baseline' 的 38ms 来自 fresh-server 首次 decode(冷 cache + Chrome 干扰, 2.13ms×18)。
- **稳态实测 CONCAT 仅 ~6.8ms(2%)**: 每 concat ~379µs(copy ~205µs strided + barrier ~174µs), 18 次。拷贝字节 3.78MiB/次(≈18 GB/s << 200 DRAM), **非带宽受限**; 主要是 `transpose(qkv_mixed)` 的 cache-hostile strided 读 + 每节点 ggml_barrier。
- 唯一真修法(融合进 ggml_ssm_conv consumer, 读 [conv_states|qkv^T] 分离 strided) 上限 ~7ms, **低于 ≥15-20ms commit 门槛**与"<10ms 不强做"纪律 → 不实现, 回退。
- 证据: logs/rank3_concat.md。HEAD 仍 dd51ed7, 无 server。

### 9.2 已完成真实收益汇总
| commit | 内容 | 稳态 |
|---|---|---|
| 基线(权威) | — | ~440ms |
| `267300f` (perf-lmhead) | Rank-1 砍未用 lm_head | ~343ms |
| `dd51ed7` (fix-ctx-default) | ctx 默认 8192 + 长请求清晰报错 | ~327-334ms(短/长) |

### 9.3 剩余真实杠杆(按价值排序, 均有实测依据)
1. **Rank-2 Q8 GEMM 内核(~250ms, ~77%)** — 唯一 100ms 级杠杆; 但已证伪 3 个 naive 方向(512-bit dpbusd / Q8×F16 激活 / F16 预反量化权重), 真正残留候选 = **L1 驻留 k-blocked 外积平铺(AMX 式)**, 改动大、不保证集成能赢。
2. **Rank-3b GDN 递归(~30ms)**: 16-head×151-token fp32 串行; chunking 改求和序 → 高红线风险, 需默认关 env + parity。
3. **CONCAT(~6.8ms)**: 已判死, 不追。
- 注: 自定义 warning —— 若追求极致, 需先解决"微基准能赢≠集成能赢"(F16 集成教训), 每一步都要集成级字节保真验证。

### 9.4 LCP 前缀复用可行性 = 技术全通但 ROI≈0, 不建(子代理 da098c13, 报告 logs/lcp_prefix_feasibility.md)
- **源码可行**: llama.cpp 原生支持"中段续算"(recurrent state = per-seq tail cell; seq_cp 浅共享; find_slot 位置续算只 WARN 不强制; hybrid seq_cp 同时浅共享 attn KV)。kevcpp 本就对每个 branch 用 seq_cp+续 decode(kev_model.cpp:277/280), 机制全同。**无需改 vendored**, 只需改 kev_prefix(LCP)/ensure_state。
- **bit-exact(实测, 决定性门槛通过)**: REF(整段 prefill) vs SPLIT(前缀 prefill→seq_cp→续尾巴) 在真实 kev-merged-q8 上 **61/61 位置逐位全等, max|Δ|=0, head probs 相同**; 负对照(从不同前缀续) max|Δ|=7.3 → 证明敏感、0.0 非伪影。计时: REF(61tok)312.5ms vs SPLIT(47+14)201.9ms, 省 ~110ms 共享前缀计算。
- **ROI≈0(实测)**: 5 个真实新闻 state(银行政策/产权/EV电池/暴雨/跨境研发) 彼此仅共享 ~3/38 tokens(~8%), 低于噪声; LCP 只在与"追加摘要"场景(37/38, 97%)受益 —— 而完全相同的 state 已被精确 LRU 覆盖。
- **结论: 不做 LCP 缓存。** 务实答案 = 精确 LRU + warm_graph + --prewarm + 自动线程。bit-exact SPLIT 配方(1-2 天, 零 llama.cpp 改动)留档, 供未来"同 news 基座 + 追加摘要"负载用。

### 9.5 【更正】LCP 通过"完整 state 槽复用"在 kevcpp 下不可行(实现子代理 22603dc3 实测证伪, 已回退)
> §9.4 的"bit-exact + 零 llama.cpp 改动可行"是**有条件**的(只在前缀作为独立 seq 停在边界 + 单流 KV 时成立); kevcpp 当前不满足, 直接实现会破坏 bit-exact。
- **实现子代理实测(真实模型, 受控对 stateA 189tok / stateB 219tok / LCP=25)**: LCP 路径在**多流 KV 下崩溃**(seq_cp assert), 在统一 KV 下跑通但 **choice 翻转(投资→产线产能), max|Δp|=0.0419 → 位一致红线被破坏**。默认 cold 全量 prefill 路径字节不变, 但 LCP 触发的路径本身是错的。
- **两个代码事实(我已亲自源码核实)**:
  1. **recurrent `seq_cp` 忽略 p0/p1, 只拷最深 tail cell**(llama-memory-recurrent.cpp:249-284 `tail_dst.tail = tail_src.tail`)。kevcpp 完整 state 槽的 tail 在其末尾(如 pos 67), 无法取其"中间 pos k"状态 → 续算用错起始 state。
  2. **kevcpp 多流 KV**(`n_stream=n_seq_max`>1): 跨流 partial `seq_cp` 断言失败(llama-kv-cache.cpp:496-506 `is_full && "seq_cp only for full KV buffers"`); 仅同流才浅共享。启用统一 KV(kv_unified)虽同流浅共享, 但多个 pos-0 state 槽在共享 cell buffer **碰撞**(实测 stateA 0.3974→0.398 漂移)。
- **验证子代理 §9.4 bit-exact 结论为何与实现冲突**: 它的实验把 prefix **作为独立 seq 停在位置 k**, 该 seq 的 tail 恰好 = pos k 状态(满足 recurrent 语义); 且假设单流。kevcpp 用"完整 state 槽"两前提都不满足。
- **结论: 不能通过复用完整 state 槽做 LCP。** 唯一正确设计 = **专用长寿命 prefix 槽 + 单流共享 KV**(类似 kev model.py 的 prefix_min_tokens), 是更大重构, 且每次只有第 3+ 次共享同一 prefix 才省(前两次要付 prefix prefill)。对当前流量, 精确 LRU + warm_graph + --prewarm 已零风险覆盖"完全相同的重复请求"。**不建议继续这条路线。**
- 证据: logs/lcp_implementation.md。相关源码已回退, HEAD dd51ed7(perf-lcp) 无 commit, tree 干净。

### 9.6 前缀槽正确设计实现 = ~41% 性能大赢 但数值仍非 bit-exact(子代理 11f68001, 已回退) — 根本根因=batch 宽度依赖
> 用户真实场景: 连续 ~50 个请求、state 前缀固定(占 ~80%)、前缀仅寥寥几种。用第 9.5 的**正确设计**(独立前缀槽: 前缀自身的 seq 恰停在边界 + 整段 seq_cp(0,-1) + 只续算尾巴)实现。
- **成本检查(达标)**: 整段跨流 seq_cp 深拷贝本身 bit-exact(纯拷贝探针 0/1024); copy+tail(10 tok)=56ms vs 全量 prefill(217 tok)=466ms(**省 87.8%**) → 成本不是瓶颈。
- **实测性能(~41% 大赢)**: 12 个共享前缀不同尾巴 state, 前缀关 baseline median **741.0ms → 前缀复用 438.8ms**。
- **BUT 位一致 FAIL**: max|Δp|=0.0051, choice 翻转(产线产能 0.353 vs 投资 0.356, 近并列)。
- **根本根因(非缓存 design, 决定性)**: llama.cpp **分段解码对 batch 总宽度数值敏感**。证伪性证明: **同一 seq 自然的拆两段 decode(完全无 seq_cp) 也偏离 max|dh|=0.0637**, 而跨流纯拷贝=0。→ 前 std 缀作为独立 decode 永远不会与"全长单次 decode 里的同一段"逐位一致, 这是 GDN/attention per-token 对总 token 宽度的依赖(PERF 附录 B 早记"拆两次 vs 整行不逐位一致")。kevcpp 多流下必现; 验证子代理 §9.4 的 bit-exact 0 是单流/单 seq 的巧合前提。
- **两条出路(需 owner 决策)**: (1) 改 vendored llama.cpp 让 per-token 数值对 batch 宽度不变(大改、可能改不动); (2) **明确 opt-in 接受 fp 级 probs 偏离**(近并列时 choice 可能翻, 如本数据)。对"完全相同的重构请求", 现有精确 LRU + warm_graph + --prewarm(~287ms branch-only)已是 bit-exact 的最优。
- 证据: logs/prefix_slot_implementation.md。回退, perf-lcp dd51ed7 无 commit, tree 干净。

---

## 10. 激活稀疏(SILU/SwiGLU) down 跳过 = 不划算(子代理 c253ea86, perf-gelu 分支探索, 已回退; 报告 logs/gelu_sparsity_exploration.md)
> 【重要前提纠正】Qwen3.5 的 FFN 激活是 **SILU(SwiGLU)** 不是 GELU: `down(silu(gate·h)⊙(up·h))`(qwen35.cpp:485/624 LLM_FFN_SILU, LLM_FFN_PAR)。此前所有"GELU 稀疏"讨论的前提是错的, 以下用实测真 SILU 激活。
- **实测稀疏度(24 层×3584×151, 短/长请求一致)**: |a|<1e-3=3.9% / <1e-2=24.6% / <5e-2=69.7% / <1e-1=89.6%; top50% 携 88.9% 质量, top10% 仅 46.2%。**分布平坦到中度偏斜, 非 DejaVu 式极端稀疏**; 23 层(末)稠密。
- **跳过能量损失小但 probs 翻转大**: skip 40%/52%/70%/84%/90% down 行只掉 0.06%/0.18%/0.6%/1.5%/2.1% 能量, **但** sharp 线性 head 读取 + 残差网络把 0.2% 能量损失放大成 0.07 概率摆动 → 真实近并列翻转。
- **近并列极常见**: 21 问/17 请求实测 **29% top-2 gap<0.02, 38%<0.05, 中位 0.064**。tmp_a_company 是真实近并列(产线产能 0.3974 vs 投资 0.3960, gap 0.0014), ≥84% skip 翻转; tmp_long 也在 thr≥0.05 翻转。即 repeat 前缀/LCP 同款近并列翻车。
- **收益上限太小**: down 仅占 MUL_MAT ~17-18%(max all-skip ≈18%≈45ms)。无翻转且 max|Δp|<0.01 的工作点(thr≤0.01, 24% down-MAC cut)只省 **~11ms(~3.4%)低于噪声**; thr=0.03(52%)省 23ms 但 Δp 到 0.073 已破质量。
- **结论: 不做固定阈值 SILU-down 跳过, 无质量可接受的工作点。** 不要为此改代码。大杠杆仍在 Rank-1 lm_head M=7 / Rank-2 Q8_0 内核。

---

## 11. 提吞吐(A 同state合并 + B 并发) = 也证伪, 根因=decode 计算绑定(子代理 f50b206c, perf-throughput 分支, 已回退; 报告 logs/throughput_batching.md)
- **实现**: 写完 BatchGate + compute_batch(BatchUnit, 同 state 跨请求合并 decode) + systemone_json compute 注入, **接线进 main_server**(--batch-window/max-items, window>0 走 batch, 否则逐 request), kev_batch.cpp 入 CMake。首个子代理中途停(kev_batch.cpp 未入 CMake / main_server 未引用), 续接子代理完成接线。
- **红线达标**: batch OFF 逐位=基线; batch ON(强并发)也逐位一致 max|Δp|=0。
- **实测吞吐 ≈ 0 增益**: 同步 K 路同 state 并发, req/s OFF K3=3.07 vs ON K3=2.97(倍率~1.0); merged-K 墙 ≈ K×单 decode, 与串行 OFF 无异; 单请求延迟不变。
- **根本根因(决定性)**: 此 0.8B 模型的 decode **是 compute-bound**(Q8 tiled ~339 GMAC/s ≈ 6% Zen5 int8 峰; MUL_MAT ~77% 前向, 见 perf_authoritative_baseline)。**M 翻倍 → GEMM 行计算翻倍, 没有权重读复用收益**(权重读一次 vs 每批读的差异在此计算绑定下可忽略)。M 合并提效的前提是变得 bandwidth-bound, 但这台 CPU+Q8 下不成立。
- **额外否决**: 能真捕获 K=4 同 state 的窗口会**崩溃** — 合并 4×151=604 tokens > n_batch=512 → `GGML_ASSERT(n_tokens_all <= cparams.n_batch)` (llama-context.cpp:1853)。
- **结论**: 不做 A+B。真提升吞吐需先把 Q8 GEMM 提到接近峰值(变成 bandwidth-bound/均摊单次权重读), 否则同 state 批合并无意义。**单请求延迟(~440ms)在 batch 开关下均不变**。
- 证据: logs/throughput_batching.md。回退 clean, perf-throughput HEAD dd51ed7 无 commit。