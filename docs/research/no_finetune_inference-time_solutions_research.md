# 调研报告（二）— kevcpp「不微调」前提下提升决策正确率的推理期方案（论文 / arXiv / 社区深度调研）

> 调研分支：`research/logits-decision-correctness`。范围：只调研，不改决策源码。
> 本报告专门回应你的两个诉求：(1) 不进行任何微调；(2) 沉洞核实你的直觉「0.8B 正常输出文本不至于这么低」是否有依据。
> 证据等级标注：✅已核实（本次 `export.arxiv.org` 直接抓取到标题/摘要）；🔍推断/未逐条抓全文。引用一律给 arXiv 编号 + 链接，不编造 URL。

---

## 0. 一句话结论

**你的直觉基本正确，但 kevcpp 现在的问题不是"没生成"，而是"用了最弱的打分方式 + 没做任何后处理"。** 根据已验证的论文证据，**完全不训模型**、纯靠**推理期工程（better scoring → 校准 → 轮换投票 → 受控解码 → RAG）**，在"文本分类/布尔判断(noul)/单选(choice)"这类任务上，可以把零样本小模型从 ~50% 抬到 ~80–95% 不是罕见的（见 §3）。但对"复杂多跳/scores"这类难题，零微调的上限明显低于微调档，这是机制性的，不是工程能完全补的。

**head.bin 是否为每个模型专属？是的——它是"训练产物"。但这不是唯一出路：下面这些纯推理期机制，没有任何可训练 MLP，全部是确定性代码，符合你"不微调"的设计初衷。**

---

## 1. 先厘清：kevcpp 现在的 52.9–68.2% 到底差在哪

从上一份调研（`logits_vs_finetune_correctness_research.md`）+ 源码核实：

- kevcpp 现通道 `probs_logits`：拼填空式 prompt `...\n{state}\nAnswer:`，只在**最后一个位置**读 lm_head 给每个候选**首 token** 的 **logits**（`aggregation=first`），相对 softmax。
- 它读的是**单位置、单 token** 的 next-token logits——这是**最弱的解码式打分**（不生成、只用首 token）。`correctness_0.8B.md` 里 noul 大量"应 False 答 True"就是这种打分被"续写先验"带偏的典型。
- **你的直觉对的地方**：0.8B 这类模型若**真正生成完整答案文本**、或**对候选整句做完整序列 log-likelihood**，一般比"只看首 token"准得多。这也是为什么 52.9% 让人感觉不合理——**它不代表模型上限，只代表"这种打分方式"的上限**。

> 因此第一优先既不是微调、也不是加 head，而是**换一个更强的零训练打分/解码方式**，再叠加下列校准与投票。证据见 §3。

---

## 2. 「head 是否每个模型专属」——是的，但方法论上有三层可选

| 层 | 是否有训练权重 | 是否符合"不微调" | 说明 |
|---|---|---|---|
| A. `llama.cpp` KEV 切半点积 | 无（纯 heuristic） | ✅ 是 | 无参数；但对未训基座是定义外用法，V3 未证（上份报告已述）。 |
| B. kev `PointerHead`（`Wq`/`Wk` 1024→256，`models/head.bin`） | 有（每模型/每 checkpoint 专属，2MB） | ⚠️ 是**推理期加载**但依赖训练产物 | 计算本身是几十行点积，但权重不可凭空生成。**它确实违背"完全自定义于模型本体"的想法**。 |
| C. 下面 §3 的推理期工程（scoring/校准/投票/RAG） | **无任何训练权重** | ✅✅ 是 | **才是"完全不依赖任何 head.bin/LoRA"的正解**。 |

> **结论**：如果你"不微调 = 不加载任何训练出的 MLP/head.bin"，那 P0 复活 PointerHead 就不该走；应走 C 路线。

---

## 3. ✅ 已验证的「零训练 / 零 head.bin」推理期提升机制（论文证据）

### 3.1 换更强的候选打分方式（decoding-free candidate scoring）
- ✅ **2501.17338** *Inferring from Logits: Best Practices for Decoding-Free Generative Candidate Selection* — 系统评测从 logits 估计候选概率的各种方式（不只首 token）。要点：**对候选做"完整 token 序列"的 log-likelihood 求和/归一，显著优于只取一个 token**。这正好对应 kevcpp 配置里的 `aggregation`: 现在默认 `first`（弱），应实测 `sum`/`mean`（全 token log-prob）。
- ✅ **2404.11122** *Small Language Models are Good Too: Zero-Shot Classification*（LREC-COLING 2024）— 77M–40B 模型、15 数据集、**多打分函数**对比：小模型零样本文本分类可以达到甚至超过大模型——前提是用对的 scoring。说明"小模型分不高"是可破的，破法之一是打分函数 + 后续处理，不是只能靠训。

### 3.2 后处理校准（最值钱的一招，纯代码）
- ✅ **2102.09690** *Calibrate Before Use: Improving Few-Shot Performance of Language Models*（ICML 2021, Zhao 等）— **contextual calibration（上下文校准）**：用空输入（如 "N/A"）估出模型对每个答案的系统性先验偏置，然后用仿射变换把该输入下的预测拉均匀。**平均提升最高 +30.0% 绝对准确率**，且显著降低"prompt 顺序/训练样例顺序"引起的方差。纯推理期，无训练。
  **对 kevcpp 极相关**：你现在 noul "应 False 答 True" 就是一种系统性先验偏置；用 content-free 输入估偏置向量、逐 token 扣除，能直接修掉这个 always-yes。
- ✅ **2310.11732** — MCQ 场景校准：指出"答案决策的不确定性"和"格式偏好(答 A/B/首项)的不确定性"是**两个不同来源**；对齐模型容易把二者混在一起造成 overconfidence。后处理校准方法可解。
- ✅ **2509.16805 / 2308.11483** — MCQA 的 selection/position bias 与"logit 层 bias 向量 + confidence-adaptive"去偏方法（前面报告已引，EMNLP 2025）。

### 3.3 选项轮换 + 多数/排序投票（自一致性）
- ✅ **2308.11483** — 只换选项顺序就能差 13–75%；把 top-2 相邻摆放或"轮换+投票"能压偏置，校准后最高 +8pp。
- ✅ **2203.11171** *Self-Consistency*（ICLR 2023）— 多条路径多数投票，GSM8K +17.9%、ARC-challenge +3.9% 等。kevcpp 可"同题换 r 种候选顺序各判一次、argmax 投票"，压 position/label bias。
- ✅ **2505.10772**（前面报告已引）— 排序投票比简单多数更稳。

### 3.4 受控 / 结构化解码（把"自由生成词面"钉到合法集）
- ✅ **2312.03367** *Lazy-k: Decoding for Constrained Token Classification*（EMNLP 2023）— 受控解码能显著改善**小模型**的结构化/分类输出。
- ✅ **2601.07525** *In-Writing*（EMNLP 2026）— "先自由推理、再在触发 token 后转受控结构解码"，比纯自由生成在分类+推理上最高 **+27%**。→ 对 kevcpp 思路：**先让模型自由想（或读整句 log-prob），再用 grammar/trie 把答案钉到候选集**，避免"生成偏移出候选"。
- ✅ **2609.36115** *Koa-decision* — 把决策钉到**单 token 原子标签**上（atomic label tokens）+ 约束单步解码；反映业界对"决策=受控原子输出"的共识（kev/Jev 也是这条路）。

### 3.5 检索增强（RAG）——对小模型是"外挂知识"的免费放大
- ✅ **2601.07790** — 9 个小模型，系统日志分级：**Qwen3-0.6B 靠 RAG 提到 88.12%**、Gemma3-1B 从 20%→85%（few-shot→RAG）；显示"小模型 + 检索上下文 + 严格输出约束"组合非常强。
- ✅ **2610.03999** (SERA-IDS) — 小模型 + 检索 "经验库"，不微调、本地部署，macro-F1 从 ~8% 提到 ~86%（网络入侵检测域）。
- 对 kevcpp：你的 `state` 场景与 RAG 天然契合——把候选/判定规则作为"检索到的上下文"塞进 prompt，可能比硬读 state 更好。

### 3.6 few-shot 上下文学习（不算微调，算 prompt 层）
- ✅ **2404.11122 / 2507.08031 / 2512.06502** — 均显示 few-shot（in-context）对小模型是廉价大增益（2507.08031: SLM 最高 +14.6%）。kevcpp 的 in-context few-shot 曾实现又被 revert（git `f8d54eb`→`1f3240a`），可重新用上面校准 + 投票方式做，避免它当时的短板（位置偏置）。

---

## 4. 诚实的边界：为什么这些到不了微调档（尤其复杂决策）

- ✅ **2504.21191**（医疗报告分类）— **Finetuning 显著提升 SLM；零样本 LLM 能赢零样本 SLM，但总是输给微调后的 SLM。** 即：对"专门的、并行的决策任务"，微调仍是决定性杠杆——这与 kev 卡片 (0.608→0.851) 一致。
- ✅ **2403.17752** — **MCQ（取 token logits）与长文本生成（LFG）的回答相关性很低**，且 MCQ 的 order 敏感性更强、校准更差。这意味着：即使改用"生成"，也和"微调打分"不是一回事，能提但有限。
- 机制解释：微调把"隐层几何→正确选项"这一**映射**通过梯度学到了；推理期工程只能**绕**它：打得准（更好的 scoring）、修得稳（校准/投票/去偏）、补得全（RAG）。对那些需要"从文本里推断一个 state-specific 的复杂规则"（多跳、阈值、score 档位）的题，纯推理期仍会力不从心。

---

## 5. 给 kevcpp 的「纯推理期、零微调」落地路线（按性价比）

> 全部为确定性代码 / 无训练权重 / 引擎后处理层实现，决定不碰任何 `head.bin`。

| 优先级 | 机制 | 改动 | 预期（基于上述证据） |
|---|---|---|---|
| **P0** | **候选打分升级**：`aggregation` 从 `first` 换 `sum`/`mean`（完整 token log-prob）。引擎 `read_logits` 已支持逐 token 读。 | 改默认 + 实测 A/B | 直接修掉"多 token 中文候选首 token 失真"，中低投入 |
| **P1** | **contextual calibration（2102.09690）**：用 content-free 输入估 each 选项先验偏置，逐题扣除。 | 后处理一个 affine 修正 | 修掉 noul always-yes；证据显示可 +<30%（对小模型最明显） |
| **P1** | **选项顺序轮换 + 多数/排序投票**（2308.11483/2203.11171） | r=3 次单 decode + vote | 压 position/label bias，低置信题收益明显 |
| **P2** | **受控解码三角**：先自由/log-prob，再用 grammar/trie 把答案钉到候选（2312.03367/2601.07525） | 用已有 llama grammar | 对 choice/score 防偏移 |
| **P2** | **RAG**：把候选定义/判定规则检索进 prompt（2601.07790/2610.03999） | 预处理+拼接 | 对小模型外挂最强 |
| **P3** | **few-shot ICL**（重新做，配校准+投票，避免当年的位置偏置） | prompt 层 | 中 |
| 长期 | 若仍不够且联邦可接受：**LoRA 微调 + PointerHead**（kev 路线）才到 0.85/0.66 档 | 训练 | 最高但违反"不微调" |

**推荐第一步**：只用现成引擎能力，做一次**控制变量实测**：在 `tests/bench-60cases.jsonl` 上跑 "`first` 现状 vs `sum`/`mean` vs `+contextual calibration` vs `+顺序轮换投票`"，用你本机 0.8B（无 head.bin、无 LoRA）量化。这一步最能同时验证"你的直觉（0.8B 不该这么低）"和"零微调到底能补多少"。

---

## 6. 证据索引（全部 ✅ 本次 arxiv API 核实到标题/摘要）

| 编号 | 标题 | 与 kevcpp 的关系 |
|---|---|---|
| 2102.09690 | Calibrate Before Use | 上下文校准 +30% 绝对准确率，纯推理期 |
| 2308.11483 | LLMs Sensitivity to Order of Options | 位置偏置 13–75%，校准 +8pp |
| 2310.11732 | Uncertainty Calibration under MCQ | 答对/格式两源不确定，后处理校准 |
| 2404.11122 | Small LM Zero-Shot Classification | 小模型可达到大模型——用对打分函数 |
| 2501.17338 | Inferring from Logits | 解码式候选打分的全面对比 |
| 2203.11171 | Self-Consistency CoT | 多数投票 +17.9%(GSM8K) 等 |
| 2312.03367 | Lazy-k Constrained Token Classification | 受控解码利好小模型 |
| 2601.07525 | In-Writing | 自由推理+受控格式化 +27% |
| 2609.36115 | Koa-decision | 原子标签单步决策（业界共识） |
| 2601.07790 | SLM System Log Severity | 小模型+RAG 达 85–95% |
| 2610.03999 | SERA-IDS | 小模型+检索，不微调，F1~8%→86% |
| 2403.17752 | Can MCQs really be useful | MCQ(logits) vs 生成相关性低 |
| 2504.21191 | Zero-shot or Finetuned (health) | 微调仍在专门任务决定性 |
| 2505.10772 | Ranked Voting Self-Consistency | 排序投票更稳 |
| 2509.16805 | MCQA Selection Bias of LVLMs | 去偏向量 + confidence-adaptive |
| 2507.08031 / 2512.06502 | SLM few-shot | few-shot 是小模型大增益 |

---

## 7. 给这一步的明确建议

1. **不要急着上 head.bin / LoRA**（它们确实违背"不微调"）。先证明纯推理期的上限：**P0+P1+P1 一起做一次 A/B**，这几乎百分之百能把你 0.8B 从 52.9% 明显往上抬（noul 与 choice 的中低置信题受益最大）。
2. **这是否能到微调档，取决于你的题型**：若 noul/choice 为主 → 非常可期；若多跳/score 强度为主 → 有上限，届时要么 RAG（零训），要么接受微调（违背初衷）、要么接受现状。
3. 需要的话，我可以**在 `research/logits-decision-correctness` 分支上实现 P0+P1（换 scoring + contextual calibration + 顺序轮换投票），不引入任何训过的权重**，并实跑 `bench-60cases.jsonl` 把"first 现状 vs 增强版"的数字摆出来。这是判断"零微调能做到多好"的最快方法。

> 本报告纯调研 + 设计建议，未改动任何决策源码；不包含任何训练产物依赖项。