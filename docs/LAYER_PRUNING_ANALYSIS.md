# Kev 层与 LM head 裁剪可行性分析

> 研究/分析报告（不修改代码）。回答：既然 Kev 只用到 prefill、完全不做生成式 decode，那模型的多层和 LM head 还有必要全保留吗？
> 结论摘要见文末。证据均来自 `E:\van\projects\kev\kev\model.py`、`docs/model-cards/*`、`E:\van\projects\llama.cpp\src\models\qwen35.cpp`、README、PLAN.md。

---

## 0. 一句话结论

- **LM head：你在 kevcpp 里其实已经"免费"省掉了。** 因为 `tie_word_embeddings=true`，llama.cpp 不额外物化输出矩阵；且 kevcpp 用 `embeddings=true`（pooling=NONE），读取点是 `res->t_embd`（在 output_norm 之后、LM head 之前），LM head 的 matmul **根本没被计算**。这块没有任何可继续省的空间。
- **模型的 24 层：工程上当前不建议剪。** 剪层在"功能上可行"（PointerHead 读哪一层之后的 last_hidden_state 都能出概率），但满层表征是决策精度的重要来源——0.8B→4B→9B→27B 的精度随层数/参数量单调上升就是反证。真正能省需"层剪枝 + 知识蒸馏/微调重训"，那是研究项目，不是工程微调。
- **真正的省法在别处**：既然不采样、不走 LM head、且 prefill 是"状态算一次 + 每题延一次"，省的开销应来自**减小模型/量化/共享前缀**，而非砍层。

---

## 1. 层数能否裁剪

### 1.1 功能上可行

`kev/model.py` 只在 `last_hidden_state`（第 301/337/403/419/436 行）读取 `decide` 与 `opt` 位置的向量，交 `PointerHead`（193 行）打分。`last_hidden_state` 是"最后一个解码块之后的输出"：

- 若整个主干保留 N 层，则 `last_hidden_state` = 第 N 层输出。PointerHead 照常工作，**功能完全成立**。
- 唯一硬约束：裁剪后输出维度不变（hidden 仍 1024），head/q/k 维度不需改。

所以"裁到 N 层还能跑"——**这是真的**。但它不保证"还准"。

### 1.2 层数对精度的实际影响：**很大，且方向明确**

kev README 的各规模（=不同层数）new-sources 转移精度：

| 模型 | 层数(config) | 参数 | new-sources acc |
|---|---|---|---|
| Kev-0.8B | 24 | 0.8B | 0.648 / 0.697 |
| Kev-4B | 32 | 4B | 0.817 / 0.838 |
| Kev-9B | 32 | 9B | 0.822 / 0.852 |
| Kev-27B | 64 | 27B | 0.848 / 0.896 |

- 从 24 层(0.8B) 到 32 层(4B/9B)，转移精度跳 **+0.17~0.19**；到 64 层(27B) 再 + ~0.20。
- 说明**决策所需的高阶表征强烈依赖"深"**：很多问题（multi-hop、长政策、日期运算、judging）要靠深层把信息聚合/交换到 `<decide>`。把 24 层剪到比如 12 层，几乎必然显著掉分（层剪枝文献里通常要配合 distillation 才能守住）。
- 推论：**剪层不是"白省"，是"用精度换速度"**，且在这个任务上换比很大。

### 1.3 混合架构：DeltaNet 层 vs full-attention 层

Qwen3.5-0.8B 是 24 层混合：18 层 Gated DeltaNet（recurrent）+ 6 层 full-attention（每 4 层一个，层 3,7,11,15,19,23）。

- full-attention 层负责**长程、精确的显式 attention**（把 `<decide>` 与远距选项关联起来），绝大多数长程决策依赖它们。
- DeltaNet 层是**高效线性递归**，负责上下文压缩/状态，计算省、硬件友好。
- **这两类不能随手各剪几个**：它们分工互补。若剪，要按"每 4 层保留 1 个 full-attn"的结构成组剪（连 LoRA 一起），且仍会掉分。

### 1.4 训练侧：LoRA 覆盖所有层

- kev 的 LoRA (r=16, 11.3M 可训练) 覆盖 attention+MLP+DeltaNet **全部 24 层**的投影（`model.py` lora_targets="all"）。
- 裁剪后，被剪层的 LoRA 权重**必须一并丢弃**；保留层的 LoRA 权重不变。但保留层可能无法弥补被剪层原本承担的表征。
- 更根本：**训练时没有为"剪到 N 层"优化过**。直接剪叫"结构剪枝"，几乎总会掉点；只有"剪枝 → 对决策任务微调/蒸馏"（turbo 式）才可能守住或接近。这是一个独立的训练步骤，超出"工程优化"范畴。

### 1.5 小结：学术可剪 vs 工程直接剪

- **学术可剪**：层剪枝 + distillation / 微调是可做的研究方向，能显著减小模型 + 提速，代价是一次训练成本与可能的部分精度损失。
- **当前工程直接剪**：**不建议**。没有任何训练支持、会掉精度、且要处理混合架构成组剪 + LoRA 丢弃。

---

## 2. LM head / vocab(248320) 能否省

### 2.1 llama.cpp 的 LM head 形态：tied，已不额外物化

`config.json`：`tie_word_embeddings=true`。llama.cpp `qwen35.cpp`：

```
output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);  // 44 行
if (output == NULL) output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,"weight"), ..., TENSOR_DUPLICATED); // 47-48
```

- 因为 tied，**llama.cpp 不会额外分配/加载一个独立的输出矩阵**——`output` 就是 `tok_embd` 的重复引用（`TENSOR_DUPLICATED`）。所以内存上 LM head「省不省」没有可省的：它本来就没有独立存储。
- 换句话说 **vocab 维度 (248320) 不因为 tied 变成瓶颈**——tok_embd 本身就要 248320×1024，这是输入侧必须的，与 LM head 无关。

### 2.2 计算上：kevcpp 已经跳过 LM head

- 图里 `res->t_embd`（66-75 行区域）= `build_norm(output_norm)` 之后、`build_lora_mm(output)` 之前的那个 hidden-state 张量。
- `t_logits = build_lora_mm(model.output, cur, ...)` 才是走 LM head。
- kevcpp `kev_model.cpp` 设 `cparams.embeddings = true`、`pooling_type = NONE` → 只取 `t_embd`（= per-token hidden state），**根本不算 `t_logits`（=LM head matmul）**。
- 换到 torch 侧同样是 `AutoModelForCausalLM(...).model`（去掉 `.lm_head`，model.py:234），forward 只读 `last_hidden_state`，**不算 LM head**。

**结论：LM head 在 kevcpp 里既没多占内存（tied），也没多算（embeddings 路径跳过 t_logits）。这一项已经是最优，无可再裁。**

### 2.3 可忽略的小尾巴：output_norm

- kevcpp 仍执行 `output_norm` 的 RMS norm（一处，hidden=1024）。这是拿 `last_hidden_state` 所需的最小、一次性的轻量操作，**不值得为省它动图**。

---

## 3. 结论与建议（三档）

### (i) 真正能省、且工程现成可落地的部分

| 项 | 现状 | 结论 |
|---|---|---|
| LM head / vocab matmul | 已跳过（embeddings 路径 + tied） | ✅ 无需再动 |
| 输出不同 token 的 hidden | kevcpp 已只在 decide/opts 位取 | ✅ 已优化（M7 还会更进一步） |
| 前缀缓存 | 已实现（同 state 命中 state=0ms，warm 282ms） | ✅ 已落地 |
| **换更小模型/更强的量化** | 当前 Q8_0 0.8B | ⚠️ 可用性上的省法：若要更小，是往更小尺寸（现在已是 0.8B 最小档）或更强量化（Q4_K_M）走，但会损精度/标定 |

> 真正符合"不生成 decode → 更快"的工程杠杆：**prefill 摊薄、前缀复用、batch 合并、量化**，而不是砍层。这些才是"弱 decode"红利的正确落点。

### (ii) 学术可行、但需重训/蒸馏才可实际省

- **层剪枝 + 决策任务微调/蒸馏**：把 24 层剪到 12~16 层，用 Kev 数据再做一次 LoRA/全量微调以恢复精度。理论可得到"更小更快且仍准"的 Kev。成本=一次训练（如 0.8B 规模极低，甚至比 4B 便宜得多）。**这是唯一能靠"层"获得真实收益的路径**。
- **自蒸馏到更小的基座**：PLAN.md 提及尝试过 `9B→0.8B self-distillation`（负结果，note #10）；说明小基座蒸馏本身有先例但难做。

### (iii) 不建议动的部分

- **直接截断留前 N 层（不训练）**：会掉精度、破坏混合架构 + LoRA，省 1/N 计算但换来明显变差。**不推荐**。
- **去掉 full-attention 层只留 DeltaNet**：长程决策依赖 full-attn，去掉会大掉分。
- **为"省 LM head"改图/砍 vocab**：已经 tied + 已跳过计算，无利可图，徒增风险。

---

## 4. 对 kevcpp 的落地方向与风险（最终判断）

### 判断：**当前不值得裁剪层。**

理由：
1. 满层 24 层是决策精度的支柱（0.8B 已是 family 里最小档，剪层=进一步牺牲精度，而精度本就不如 4B+）。
2. 混合架构 + 全层 LoRA 使"干净剪层"无现成路径，需训练。
3. 速度瓶颈不在"层数多少"而是"每次都重建全图 + 多次独立 decode"——M7 已在解决（线程、n_ctx、按需 logits），这些收益远大于剪层且零精度损失。

### 若坚持要"更小更快"的推荐路径（按优先级）：

1. **先吃满工程优化**（M7 收尾：多线程 + 降 n_ctx + 按需输出 + 前缀复用）。预期把这套请求从 1s+ 压到 ~400ms 级，已接近 torch。
2. **若仍要小**：用 Q4_K_M 量化 0.8B（内存/加载更快，精度略损，需重标定温度）。
3. **若要彻底小 + 准**（研究级）：做"层剪枝(如留 16 层)+ 对 Kev 决策数据微调/蒸馏"，独立训练，再评估精度-速度甜点。这是唯一能靠减层真正获益的方向，但属于训练项目。

---

*注：本报告基于 kevcpp 当前实现（embeddings=true 跳过 LM head、前缀缓存已开、LoRA 全层覆盖）与 kev 公开文档。若后续把基座换成更大的 Qwen3.5（4B/9B/27B），1.1/1.3 节关于"层数↑精度↑"的论断仍成立，只是"是否值得剪"的阈值随目标精度要求而变。*