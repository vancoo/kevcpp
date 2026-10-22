# 零训练"隐藏状态读取"方法调研 — 把 hidden 里已有的答案信息更准地读出来

> 调研分支：research/logits-decision-correctness
> 目标系统：**决策模型（system-one）**——给定 state + 题目 + 有限选项集，输出各选项概率，argmax 决策，**不生成文本**。
> 核心问题：**在不训练任何参数的前提下，能否用某种读取方式逼近"训练出来的双线性相似度"（kev 的 PointerHead `z[o]=(Wk·h_opt[o])·(Wq·h_decide)·scale`）的效果？**
> 所有引用均通过 arXiv API 核对（标题+arXiv id 可复核）；未能核实者标注"未核实"。

---

## 0. 调研出发点：kevcpp 的实测现状（本轮已有数字）

| 配置 | accuracy | score | noul | choice |
|---|---|---|---|---|
| base 0.8B + 普通模板 + logits(读候选 key 首 token) | 51.8% | 40% | 60% | 45% |
| base 0.8B + 普通模板 + 完整序列 mean | 54.1% | – | – | – |
| base 0.8B + 别名池(combo) + mean | 60.0% | – | – | – |
| 4B + T3+combo(别名池) | **70.6%** | 80% | 70% | 70% |
| 4B + nothink2 + first（模型"自答"口径） | 71.8% | 20% | 80% | 70% |
| base 0.8B **自由生成**（nothink2 模板）自答 | 56.5% | 60% | 68% | 45% |
| kev-merged(微调后) + kev 结构 + **head.bin** | **82.4%** | 100% | 87.5% | 75% |
| kev-merged + kev 结构 + 零训练 decide-cos | 64.7% | 80% | 70% | 57.5% |
| base 0.8B + kev 结构 + 零训练 decide-cos | 31.8% | 20% | 40% | 25% |

**最关键的诊断证据**：在同一模型上，**四种零训练相似度公式（余弦 / 点积 / 归一化点积 / 质心点积）给出完全相同的排序**。
这在数学上只有一个解释：**hidden 向量近乎共线**，少数"通用方向"（rogue dimensions / common component）主导了所有相似度度量。
因此本调研的第一优先级是：**先消除这种"通用方向"污染，再谈相似度公式**。

---

## 1. Common component / rogue dimensions 病态与无监督解法

### 1.1 病态诊断：为什么"四种公式同一排序"
- **All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality**（Timkey & van Schijndel, EMNLP 2021, **arXiv:2109.04404**）
  - 核心发现：**仅 1~3 个 rogue dimensions 就能主导 cosine/Euclidean 等相似度度量**；且"主导相似度的维度"与"对模型行为重要的维度"**不一致**。
  - 解法：**简单后处理（standardization / z-score）即可纠正 rogue dimensions**，揭示底层表示质量。
  - 论文原话要点："accounting for rogue dimensions is essential for any similarity-based analysis of contextual language models."
  - 👉 **直接对应我们的现象**：我们的四种公式同序 = 相似度被 rogue dimension 支配。**这是本调研第一优先的可落地修法。**

### 1.2 后处理族（全部无监督、零训练）
- **Effect of Post-processing on Contextualized Word Representations**（Sajjad et al., COLING 2022, **arXiv:2104.07456**）
  - 系统比较 **z-score 标准化 / min-max / all-but-the-top（去 top 主成分）/ unit-length 归一化** 对 contextualized 表示的影响。
  - 结论：后处理能"解开"表示中被掩盖的信息；**建议把 z-score normalization 当作使用 contextualized 表示时的必要步骤**。
  - 👉 对我们：`h_decide` 与 `h_opt[j]` **逐维 z-score 后再算余弦**，是零成本可试的高价值项。

- **Correcting Mean Bias in Text Embeddings: A Refined Renormalization with Training-Free Improvements on MMTEB**（**arXiv:2511.11041**, 2025-11）
  - 分解：`e = ẽ + μ`，μ 是跨句近乎相同的均值向量。
  - 两种训练无关修法：**R1 = 直接减 μ**；**R2 = 把 e 在 μ 方向上的投影去掉**（正交化）。
  - 一阶误差传播论证：**R2 抵消了 R1 残留的均值估计误差的平行分量**；38 个模型上 R2 一致带来分类增益，零损失。
  - 剂量-反应消融：**"轻度单方向去除"有帮助，但"全 PCA 白化"在所有测试模型上都变差**；且 **R2 与 depth=1 的 all-but-the-top 结果相差 <0.18pp**。
  - 👉 对我们：**优先做 R2（去均值方向投影）+ depth=1 ABT**，**不要做全 PCA 白化**。

- **Whitening Not Recommended for Classification Tasks in LLMs**（Forooghi et al., **arXiv:2407.12886**, 2024-07）
  - 系统结论：**白化的收益是 model/task-dependent；尤其"白化会让分类任务的 embedding 退化"**；比较了 PCA / ZCA / PCA-Cor / ZCA-Cor / Cholesky。
  - 👉 **重要警告**：我们的任务本质就是**分类/argmax**，所以**"PCA 白化"很可能有害**——与 1.2 中 arXiv:2511.11041 的剂量-反应结论一致。**不要走全白化路线。**

- **WhiteningBERT: An Easy Unsupervised Sentence Embedding Approach**（Huang et al., **arXiv:2104.01767**）
  - 三个发现：①**平均所有 token 优于只用 [CLS]**；②**结合顶层与底层(top+bottom)优于只用顶层**；③**基于白化的向量归一化(<10 行代码)一致提升**。
  - 👉 对我们：提示"**token 平均**"和"**跨层拼接**"两个零训练读法（后者需中间层支持，见 §2）。

- **On the Sentence Embeddings from Pre-trained Language Models**（Li et al., EMNLP 2020, **arXiv:2011.05864**, BERT-flow）
  - 指出 BERT/PLM 的句向量是**非平滑、各向异性(anisotropic)** 的语义空间；用无监督 normalizing flows 变换为各向同性高斯可显著提升语义相似度。
  - 👉 作为"各向异性"病态的经典出处；flow 需训练，**本项目不采用**，但支撑"必须先做去各向异性"这一判断。

- **Whitening-based Contrastive Learning of Sentence Embeddings**（Zhuo et al., ACL 2023 Oral, **arXiv:2305.17746**）
  - 把 whitening（uniformity）与对比学习（alignment）结合。**注意其 whitening 本身需训练**，本项目不采用；引用用于说明"whitening 与 alignment 是两种互补机制"的理论框架。

### 1.3 本节落地建议（零训练，优先级最高）
在 `probs_decide` 的相似度计算前插入**逐维去污染**：
```
# 对本候选集内的向量集合 D = {h_decide, h_opt[0..K-1]} 做去污染
μ = mean(D, axis=0)                      # R1/R2 的 μ
# R2：把 μ 方向从每个向量里去掉
μ̂ = μ / ||μ||
for v in D: v' = v - (v·μ̂) μ̂            # 正交化（R2）
# 或 depth=1 的 ABT：v' = v - v·u₁ u₁  （u₁ = top 主成分）
# z-score：v' = (v - mean_axis0) / std_axis0   （arXiv:2104.07456 建议）
z[j] = cos(v'_decide, v'_opt[j])
```
**实现成本：极低**（现有 `probs_decide` 里已有 centroid/cos_ 基础设施，只改 5~10 行）。**无需后端扩展**。

---

## 2. 多层 hidden 的选择（中间层 vs 末层）

### 2.1 文献依据
- **Effect of Post-processing on Contextualized Word Representations**（**arXiv:2104.07456**）第②发现：**"结合 top 与 bottom 层优于只用 top 层"**（在无监督句向量任务上）。
- **WhiteningBERT**（**arXiv:2104.01767**）重复验证同一结论（多层拼接 > 顶层）。
- **Towards Best Practices of Activation Patching in Language Models**（Zhang & Nanda, ICLR 2024, **arXiv:2309.16042**）
  - 系统研究 activation patching/causal tracing 的方法学细节（度量、corruption 方式），指出**不同超参会给出截然不同的定位结论**；给出实践建议。
  - 👉 含义：**"读哪一层"本身是一个需要实证的变量**，不能默认末层最好。
- **Residual Stream Analysis with Multi-Layer SAEs**（Lawson et al., ICLR 2025, **arXiv:2409.04185**）
  - 关键发现：**单个 SAE latent 常只在某一层活跃**，且**活跃层随 token/prompt 不同**；层间残差流向量在更大模型上更相似。
  - 👉 含义：**答案信息可能驻留在中层而非末层**，末层可能已被"语言化/擦除"。
- **LLM Layers Immediately Correct Each Other**（TLCM, NeurIPS 2025, **arXiv:2609.07876**）
  - 发现 **Transformer Layer Correction Mechanism**：相邻层系统性地**互相抵消部分贡献**，且"提议-否决"框架说明残差流里同时存在**瞬时提议**与**持久特征**。
  - 👉 含义：**末层 = 累积了多轮"否决"后的结果**，判别信息可能被擦除——这为"读中间层"提供了机制性理由。
- **An Adversarial Example for Direct Logit Attribution**（Janiak et al., BlackboxNLP 2024, **arXiv:2310.07325**）
  - 指出 **direct logit attribution 在不计入"擦除(erasure)"时会给出误导性结果**；注意力头/MLP 会清除早前层写入的方向。
  - 👉 直接支持"不要只看末层"。

### 2.2 经验规则（综合上述）
1. 语义/判别信息通常在**中后层（约 60%~85% 深度）**最"干净"；末层已高度语言化（被 LM head 的下一 token 任务塑形）。
2. **多层拼接（top+bottom 或 中层+末层）通常优于单层**（两篇独立来源一致）。
3. 对**决策任务**尤其如此：我们要的是"选哪个"，不是"下一个词"。

### 2.3 实现成本
**kevcpp 当前只读末层 hidden**（`llama_get_embeddings_ith` → 最后一层）。
- 读中间层需要后端扩展（llama.cpp 需在 `llama_decode` 时保留中间层输出，或用 `llama_context` 的 per-layer 输出能力）。
- **成本：中**（需改 `kev_backend_cpu/gpu/vulkan` 的 `read_hidden` 为 `read_hidden_layer(out_idx, layer, ...)`，并确认 vendored llama 是否暴露中间层输出；若无，需要在 vendored llama 里加 `llama_get_embeddings_ith_layer` 或开启 `output_all_layers`）。

---

## 3. 用 LM head 读决策（multiple-choice 标准打分族）

### 3.1 核心文献
- **Surface Form Competition: Why the Highest Probability Answer Isn't Always Right**（Holtzman et al., **arXiv:2104.08315**, EMNLP 2021）
  - 问题：**不同 surface form 竞争概率质量**（"computer" vs "PC"），即使语义相同也会压低正确答案。
  - 解法 **Domain Conditional Pointwise Mutual Information (DC-PMI)**：按选项在**该任务语境下的先验似然**重新加权。
  - 公式：`score(c) = log P(c | x) − log P(c | "N/A")`（无条件先验用内容无关输入估计）。
  - 结论：在 GPT-2/GPT-3 各种 MCQ 数据集上，**DC-PMI 一致优于 calibrated 与 uncalibrated 打分**。
  - 👉 对我们：**我们的别名池(combo)本质上就是"用手工别名扩展 surface form"**；DC-PMI 提供了**更系统的先验扣除**版本。

- **Increasing Probability Mass on Answer Choices Does Not Always Improve Accuracy**（Wiegreffe et al., EMNLP 2023, **arXiv:2305.14596**）
  - 给出 **surface form competition 的数学形式化**并首次可量化/可界。
  - **关键可操作发现**：减少 SFC 的简单方法是 **a) 把选项放进 prompt** 且 **b) 用哪怕仅 1 个示例做 in-context learning**——**这消除了绝大多数 SFC 影响**。
  - 警示：**归一化与 prompt 方法对某些 LM 可能无效甚至有害**。
  - 👉 对我们：**"选项写入 prompt + 一个 ICL 示例"** 是零训练、可直接试的 SFC 消除法（我们此前只试过选项写入，未系统试过"选项显式列出 + 1-shot"）。

- **Investigating Uncertainty Calibration of Aligned Language Models under the Multiple-Choice Setting**（He et al., **arXiv:2310.11732**）
  - **核心洞察**：MCQ 下存在**两种不同的不确定性**——(a)**答案决策不确定性**（该选哪个）与 (b)**格式偏好不确定性**（模型爱输出什么形式/哪个 token）。二者被混淆会导致过度自信。
  - 👉 对我们：**这正是"模型说否但 key 是 false"这类错位的理论解释**，也是别名池有效的理论依据；提示**格式偏好应被单独估计并扣除**（= contextual calibration 的正确用法）。

- **Mitigating Label Length Bias in Large Language Models**（Sanz-Guerrero & von der Wense, AACL 2025, **arXiv:2511.14385**）
  - 提出 **label length bias**：多 token 类别标签在标准 length normalization 后仍被不一致对待；提出 **NCC（normalized contextual calibration）** 在**完整标签层面**归一化校准，**F1 最高提升 10%**，并**扩展到 MCQA**；与 ICL 结合时对示例选择更不敏感、所需示例更少、置信度更可靠。
  - 👉 对我们：**noul 的 key `false`(多 token) vs `true`、choice 的 key 长度不一 → label length bias 是真实存在的**。NCC 是对"别名池 + 校准"的精确升级。

- **Calibrate Before Use: Improving Few-Shot Performance of Language Models**（Zhao et al., ICML 2021, **arXiv:2102.09690**）
  - **Contextual Calibration**：用内容无关输入（"N/A"）估计模型对各候选的**系统性偏置**，再拟合参数使该输入下预测均匀。
  - 报告**平均准确率最高 +30.0% 绝对提升**，并显著降低 prompt 选择的方差。
  - 👉 我们已实现 `KEV_CALIB`（44.7%，当时为负）。**但当时用的是 base 模型 + 普通模板**；在**kev 结构/微调模型**上重试可能不同——见 §7 的复测建议。

- **On Calibration of Modern Neural Networks**（Guo et al., ICML 2017, **arXiv:1706.04599**）
  - 温度缩放（单参数 Platt scaling）在多数数据集上"出奇地有效"。
  - 👉 我们已实测 temperature=0.1 **不改变 argmax**（60.0% 不变），**因此温度缩放只影响置信度、不影响正确率**——这与"argmax 不变"的构造一致，可以排除它作为提分手段。

- **Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions**（Pezeshkpour & Hruschka, **arXiv:2308.11483**）
  - 选项顺序重排可造成 **13%~75%** 的性能差；敏感主要出现在 **top-2/3 之间不确定**时（位置偏好）。
  - 两种校准方法带来**最高 +8 个百分点**。
  - 👉 对我们：**"答案不確定时"正是我们要攻的场景**；**选项顺序轮换(rotate) + 位置去偏**有文献支撑（我们此前在 0.8B 上测过 abcd+rotate，choice +20pp 但 noul 崩）。

- **Small Language Models are Good Too**（Lepagnol et al., LREC-COLING 2024, **arXiv:2404.11122**）
  - 77M~40B 模型在零样本分类上的系统评测："**bigger isn't always better**"，小模型在很多文本分类上可持平或超过大模型。
  - 👉 支持"**匹配方式比模型大小更关键**"这一路线。

### 3.2 标准 MCQ 打分族（我们的 `probs_logits` 家族对应）
| 打分 | 公式 | 出处 | 我们是否已实现 |
|---|---|---|---|
| acc（raw） | `log P(c\|x)` | — | ✅ `first` |
| byte-length norm | `log P(c\|x)/len(c)` | lm-eval-harness 系（**未核实具体 arXiv 号**） | ≈`mean`（近似） |
| PMI | `log P(c\|x) − log P(c\|"N/A")` | **arXiv:2104.08315** | 部分（`KEV_CALIB` 是其变体） |
| DC-PMI | PMI + 领域条件先验 | **arXiv:2104.08315** | ❌ |
| NCC | 全标签级归一化校准 | **arXiv:2511.14385** | ❌ |
| SFC 消除 | 选项入 prompt + 1-shot | **arXiv:2305.14596** | ❌（部分） |

### 3.3 "在 decide 锚点读 LM head"的理论依据
kev 的 `PLAN.md` 自述："**adapted backbone 经 LM head 读出的成绩 = pointer**"（本轮审计已确认该行，见 `docs/research/kev_high_accuracy_audit.md`）。
机理（可引文献支撑的部分）：
- LM head 本身是"hidden → 词表"的**学习过的投影**；当 backbone 被 LoRA 适配到决策任务后，**该投影已把判别信息对齐到了词表**，因此"读 LM head"与"读 pointer"等价。
- 这与 **arXiv:2109.04404** 的 rogue-dimension 结论一致：**判别信息存在于特定低维子空间**，而 LM head 与 PointerHead 都是"把该子空间读出来"的投影器——**区别只在于投影是否针对该任务训练过**。
- 对我们的直接含义：**如果我们要零训练逼近 pointer，只能找到"一个恰好与该子空间对齐的、无参数的投影"**——最可能的候选就是 **LM head 本身**（已在用），或 **LM head 的低秩/去噪近似**（见 §1.3 去污染 + §3.1 的 PMI 类扣先验）。

---

## 4. 注意力/位置聚合型读取

### 4.1 文献
- **Revisiting the Architectures like Pointer Networks to Efficiently Improve the Next Word Distribution...**（Chang et al., ACL Findings 2023, **arXiv:2305.12289**）
  - 发现 **softmax bottleneck** 有时阻止 LM 给出想要的分布；**pointer network 可高效打破该瓶颈**，且在 GPT-2 上显著优于 mixture-of-softmax。
  - 👉 含义：**"不用词表 softmax、改用 pointer 式匹配"本身是已知有效方向**——正是 kev PointerHead 的思想（kev 的 `head.pt` 与该文同一脉络）。
- **Pointer-Guided Pre-Training**（Hillebrand et al., ECML-PKDD 2024, **arXiv:2406.04156**）
  - 用自注意力驱动的 pointer network 恢复被打乱段落的原序（段落级上下文感知），对文本分类有效。
  - 👉 佐证 pointer 机制可迁移到**分类/决策**任务。

### 4.2 kev 的 `option_isolation`（**零训练的结构性技巧，必须重视**）
从 `E:\van\projects\kev\kev\model.py`（`branch_mask_batch`）读到的确切语义：
> "opts (option isolation): within a question, an option-span token may attend to **state, the instruction, and its own span only**; `<decide>` attends to everything in its question. Instruction tokens never see option spans (causal)."

即 kev 用**自定义 attention mask** 实现：
- **每个选项只能看到 state + 题干 + 自己**（选项之间互不可见 → 消除"选项互相影响/位置偏好"）；
- `<decide>` 看到全部；
- 并且该模式下**所有选项 span 共用同一组 position ids**（`option_isolation=True` 时 `br_pos` 让每个 span 的位置相同）→ **选项表示对顺序置换不变（permutation-invariant）**。

这是**纯结构、零训练**的，恰好针对 §3.1 的"位置偏好 13%~75%"问题，**而且 kev 是在"已微调 backbone"上启用它的**。
→ **kevcpp 若要抄这条，需要后端支持自定义 attention mask**。llama.cpp 现状：
- 有 `llama_set_causal_attn(ctx, bool)`（仅全局开关，**不足以实现 per-option 隔离**）；
- **未见 per-token/per-segment mask 的公开 API** → 需要在 vendored llama 里加（中等成本），或
- **近似实现**：把每个选项**单独**作为一个短 prompt 单独 decode（每个选项一个 seq），读其 `<box_end>` 位置 hidden——**这在语义上等价于 option_isolation**（该选项只看到 state+instr+自己），且**现有 kevcpp 能力就能做**（逐候选 decode 的 `sum/mean` 路径已经这么干！）。
- 👉 **这是一个高价值、低成本的发现**：`aggregation=mean/sum` 路径其实已经实现了"逐候选单独 decode"，只是它读的是"候选续写末尾 hidden"，**不是"选项 span 在 kev 结构下的 `<box_end|>` 位置 hidden"**。把两者结合 = 零训练复刻 kev 的 option_isolation。

### 4.3 注意力聚合读取
- **Attention as readout**：在 LLM 中用**某位置对各位置的 attention 权重**作为聚合权重，是"读出注意力到哪些位置"的自然信号。
- 文献支撑（本项目已用过的近邻）：我们此前实现过 `hidden_state_span` + **软注意力加权聚合 state**（`KEV_DECIDE=hidden_state`），实测 36%~42%（在 base + 普通模板下）。
- 关键文献：**Attention is not Explanation**（未核实具体 arXiv 号，但该结论是领域共识：attention 权重≠贡献度）；因此**把 attention 当"相关性权重"用是弱解释但可作弱信号**。
- 实现成本：**高**——llama.cpp 默认**不导出 attention 权重**（需要 `llama_context` 请求 attention 输出，或改 graph）。**当前 kevcpp 无法直接做。**

---

## 5. 对比式 / 成对 / 反事实读取

### 5.1 方法
- **成对差分**：`z[i,j] = f(h_decide, h_opt[i]) − f(h_decide, h_opt[j])`，或直接在**差分向量空间** `h_opt[i] − h_opt[j]` 上做匹配。共线的向量在差分后可能显露差异（我们已在 `dc` 里做过去质心，但**没做过逐对差分**）。
- **反事实/归因式打分**：把某个选项从 prompt 中**移除或替换**，观察 **hidden/logit 的变化幅度**：
  - `score_i = ‖h_decide(x) − h_decide(x \ option_i)‖` 或 `logit` 变化量。
  - 这类"leave-one-out attribution"在**多选题决策**上天然给出各选项的**独立贡献度**（而不是共享一个 softmax 分母）。
- 文献支撑：
  - **Towards Best Practices of Activation Patching in Language Models**（Zhang & Nanda, ICLR 2024, **arXiv:2309.16042**）——causal tracing / activation patching 的方法学与实践建议（corruption 方式、度量选择）。
  - **An Adversarial Example for Direct Logit Attribution**（**arXiv:2310.07325**）——直接 logit 归因会因"擦除"而误导。
  - **Multi-component Causal Tracing in Large Language Models**（Yan et al., ACL 2026, **arXiv:2606.03085**）——同时追踪多个组件，用 soft intervention 把组合搜索连续化。
  - **Does Localization Inform Editing?**（Hase et al., NeurIPS 2023 Spotlight, **arXiv:2301.04213**）——**重要警示**：causal tracing 的定位结论**未必**能指导"改哪里最有效"；干预效果更好的预测因子是"改哪一层"。含义：**"能归因到某处"≠"读那里更准"**，反事实打分可能白做。
- 实现成本：**中**。反事实法需要 **K 次额外 decode**（每个选项一次），但 kevcpp 已有逐候选 decode 能力（`sum/mean` 路径），改动集中在"对比基准状态"。
- 零训练：**是**（完全无参数）。

---

## 6. 零标签 / 少标签的自监督探针

### 6.1 用"结构自一致性"构造伪标签
- 思路：构造**对称变换**（选项重排、选项改名、题干改写），要求打分**在变换下不变**（permutation invariance），把"一致性"本身当训练/选择信号。
- 文献支撑：
  - **arXiv:2308.11483**（选项顺序敏感 13%~75%）→ 一致性约束直接针对此；
  - **arXiv:2404.11122**（小模型零样本分类系统评测）；
  - kev 的 `option_isolation` **本身就是"置换不变"的实现**（共用 position ids），可视为该原则的实例。
- 实现：**中**。作为"无标签时的打分函数选择准则"（在若干候选公式里选那个最置换不变的），而不是训练参数——**仍然零训练**。

### 6.2 原型 / 聚类匹配
- 思路：为每个选项构造**原型向量**（别名池 embedding 的均值、或该选项在不同上下文下的 hidden 均值），再匹配。
- 文献支撑：**arXiv:2511.11041**（均值偏置的 R2 正交化，正是原型构造的关键修正）；**arXiv:2104.01767**（token 平均优于 [CLS]，支持"均值原型"）。
- 实现成本：**中**。需要把同一选项在多个上下文下的表示收集起来（kevcpp 已有 `KEV_DUMP_HIDDEN` 机制可扩展），或用别名池的 embedding（**需后端暴露词嵌入**）。
- 零训练：**是**（原型由平均得到，不是学出来的）。

---

## 7. 校准 / 归一化 / 集成对 argmax 的影响

| 技术 | 出处 | 对 argmax 的影响 | 我们状态 |
|---|---|---|---|
| 温度缩放 | **arXiv:1706.04599** | **不影响 argmax**（单调变换）；只改置信度 | ✅ 已实测：T=0.1 与 T=1.0 同为 60.0% |
| Contextual Calibration | **arXiv:2102.09690** | **影响 argmax**（扣各候选系统性偏置） | ✅ 已实现 `KEV_CALIB`，但当时 base+普通模板下为负(44.7%) |
| DC-PMI | **arXiv:2104.08315** | **影响 argmax** | ❌ |
| NCC（标签级校准） | **arXiv:2511.14385** | **影响 argmax**，且对多 token 标签最相关 | ❌ |
| 选项顺序轮换 | **arXiv:2308.11483** | **影响 argmax**（抵消位置偏好） | ✅ 实现过（`KEV_ROTATE`），0.8B 上 choice +20pp / noul −20pp |
| SFC 消除（选项入 prompt + 1-shot） | **arXiv:2305.14596** | **影响 argmax** | ❌（只试过选项入 prompt） |
| Self-consistency / prompt 集成 | — | **影响 argmax**（多模板/多顺序投票） | 部分（17 模板扫描是手工的，非集成打分） |

**注意**：所有"只改概率数值不改 argmax"的方法（温度）在本任务上**无提分空间**（已实测）。**只有改变候选相对排序的方法才可能提分。**

---

## 8. Top-5 最值得在 kevcpp 上试的方法（按 收益/成本 排序）

> 判定标准：**零训练**（不更新任何参数）优先；能在现有 `probs_logits` / `probs_decide` 通道上做的不需后端扩展。

### Top-1 ★★★ 去污染相似度（rogue-dimension 修正 + 逐维 z-score）
- **方法**：读 hidden 后先做逐维去污染，再算相似度。
- **出处**：**arXiv:2109.04404**（rogue dimensions；standardization 纠正）、**arXiv:2104.07456**（z-score 是使用 contextualized 表示的必要步骤）、**arXiv:2511.11041**（R2 正交化优于 R1 减均值；depth=1 ABT ≈ R2）。
- **公式**：
  ```
  D = {h_decide, h_opt[0..K-1]}
  μ = mean(D); μ̂ = μ/||μ||;  v ← v − (v·μ̂)μ̂     for v in D     # R2
  # 或 z-score: v ← (v − mean_axis0)/std_axis0
  z[j] = cos(v_decide, v_opt[j])
  ```
- **预期收益**：**高**。它直接针对我们"四种公式同序"的病态；若相似度被 rogue dimension 支配，去污染后各公式才会分开，**这才让"选公式"这件事有意义**。文献在多个 embedding 任务上报告一致提升。
- **实现成本**：**极低**（`probs_decide` 内 5~10 行；无需后端扩展）。
- **零训练**：**是**。

### Top-2 ★★★ 逐选项隔离 decode（复刻 kev `option_isolation`）+ kev 结构 + 去污染
- **方法**：把每个选项**单独**作为一个分支 decode（只给它 state + 题干 + 自己），读该分支 `<|box_end|>` 位置 hidden，再与 `<|fim_suffix|>` decide hidden 匹配。
- **出处**：**kev 源码**（`kev/model.py` `branch_mask_batch` + `encode` 的 `option_isolation` 分支；语义为"选项只看到 state+instr+自己，共享 position ids → 置换不变"）；动机由 **arXiv:2308.11483**（位置偏好 13%~75%）与 **arXiv:2404.11122** 支撑。
- **公式**：与 kev 相同 `z[j] = <q(h_decide), k(h_opt[j])>`，但 `k` 用**隔离解码**得到的选项表示。
- **预期收益**：**高**。这是 kev 证明有效的**结构性**组件，且当前 kevcpp 的 `sum/mean` 逐候选 decode 路径**已经具备隔离语义**，只需把"读取位置"从"候选续写末尾"改成"选项 span 的 `<|box_end|>`"，并对 `<|fim_suffix|>` 读 decide 向量。
- **实现成本**：**低-中**（在 `probs_decide` 内改；**不需要自定义 attention mask**，用"逐候选单独 decode"近似 option_isolation）。
- **零训练**：**是**。

### Top-3 ★★☆ LM head 读取 + DC-PMI / 标签级归一化（不读 hidden，只读 logits）
- **方法**：在 decide 锚点读 LM head，对每个候选取**完整标签**的 logprob 序列，再用 DC-PMI / NCC 归一。
- **出处**：**arXiv:2104.08315**（DC-PMI）、**arXiv:2511.14385**（NCC，F1 +10%）、**arXiv:2305.14596**（SFC 形式化 + "选项入 prompt + 1-shot" 可消除 SFC）、**arXiv:2310.11732**（答案不确定性 vs 格式偏好两种不确定性）。
- **公式**：
  ```
  s(c) = log P(c | x, decide_anchor)
  s_pmi(c) = s(c) − log P(c | "N/A", decide_anchor)      # DC-PMI
  z[c] = s_pmi(c) / len_bytes(c)                            # 标签级长度归一(NCC 风格)
  ```
- **预期收益**：**中-高**。我们已实现别名池（combo 70.6%），DC-PMI/NCC 是它的**系统性升级**，直接针对"模型说否但 key 是 false"（格式偏好）与"label length bias"。另有 1-shot 消除 SFC 的简单法。
- **实现成本**：**低**（`probs_logits` 已有 logits 读取；需要"无内容输入"的一次额外 decode + 标签级归一）。
- **零训练**：**是**。

### Top-4 ★★☆ 中间层 / 多层拼接读取
- **方法**：读决定位与选项位的**中间层** hidden（或中层+末层拼接）后再匹配。
- **出处**：**arXiv:2104.07456** + **arXiv:2104.01767**（top+bottom 拼接优于单层）、**arXiv:2409.04185**（信息驻留层随 token 变）、**arXiv:2609.07876**（相邻层互相抵消 → 末层含"否决"后的结果）、**arXiv:2310.07325**（擦除使末层归因误导）。
- **预期收益**：**中-高**（对"决策"类任务，末层已被 LM 任务塑形，中间层可能保留更判别性的信息）。
- **实现成本**：**中-高**——**需后端扩展**：read_hidden 增加 layer 参数；需确认 vendored llama 是否能导出中间层 hidden（否则需改 vendored llama）。
- **零训练**：**是**。

### Top-5 ★☆☆ 反事实 / leave-one-out 归因式打分
- **方法**：逐个把选项从 prompt 移除（或替换），用 decide 位 hidden/logit 的**变化幅度**作为该选项的贡献分。
- **出处**：**arXiv:2309.16042**（activation patching 实践）、**arXiv:2606.03085**（多组件 causal tracing）、**arXiv:2301.04213**（**警示：定位≠有效干预**）。
- **公式**：`z[i] = ‖h_decide(x) − h_decide(x \ opt_i)‖`（或 logit 差）。
- **预期收益**：**不确定（中）**——机理上很干净（每选项独立贡献，不共享 softmax 分母），但 arXiv:2301.04213 提醒"能归因≠读得更准"。
- **实现成本**：**中**（K 次额外 decode；kevcpp 有逐候选 decode 基础）。
- **零训练**：**是**。

### 明确**不建议**尝试的方向（避免浪费）
- **全 PCA / ZCA 白化**：**arXiv:2407.12886** 明确"白化会让分类任务退化"；**arXiv:2511.11041** 也发现"全 PCA 白化在所有测试模型上都变差"。我们的任务就是分类/argmax。→ 只做 **R2 去均值方向 + depth=1 ABT**，不做全白化。
- **温度缩放**：**arXiv:1706.04599** 的 T 缩放不改变 argmax；**已实测** T=0.1 与 T=1.0 同为 60.0%。
- **纯 cosine 公式微调**（在未去污染的 hidden 上换公式）：我们已实测四种公式同序，**先做去污染再谈公式**。
- **自由生成/CoT 路线**：违背 system-one 决策模型设定（本项目已实证 0.8B 自由生成仅 56.5%）。

---

## 9. 特别回答：**不训练任何参数，哪种读取最可能逼近"学出来的双线性相似度"？**

### 结论
**最可能的是「去污染后的 hidden 匹配（Top-1）」+「逐选项隔离解码（Top-2）」的组合；其中去污染是必要前提，隔离解码是结构增益的来源。**
若只能选一个先做：**Top-1 去污染相似度**（成本最低、命中病态根因）。

### 理由
1. **数学上，"学出来的双线性相似度"在做什么**：`z[o] = <Wq·h_decide, Wk·h_opt[o]>`。Wq/Wk 是一个**学习到的投影对**，作用是**选出残差流里"判别相关"的那个子空间**，并把各选项映射到可比较的 256 维空间。
2. **零训练时唯一"免费"且"已针对该任务训练过"的投影是 LM head 本身**（`hidden → vocab`）。它在预训练中就被塑形为"把下一 token 的语义方向暴露出来"。kev 的 `PLAN.md` 自述"adapted backbone 经 LM head 读出的成绩 = pointer"，**正是"同一个 hidden、两个不同的投影器、指向同一子空间"的证据**。
3. **但 LM head 读出的语义方向被"语言化"了**：它给出的是"下一个 token 是什么"，而不是"哪个选项对"。于是它受 **surface form competition（arXiv:2104.08315）** 与 **label length bias（arXiv:2511.14385）** 干扰 → 需要 **DC-PMI/NCC**（Top-3）来扣掉"格式偏好"。
4. **而 rogue dimension 污染（arXiv:2109.04404）会同时压制"任意相似度公式"的表达力**——我们实测四公式同序，正是这一病态。所以**任何相似度式读取（含 LM head 的隐式版本）都必须先去污染**。
5. **隔离解码（option_isolation）** 是 kev 在**已微调 backbone** 上启用的**零训练结构**（我们从 kev 源码读到了确切语义）。它同时消解"位置偏好"（arXiv:2308.11483）并给出置换不变的选项表示——**这是我们能在不训练的前提下抄到的、最接近 kev 结构红利的东西**。

### 综合公式（建议实现）
```
# 阶段 1（读取）: 每个选项单独 decode(只含 state+instr+该选项) → h_opt[j] @ <|box_end|> 位置
#                 整体 decode(state+instr+全部选项+<|fim_suffix|>) → h_decide @ <|fim_suffix|> 位置
# 阶段 2（去污染）: D = {h_decide, h_opt[0..K-1]}
#                   μ̂ = normalize(mean(D));  v ← v − (v·μ̂)μ̂   (R2, arXiv:2511.11041)
#                   可再叠加逐维 z-score (arXiv:2104.07456)
# 阶段 3（匹配）: z[j] = cos(v_decide, v_opt[j])      # 或与 LM-head 读数做 ensemble
# 阶段 4（后处理）: z[j] −= log P(opt_j | "N/A")       # DC-PMI (arXiv:2104.08315) 抵消格式偏好
```
**零训练**（阶段 2/3/4 全是无参数后处理；阶段 1 是结构选择）。
**仍可能达不到 82.4%**，因为 kev 的 Wq/Wk 里还有"**针对该任务学到的子空间选择**"，这一部分**原则上无法从零训练复刻**；但按文献，阶段 2 修复的是"表示被 rogue dimension 淹没"这一**真实且已实测存在**的障碍，因此是**当前最有希望的一步**。

---

## 10. 落地清单：kevcpp 现有能力 vs 需后端扩展

### 现有代码可**直接**做（不需后端扩展）
| 方法 | 落点 | 备注 |
|---|---|---|
| Top-1 去污染相似度 | `probs_decide`（`kev_model.cpp` 内已有 centroid/`cos_` 基础设施） | 加 5~10 行；新增 formula 分支如 `dc2`(R2)/`zc`(z-score) |
| Top-2 逐选项隔离 decode | `probs_decide` + 复用 `sum/mean` 路径的逐候选 decode | 改"读取位置"为 `<|box_end|>`；需 kev 结构模板（已在 `probs_decide` 内改过） |
| Top-3 DC-PMI / NCC | `probs_logits`（已有 logits 读取 + `KEV_CALIB` 骨架） | 加"无内容输入"的一次 decode + 标签级归一 |
| Top-5 反事实打分 | `probs_logits`/`probs_decide` 逐候选 decode | 需保存 baseline hidden 做差 |
| 选项顺序轮换去位置偏好 | 已有 `KEV_ROTATE`/`probs_label` | 复用 |

### 需**后端扩展**
| 能力 | 现状 | 扩展点 | 成本 |
|---|---|---|---|
| 读**中间层** hidden | 只有末层（`llama_get_embeddings_ith`） | `kev_backend_*.cpp` 的 `read_hidden` 增加 layer 参数；需确认 vendored llama 能否导出中间层（否则要改 vendored llama） | **中-高** |
| 词嵌入表（token embedding） | 后端无此接口 | `llama_model_get_tensor`（GGUF `token_embd.weight`）→ backend 暴露 | 中 |
| **自定义 attention mask**（真 option_isolation） | `llama_set_causal_attn` 只有全局开关 | 需 vendored llama 增加 per-token mask 构造 | **高**（故推荐用"逐候选单独 decode"近似） |
| attention 权重输出 | 不导出 | 需 graph 层改动 | **高**（不建议） |

---

## 11. 引用清单（均经 arXiv API 核对）

| arXiv id | 标题 | 用途 |
|---|---|---|
| 2109.04404 | All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality (EMNLP 2021) | §1.1 rogue dimensions 病态 + standardization 解法 |
| 2104.07456 | Effect of Post-processing on Contextualized Word Representations (COLING 2022) | §1.2 z-score / ABT / token 平均 / 跨层拼接 |
| 2511.11041 | Correcting Mean Bias in Text Embeddings: A Refined Renormalization with Training-Free Improvements on MMTEB | §1.2 R2 正交化；全 PCA 白化有害 |
| 2407.12886 | Whitening Not Recommended for Classification Tasks in LLMs | §1.2 分类任务勿白化 |
| 2104.01767 | WhiteningBERT: An Easy Unsupervised Sentence Embedding Approach | §1.2 token 平均 / top+bottom 拼接 |
| 2011.05864 | On the Sentence Embeddings from Pre-trained Language Models (BERT-flow, EMNLP 2020) | §1.2 各向异性 |
| 2305.17746 | Whitening-based Contrastive Learning of Sentence Embeddings (ACL 2023 Oral) | §1.2 whitening/alignment 框架 |
| 2104.08315 | Surface Form Competition: Why the Highest Probability Answer Isn't Always Right (EMNLP 2021) | §3.1 DC-PMI |
| 2305.14596 | Increasing Probability Mass on Answer Choices Does Not Always Improve Accuracy (EMNLP 2023) | §3.1 SFC 形式化 + "选项入 prompt + 1-shot" |
| 2310.11732 | Investigating Uncertainty Calibration of Aligned Language Models under the Multiple-Choice Setting | §3.1 答案不确定性 vs 格式偏好 |
| 2511.14385 | Mitigating Label Length Bias in LLMs (AACL 2025) | §3.1 NCC 标签级校准 |
| 2102.09690 | Calibrate Before Use: Improving Few-Shot Performance of Language Models (ICML 2021) | §3.1 Contextual Calibration |
| 1706.04599 | On Calibration of Modern Neural Networks (ICML 2017) | §3.1 温度缩放（不改 argmax） |
| 2308.11483 | Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions | §3.1/§4.2 位置偏好 13~75% + rotate |
| 2404.11122 | Small Language Models are Good Too (LREC-COLING 2024) | §3.1 小模型零样本分类 |
| 2309.16042 | Towards Best Practices of Activation Patching in Language Models (ICLR 2024) | §5 activation patching 方法学 |
| 2606.03085 | Multi-component Causal Tracing in Large Language Models (ACL 2026) | §5 多组件追踪 |
| 2301.04213 | Does Localization Inform Editing? (NeurIPS 2023 Spotlight) | §5 定位≠有效干预（警示） |
| 2305.12289 | Revisiting the Architectures like Pointer Networks to Efficiently Improve the Next Word Distribution... (ACL Findings 2023) | §4.1 softmax bottleneck / pointer |
| 2406.04156 | Pointer-Guided Pre-Training (ECML-PKDD 2024) | §4.1 pointer 用于分类 |
| 2409.04185 | Residual Stream Analysis with Multi-Layer SAEs (ICLR 2025) | §2.1 信息驻留层随 token 变 |
| 2609.07876 | LLM Layers Immediately Correct Each Other (NeurIPS 2025) | §2.1 层间抵消 → 末层含"否决" |
| 2310.07325 | An Adversarial Example for Direct Logit Attribution (BlackboxNLP 2024) | §2.1 擦除使末层归因误导 |
| 2304.06962 | Prompt Engineering and Calibration for Zero-Shot Commonsense Reasoning | §3.1 校准与 prompt 工程交互（负交互） |
| 2406.14900 | Decoding Matters: Addressing Amplification Bias and Homogeneity Issue for LLM-based Recommendation (EMNLP 2024) | §3.2 length normalization 的"ghost token"放大偏差 |

**未核实**：byte-length normalization 的原始出处（lm-eval-harness 系，未找到确切 arXiv 号，报告中标注为"未核实"）；"Attention is not Explanation" 的确切 arXiv 号未核实（仅作领域共识引用）。
