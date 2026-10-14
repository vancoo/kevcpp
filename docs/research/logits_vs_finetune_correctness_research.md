# 调研报告 — kevcpp「logits 决策」vs「kev 微调」：为什么微调能提升正确率，kevcpp 如何在不训模型前提下提升正确性

> 调研分支：`research/logits-decision-correctness`（本文档随分支提交）
> 调研对象：`kevcpp`（本仓库，C++/llama.cpp 决策服务）与 `kev`（`E:\van\projects\kev`，Python/torch 决策模型 + 微调）
> 范围：只调研 + 设计建议；不改动主分支决策代码。文档落在 `docs/research/`。

---

## 0. 结论速览（TL;DR）

**你的问题「微调只是让输出格式正确了，为什么还能提高决策正确率」——这个前提是错的。** kev 项目的微调**不是**在修"输出格式"：kev 决策压根不走文本生成，它**不生成 token**，所以不存在"格式"这一环。它微调的是**一个独立的、可学习的打分网络（PointerHead）**，用**交叉熵直接优化"选项打分"这个决策目标**，并顺带把梯度传回主干（LoRA / 全量），让主干隐层表示变得"和选对选项对齐"。这才是它正确率高的机制根源。

kevcpp 目前的"logits 机制"是**零样本读基础模型词汇表 next-token logits**（aggregation=first 时只取每个候选的**首 token**），没有任何学习也没有 p>0 的校准，所以准确率被锁在 ~52.9%–68.2%（见 `correctness_benchmark_report.md`）。二者的差距本质是：**一个打分器有被梯度塑形的"选对的几何"，另一个是冻结模型的"续写先验"，两者与回答正确性的相关性不在一个量级。**

**kevcpp 在不重新微调模型的前提下**，不能凭空补回"训出来的对齐"，但可以用下列（按性价比排序）机制显著兜底：

1. **P0 复用已训好的 kev 权重**——仓库里已有 `models/head.bin`（指针头，d=1024→dp=256，T=2.35）与 `models/kev-merged-q8.gguf` / `kev-lora-qwen35.gguf`（LoRA 融合后的 0.8B 决策主干）。旧版 kevcpp 的 `kev_head.cpp` **本就忠实实现了 PointerHead**（是 `kev/model.py::PointerHead` 的 C++ 直译），只是 `feature/logits-primary-nokev` 分支为了模型无关性把它下线了。把"hidden 读取 + PointerHead"路径重新接回（与 logits 通道并存、可开关），是**正确率提升最大、成本最低**的一招。
2. **P1 候选顺序轮换 + 多数/排序投票**：现有研究已证（`research_logits_discrimination_report.md`），同一题轮换选项顺序 r 次、对 argmax 投票，在低置信/小模型上可抬几个百分点；kev 训练里的 `permutation_kl` 也正是为了压制这种位置偏置。
3. **P1 多 token 候选的完整 log-prob 聚合**：中文候选多为多 token，`aggregation="first"` 只比首 token 会失真；改用 `sum`/`mean` 或归一化 log-prob。
4. **P2 logits/选项层 debias + 置信度校准门控**：temperature scaling 不改变 argmax、只改置信度，但能配合"低置信→不判定/降级/交人工"的安全策略；`position/label` 偏置向量扣除（仿 2509.16805）。
5. **P3 若允许训模型**：最小成本是 LoRA(r=16 框架已有) 训主干 + PointerHead 从零练，也就是 `kev` 那套；kevcpp 可长期走 `train.py` 产出的 `head.pt`+LoRA 再转 GGUF 的管线。

一句话决策建议：**"零微调 + 零 head 直接用开源 GGUF 的 logits/KEV 切半点积出高正确率决策"在机制上不成立**（`PLAN_B_design.md` 的 V3 假设未证、`FEASIBILITY_headless_gguf_research.md` 已明确）；要正确率，**要么复用一个已有决策头/open 权重，要么接受"兜底机制"的有限提升**，真正的高正确率只能来自"训过"（你的 kev-0.8b 实现在 0.85 documents / 0.66 hard-v1，远超零样本 logits）。

---

## 1. kev 项目到底在微调什么？—— 核心纠正

### 1.1 决策输出 = PointerHead 的"打分"，不是文本生成

从 `E:\van\projects\kev\kev\model.py`：

```python
class PointerHead(nn.Module):
    def __init__(self, d, dp=256):
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)
        self.temperature = 1.0
    def forward(self, h_decide, h_opts):   # h_decide[d], h_opts[K,d] -> logits[K]
        z = (self.k(h_opts) @ self.q(h_decide)) * self.scale
        return z / self.temperature  # (eval 时)
```

- 模型的输入是"状态文本 + 每问的选项"，输出是 **K 个选项的打分 logits**，再 softmax 成概率。
- **全程不生成一个 token**。`DecisionModel` 注释明确："backbone only (no vocab head): we never generate text."
- 打分 = 双线性匹配：把"`<decide>` 位置"的隐层 `h_decide` 经 `W_q` 投影成 query，把"每个选项 marker 位置"的隐层 `h_opts` 经 `W_k` 投影成 key，点积 + `scale`。**哪个选项的点积最大，就选哪个。**

### 1.2 训练目标：对"选项打分"直接做交叉熵

从 `E:\van\projects\kev\kev\train.py`：

```python
def question_loss(z, q, ...):                    # z = PointerHead 打分
    ...
    y = torch.tensor([q["label"]], device=dev)
    loss = F.cross_entropy(z[None], y, label_smoothing=label_smoothing)   # ← 决策 CE
    if q["qtype"] == "score" and ord_w > 0:      # 评分题有排序惩罚 NRPS
        p = F.softmax(z, -1); ... loss += ord_w * (p.cumsum(-1)[:-1] - observed_cdf).square().mean()
def batch_loss(...):
    ce = sum(question_loss(z, q, ...) for ...)   # 每问 CE 求和
    ...
    kl = permutation_kl(...)                     # 选项顺序翻转的一致化(压位置偏置)
```

- **loss 是 `CrossEntropy(PointerHead(z), 正确选项标签)`**，并可选叠加：排序惩罚（NRPS，score 题）、锚定 KL（对齐冻结基座分布）、**排列 KL（`permutation_kl`，惩罚"换选项顺序就变预测"，这正是压 position/label bias）**。
- **关键**：这个 CE 的梯度经 `PointerHead` 一直**反传到主干**（LoRA 或全量权重，`full_ft.py` 全量训练、`train.py` LoRA）。也就是说，主干隐层在"`<decide>`/选项位置"表示的**方向**会被梯度推向"`q(decide)·k(选项)` 对正确选项最大"。**这是"微调改变 backbone 表示以保证决策正确"的机制根源。**

### 1.3 模型卡的确凿数据（`kev\docs\model-cards\kev-0.8b.md`）

- Kev-0.8B = "a LoRA adapter (r=16, 11.3M trainable params) plus a pointer head on Qwen/Qwen3.5-0.8B-Base"，`task_type="FEATURE_EXTRACTION"`，**无文本生成**。
- 零样本基线(未微调) → 微调后：documents **0.608 → 0.851**，hard-v1 **0.396 → 0.665**，devtools **0.472 → 0.637**，out-of-domain 0.684 → 0.697。
- 自带 `option-order flip rate`(0.08) 与拟合温度 `T=2.35`。

> **这就是对照**：kevcpp 的零样本 logits 通道 ≈ kev 模型卡里的"未微调基线"一列（0.6/0.4 上下，正是 52–68% 那个量级）；kev 微调后抬到 0.85/0.66。差距就在"有没有训练过决策打分目标"。

---

## 2. kevcpp 当前的"logits 机制"到底是什么、局限在哪

### 2.1 机制（`src/kev_model.cpp::probs_logits`）

- 对每问，用 `prompt_template="{instructions}\n[ {options} ]\n{state}\nAnswer:"` 组一个**填空式 prompt**（模型无关）。
- 把每个候选 key **作为续写答案 token**：`aggregation="first"` 时只取候选的**首 token id**，在最后位置读 `lm_head` 给该 token 的 **logits**（`read_logits`），对全体候选做相对 softmax（可选 `/temperature`）。
- **读的是基础模型词汇表的 next-token logits**——即 `attention + FFN + lm_head(W vocab)` 输出，不是隐层，也没有 PointerHead。`bopt.lm_head = 1, embeddings = 0`（`kev_model.cpp:87-88`），也就是只取 lm_head 分类概率，不取隐层。

### 2.2 局限（实测 + 机制层面）

准确率（`correctness_benchmark_report.md`，Vulkan，85 条）：0.8B 52.9% / 4B 56.5% / 9B 67.1% / 35B-q3.6 **68.2%**。共性强项是多候选、弱项是 score 中档塌缩、小模型"always-yes"等。

机制层面的根因（与 `research_logits_discrimination_report.md` 一致）：

1. **没有训过的"选对"几何**：基础模型从没按"`state↔选项` 点积越大越对"优化过，它的 next-token 先验只保证"续写更通顺"，不保证"选谁更对"。当候选间 logits 接近时，直接退化成位置/顺序/频率先验。
2. **只取首 token（`aggregation="first"`）**：中文候选常多 token，首 token 往往无独立语义，张量切分不一致 → 判别失真。`sum`/`mean`（读全候选 token log-prob）在配置里已具备，但默认是 `first`。
3. **position/label bias**：选项顺序、字母标签(A/B/C)、内容位置都会倾斜高 logits 候选；小模型信噪比低、受影响最大（kev 用 `permutation_kl` 专门压制）。
4. **无校准**：`temperature` 默认 1.0，即使配了也只改置信度不改 argmax；没有一个"低置信就不判"的门控，容易把低置信也硬答出去（小模型 always-yes 的来源之一）。

> 重要概念区分（避免混淆，见 `PLAN_B_design.md` §5.3、`FEASIBILITY_headless_gguf_research.md`）：
> - **kev PointerHead**：**有学习参数**（`W_q`,`W_k` 1024→256 + T），是"决策本体"。
> - **llama.cpp KEV 切半点积**（`server-context.cpp`）：**无学习权重**，隐层对半 q/k 直接点积/√n，是给"专为 KEV 训练过的模型"用的；对未训开源基座是定义外用法（`PLAN_B_design.md` V3 未验证）。
> - **kevcpp 现 logits 通道**：第三种，读词汇表 next-token logits。三者**不是一回事**。

---

## 3. 为什么"微调只改格式也能提正确率"是伪命题，正确机制是什么

澄清链条：

| 说法 | 对 / 错 | 解释 |
|---|---|---|
| 微调只是把输出格式修对 | **错** | kev 决策**不生成文本**，无格式可言；微调优化的是"选项打分 CE"。 |
| 微调通过改隐藏表示来提正确率 | **对** | CE 梯度反传主干 → `<decide>`/选项位置的隐层方向被塑造成"对正确选项 q·k 最大"。 |
| 微调还顺带校准概率 | **对** | 训后拟合 temperature 存进 `head.pt`，serving 时按 T 除 logits，改善 Brier/ECE/置信度（不改变 argmax）。 |
| 微调学到的能力是 task-specific | **对** | 所以 out-of-domain 提升有限(0.684→0.697)，domain 内大(documents +24pp)——**决策正确率提升本质是"对这套题型学出对齐"**。 |

一句话机制：**微调让"打分器"和"主干表示"一起朝"回答正确最大化"走**。零样本 logits 通道没有这步，所以只有基础模型"碰巧能用"的续写先验，正确率自然上不去——这与你观察到的"kevcpp 用 logits 准确率一般"完全自洽。

---

## 4. kevcpp 可不训模型就提升正确性的机制（按性价比）

> 目标层：不改模型权重、不动 main 分支决策主线，增量补正确性。所有方案都建议在**独立分支/开关**下做、跑 `tests/bench-60cases.jsonl` A/B 量化。

### P0（推荐，正确率提升最大）——重新启用已训好的 kev PointerHead 通道

- 仓库已具备成品：`models/head.bin`（KEVH，d=1024→dp=256，**T=2.35**，正是 kev-0.8b 的 fitted head）、`models/kev-merged-q8.gguf` / `kev-lora-qwen35.gguf`（LoRA 融合 0.8B 决策主干）、`models/build_real_head_bin.py`（head.pt→head.bin）。
- 旧版 `src/kev_head.h/.cpp` 是 `kev/model.py::PointerHead` 的**忠实 C++ 直译**（`k(h_opts)·q(h_decide)*scale`，`load_from_file` 读 `head.bin`），git 里可回溯（`b4fccf4` 首版）。`feature/logits-primary-nokev`（`c160dfd`）才把它下线。
- **落地**：fetch 回 `kev_head` + `backend_->read_hidden` 读取路径；`main_server` 加 `--decide-mode head|logits|auto`；默认/推荐 `auto`（有 head.bin → 走 PointerHead，否则回退 logits）。成本=把写过一次的实现接回来，收益=直接来到 kev-0.8b 的 0.85/0.66 档。
- 若想纯 logits 之上升级，且当前模型是**融合了 kev LoRA 的主干**（`kev-merged-q8`），那 lm_head logits 里已带上训练信号，比裸基座好很多——**先确认当前基准用的是哪一个 GGUF**（`models/Qwen3.5-0.8B-Q8_0.gguf` 裸基座 vs `kev-merged-q8.gguf`）。

### P1 —— 候选顺序轮换 + 投票，与多 token 完整聚合

- 候选把 prompt 选项 **顺序轮换 r 次**（默认 r=3），各跑一次 argmax，**多数投票**（或排序投票 instant-runoff/Borda），票数分散＝低置信（见图：`research_logits_discrimination_report.md` §1/§5 的 2308.11483、2203.11171、2505.10772）。
- 把 `aggregation` 默认从 `first` 测成 `sum`/`mean`（候选完整 token log-prob 求和/归一），对多 token 中文候选更稳。
- kev 侧 `permutation_kl` 说明"顺序一致性"对正确率是真杠杆。

### P2 —— logits 去偏 + 温度校准 + 置信度门控

- start：拟合一个**校准温度**（拿一小批基准基准，最小化 Brier/ECE，不改变 argmax），得到可信的 probability。
- 用"低置信→不答/降级/交人工"的安全门：`coverage@≤5% error` 就是 bu解siness 指标。
- 高级：选项 token **位置/字母偏置向量**扣除 + confidence-adaptive（仿 2509.16805），它们正好对应"多候选弱、小模型分不开"的实测短板。

### P3（允许训）——最小复刻 kev 微调管线

- 若最终正确率要靠微调：用 `kev train.py`/`kev-finetune`：LoRA(r=16) 训主干 + PointerHead 从零，1 H100 一次 $1–3；产物 `head.pt`+LoRA → `build_real_head_bin.py` + merge 转 kevcpp 可载的 `head.bin`+`*q8.gguf`，喂给 P0 的 head 通道。

---

## 5. 结论性建议

1. **先回答"要正确率还是要模型无关"**：若只要正确率，**P0 直接把已训好的 kev head/融合主干接回来**，立得 0.85/0.66 档；kevcpp 旧实现已证明可行，工程量最小。
2. **若坚持不碰训过的权重**：P1+P2 是零训练、纯在 logits 通道上的有限提升（实测该在 52–68% 之上再抬几个点），但**无法到微调档**。
3. **"零 head + 零训练直接用开源 GGUF 高正确率决策"在机制上不成立**，不要再投入验证（`PLAN_B_design.md` V3 / `FEASIBILITY_headless_gguf_research.md` 已结论）。kevcpp logits 通道作为"模型无关兜底"保留即可，作为"高正确率决策主通道"方向不对。

---

## 6. 证据索引

- kev 打分/训练：`E:\van\projects\kev\kev\model.py:182-200`(PointerHead)、`:303-371`(`_readout`)、`E:\van\projects\kev\kev\train.py:41-77`(loss)、`:342-365`(batch_loss)、`:576-577`(head 独立 lr)。
- kev 成果：`E:\van\projects\kev\docs\model-cards\kev-0.8b.md`（0.608→0.851 等，T=2.35，LoRA+16+pointer head）。
- kevcpp logits 通道：`src/kev_model.cpp:327-534`(`probs_logits`)、`src/kev_config.h:44`（aggregation 默认 first）、`kev_model.cpp:87-88`（lm_head=1/embeddings=0）。
- kevcpp 旧 PointerHead：git `b4fccf4:src/kev_head.h`（`kev/model.py::PointerHead` 直译）。
- 正确性基准：`docs/research/correctness_benchmark_report.md`（52.9–68.2%）。
- 正确性提升调研：`research_logits_discrimination_report.md`（顺序偏差/投票/去偏/温度）。
- 概念边界：`docs/research/PLAN_B_design.md`（KEV 切半点积=无权重，V3 未验证）、`docs/FEASIBILITY_headless_gguf_research.md`（PointerHead=决策本体）。

> 本调研只读 + 设计建议，未改主分支决策代码；相关改动/开关请在独立代码分支上进行并跑基准 A/B。