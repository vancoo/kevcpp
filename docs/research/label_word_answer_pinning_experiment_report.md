# 实验报告 — 选项标签化 + 答案位锁死（零训练，KEV_ANSWERLABEL=num|yn）

> 调研分支：`research/logits-decision-correctness`
> 动机（源自用户两个观点）：(1) 不严格限定时 `Answer:` 后首 token 可能被解释/铺垫污染、与答案无关；
> (2) 比对的对象应是"选项（稳定标签）"而非"选项内容（description）"。
> 实现：`src/kev_model.cpp` 新增 `KEV_ANSWERLABEL` 开关，把 prompt 改成带编号/YesNo 的强约束形式，
> 并把打分候选 token 从 `q.keys[j]`（选项内容/别名）换成稳定标签 token。零训练，默认 off。
> 数据：`tests/bench-60cases.jsonl`（85 题），模型 `models/Qwen3.5-0.8B-Q8_0.gguf`，`--backend cpu --ctx 2048`。

---

## 1. 结论（TL;DR）

**"选项标签化 + 答案位锁死"没有超过 plain mean，且数字标签显著变差。** 唯一仍然成立的零训练增益仍是 `aggregation=mean`（54.1%）。这三点用户观点在 0.8B 上「方向对、但补不回语义」，且比内容更稳的标签在此模型上反而更弱。

| 实验 | accuracy | score | choice | noul |
|---|---|---|---|---|
| (a) 基线 first | 51.8% (44) | 40% (2/5) | 45% (18/40) | 60% (24/40) |
| (b) **mean**（已知最优） | **54.1% (46)** | 40% (2/5) | **50% (20/40)** | 60% (24/40) |
| (c) label_num + first | 38.8% (33) | 60% (3/5) | 25% (10/40) | 50% (20/40) |
| (d) label_num + mean | 38.8% (33) | 60% (3/5) | 25% (10/40) | 50% (20/40) |
| (e) label_yn（noul Yes/No）+ mean | 54.1% (46) | 40% (2/5) | 50% (20/40) | 60% (24/40) |

---

## 2. 两个观点的落实方式

### 观点 1：锁死"答案位"，避免首 token 与答案无关
- 默认：`{instructions}\n[ {options} ]\n{state}\nAnswer:`，`Lp-1` 处直接读候选首 token logit，模型若先给解释/铺垫就被浪费。
- 标签化：把 prompt 改成强约束"只输出编号/YesNo"，并把打分候选 token 换成**稳定标签 token**（`"1".."K"` / `"No"`/`"Yes"`）。即使模型不按严格格式走，logits 通道只读 `Lp-1` 处各标签 token 的 logit 再 softmax，不解析文本。

### 观点 2：比对"选项"而非"选项内容"
- 默认：`cand_tok[j]=tok(q.keys[j])`，比对的是 choice 的 **key**（`投资`/`并购`…）、noul 的 `["false","true"]`、score 的 `["0"..]`——即"选项内容/别名"。
- 标签化：`cand_tok[j]=tok(label_strs[j])`，比对的是**稳定标签**（数字编号或 `Yes/No`）。probs 顺序仍按原候选索引 0..K-1，`to_answers` 的 key 映射不变。

---

## 3. 为什么标签化没用（甚至数字标签崩）

1. **数字标签对 0.8B 没有判别力（-13pp）**：`label_num` 无论 first/mean 都掉到 38.8%。机理：模型在 `Answer:` 处对 `"1","2","3"` 的 next-token 概率基本由**位置/序号先验**决定（首项/靠前项偏高），与"哪个选项内容正确"无关——恰好落入 2308.11483 说的 label/position bias，且数字是极泛 token、丢弃了选项内容的语义关联。choice 从 45%→25% 即此。
2. **`Yes/No` 与 `false/true` 无差异（0pp）**：`label_yn` 全等于 plain mean（54.1%，noul 60%、choice 50%、score 40%）。即**"换 label 词"不改变** 0.8B 的 noul 判别——它读不懂否定/反义时，无论标签是 `Yes/No` 还是 `true/false` 都一样答错（always-yes 依旧）。
3. **共性**：0.8B 的瓶颈仍是**内容语义理解**，不是"首 token 被污染"或"标签不稳定"。锁死答案位 / 换稳定标签只改变"打分对象"，改变不了模型对题干语义的把握。

---

## 4. 结论与建议

- **仍建议**：`aggregation=mean`（候选完整序列 log-prob 平均）是零训练里唯一净正、最稳的改进（54.1% vs 51.8%，全来自 choice）。这是比"改标签"更有效的"用足选项内容"。
- **不采用**：数字/字母标签化（KEV_ANSWERLABEL=num）——引入位置/序号偏见且丢语义，在 0.8B 上负收益。`Yes/No` 标签对 noul 无益也无害。
- **方向反思**：`mean` 已把"比对选项内容"用足（完整序列），标签化则走向"抛内容只比标签"——实测证明**相反方向**（保留内容+整序列）才是对的。
- **下一步更可能有效的零训练杠杆**（by 前排实验的横向结论）：choice 上做**选项顺序轮换+投票**（2308.11483 位置偏置最大达 13–75%；`mean` 增益恰在 choice），或 **RAG** 把判定规则检索进 prompt（小模型+检索 85–95%）。二者都不改变"用足选项内容"这一已被证明正确的取向。

---

## 5. 复现

```
# label_num + mean
$env:KEV_ANSWERLABEL="num"
build/bin/Release/kevcpp.exe --model models\Qwen3.5-0.8B-Q8_0.gguf --backend cpu --port 18110 --ctx 2048 --quiet --decide-config config\decision.agg-mean.json
python build\correctness_harness.py --port 18110 --model x

# label_yn + mean
$env:KEV_ANSWERLABEL="yn"

# plain mean
(不设 KEV_ANSWERLABEL) + --decide-config config\decision.agg-mean.json
```
产物：`build/correctness_0.8B-{labelnum-mean,labelnum-first,labelyn-mean,plain-mean}.json`。

---

## 6. 证据索引
- 实现：`src/kev_model.cpp` `probs_logits`（`KEV_ANSWERLABEL` 分支；候选 token 切换；label 模式下关前缀缓存）。默认 off，行为不变。
- baseline `first`/`mean` 来自近期同平台实测（51.8% / 54.1%）。
- 理论：arXiv 2308.11483（选项顺序/标签偏置 13–75%）、2505.10772（排序投票）、2501.17338（完整序列 log-prob 优于首 token）。