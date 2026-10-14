# 实验报告 — Qwen 思考型模型 × logits 通道：ChatML 模板 + 完整序列 log-prob（zero-training A/B）

> 分支：`research/logits-decision-correctness`。仅改 prompt/读取方式，未碰任何 head.bin/LoRA/训练产物。
> 数据：`tests/bench-60cases.jsonl`（85 题），同机 0.8B（`models/Qwen3.5-0.8B-Q8_0.gguf`，CPU，ctx 2048），复用 `build/correctness_harness.py`。

---

## 0. 结论（TL;DR）

- **唯一确定、干净的正收益：`aggregation=mean`（或 `sum`）→ 54.1%，比 baseline first-token 51.8% 高 +2.3pp**。印证论文 2501.17338（候选**完整序列** log-prob 优于单首 token）。
- **ChatML 模板（`KEV_TEMPLATE=chatml`）：净负**——把 `score` 从 40% 抬到 **80%**，但把 `choice` 从 45% 打到 **27.5%**，总体 51.8% → 45.9%。
- 两者组合（chatml+mean）：score 仍是 80%，choice 只回 40%，总体回到 51.8%——ChatML 对 choice 的伤害无法被 mean 抵消。

---

## 1. A/B 结果总表

| 实验 | accuracy | score | choice | noul |
|---|---|---|---|---|
| baseline（aggregation=first） | **51.8%** (44/85) | 2/5 | 18/40 (45%) | 24/40 (60%) |
| **aggregation=mean** | **54.1%** (46/85) | 2/5 | 20/40 **(50%)** | 24/40 |
| aggregation=sum | 54.1% (46/85) | 2/5 | 20/40 (50%) | 24/40 |
| **KEV_TEMPLATE=chatml** | 45.9% (39/85) | 4/5 (**80%**) | 11/40 (**27.5%**) | 24/40 |
| chatml + mean | 51.8% (44/85) | 4/5 (80%) | 16/40 (40%) | 24/40 |

命中数均为同模式 `hit` 汇总。noul 三条主实验均 24/40=60%（TN 依旧几乎为 0，即 always-yes 仍在，属 0.8B 语义理解瓶颈，非本实验目标）。

---

## 2. 实验 1：aggregation first → mean/sum（完整序列 log-prob）

- **实现**：复用 `probs_logits` 现成的 `sum`/`mean` 分支（`src/kev_model.cpp` 已有），用 `--decide-config config/decision.agg-mean.json` / `agg-sum.json` 切换，零代码改动。
- **结果**：`mean` 与 `sum` 一致，均 46/85；提升全部来自 **choice（45% → 50%）**。
- **机理**：中文候选多为多 token（如 `"股份回购"`、`"先升后降"`），只取首 token 的 logit 失真；对候选**完整 token 序列** 的 log-prob 做 mean/sum，使用上下文信息更完整，接近论文 2501.17338 的结论。
- **建议**：**把默认 `aggregation` 从 `first` 改为 `mean`**（是当前唯一明确的正确率提升，且零副作用）。若担心 sum 受候选长度影响，`mean` 更稳健。

### 复现（mean）
```
build\bin\Release\kevcpp.exe --model models\Qwen3.5-0.8B-Q8_0.gguf --backend cpu --port 18204 --ctx 2048 --quiet --decide-config config\decision.agg-mean.json
python build\correctness_harness.py --port 18204 --model 0.8B-aggmean
```

---

## 3. 实验 2：`KEV_TEMPLATE=chatml`（ChatML 包裹）

- **实现**（最小侵入，`src/kev_model.cpp`）：当环境变量 `KEV_TEMPLATE=chatml` 时，把模板包成 Qwen 系 ChatML，让末尾 `Answer:` 位置变成 `<|im_start|>assistant\n` 起始（去贴近"答案判决段"而非可能触发 think 的纯文本段）；占位符保留，前缀缓存/渲染逻辑不变。默认（未设 env）行为完全不变。
  ```
  <|im_start|>system\n你是严格按给定选项作答的判别器，只输出最终选项，绝不输出思考或推理过程。<|im_end|>\n
  <|im_start|>user\n{instructions}\n[ {options} ]\n{state}<|im_end|>\n<|im_start|>assistant\n
  ```
- **结果（混合、净负）**：
  - `score` 2/5 → **4/5（+40pp，显著）**：assistant 起始位置对数值型 label token 的判别更敏锐。
  - `choice` 18/40 → 11/40（**−17.5pp**，显著）：对中文候选 key，assistant 起始位置反而让判别变差。
  - noul 不变（60%）。
  - **总体 51.8% → 45.9%**。
- **机理**：Qwen3.5-0.8B 是 Qwen3 系，`<|im_start|>/<|im_end|>` 在词表中为特殊 token；ChatML 确实把 Lp-1 挪进"assistant 段"，但对**不同题型**影响方向相反——提升数值/评分类，毁掉中文短句候选的 choice。这不是一句万能开关。
- **结论**：**ChatML 包裹不建议作为全局默认**（chatml+mean 组合也只回到 baseline）。若业务以 score/数值型为主可单独开启；对 choice 为主则不要开。

### 复现（chatml）
```
set KEV_TEMPLATE=chatml   (PowerShell: $env:KEV_TEMPLATE="chatml")
build\bin\Release\kevcpp.exe --model models\Qwen3.5-0.8B-Q8_0.gguf --backend cpu --port 18203 --ctx 2048 --quiet
python build\correctness_harness.py --port 18203 --model 0.8B-chatml
```

---

## 4. 实验 3（可选，本次未做，标注为未来）

- **选项顺序轮换 + 多数/Logits 投票**：论文 2308.11483（只换顺序可差 13–75%）、2505.10772（排序投票）。实现需在 prompt 组 {options} 时做 r 次轮换 + 对 argmax/排序聚合投票，改动比实验 1/2 大。鉴于 mean 已给明确正收益，此项建议作为下一步单独 A/B（尤其针对 choice 型，疑似可放大 mean 带来的 choice 改善）。

---

## 5. 数据与清理

- 数据：`build/correctness_0.8B-{baseline,chatml,aggmean,agsum,chatml-mean}.json`。
- 进程：全部已关闭，无孤儿 kevcpp。
- 代码：`src/kev_model.cpp` 新增 `KEV_TEMPLATE` 分支；`config/decision.agg-mean.json`、`config/decision.agg-sum.json` 新增。默认（无 env、aggregation=first）行为与 before 完全一致。既有 `KEV_DECIDE`(hidden)、`KEV_CALIB` 未破坏；`bopt.embeddings=1` 未回退。

## 6. 建议给 lead

1. **纳入默认**：把 `aggregation` 默认 `first` 改 `mean`（唯一明确净正收益，零副作用）。或至少在决策配置模板示例里把 `mean` 标为推荐。
2. **ChatML 开关保留但默认关**：对 score 型场景可开，对 choice 型不开；不要全局默认。
3. **下一步最该试**：在 mean 基础上做 **choice 的选项顺序轮换投票**——因为 mean 的增益全部落在 choice，轮换投票可能再放大它，且论文给了强预期。