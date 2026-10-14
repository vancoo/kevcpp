# 实验报告 — kevcpp 上下文校准（contextual calibration）决策通道 实测结果

> 调研分支：`research/logits-decision-correctness`
> 适用范围：0.8B（`models/Qwen3.5-0.8B-Q8_0.gguf`），`tests/bench-60cases.jsonl`（85 题），CPU，`--ctx 2048`。
> 理论依据：Zhao et al., *Calibrate Before Use*, ICML 2021（arXiv:2102.09690）。
> 开关：`KEV_CALIB=1`，默认关（行为完全不变）；`KEV_CALIB_SCALE`（默认 1.0）放大偏置扣减。零训练、纯后处理，不含任何 head.bin/LoRA。

---

## 0. 结论速览

**上下文校准在本数据集上未能提升决策正确率，反而轻微变差**：baseline 51.8% → 校准后 44.7%（scale=1 与 scale=4 一致）。且**未治好 noul 的 always-yes**。这是一个清晰、可复现的**负结果**，并给出可操作的机理结论（见 §3）。

| 通道 | accuracy | noul | choice | score |
|---|---|---|---|---|
| logits 基线（未校准） | **51.8%** (44/85) | 60% (24/40) | 45% (18/40) | 40% (2/5) |
| 校准 scale=1 | 44.7% (38/85) | 50% (20/40) | 42.5% (17/40) | 20% (1/5) |
| 校准 scale=4 | 44.7% (38/85) | 50% (20/40) | 42.5% (17/40) | 20% (1/5) |

---

## 1. 实现（`src/kev_model.cpp` / `kev_model.h`）

1. **开关**：`init()` 读 `KEV_CALIB=1` → `calibrate_=true`；校准开启时 `probs_logits` 强制走**串行路径**（`batched=false`），以在同一 `compute_mutex_` 内安全解码并惰性估算偏置。
2. **偏置估算** `estimate_answer_bias(keys)`：用内容中性输入（`instructions=""`、`state="N/A"`）拼同一模板，走与 `probs_logits` "first" 相同的语义（单次 decode 末位置读各候选首 token logits），得偏置 logit 向量 `z_bias[c]`。
3. **校准变换** `apply_calibration(q,z)`：softmax 前 `z'[c] = (z[c] - z_bias[c]) * KEV_CALIB_SCALE`。缓存按「候选 key 列表 + 模板 + 聚合」键控，惰性估算一次。
4. 集成点：`probs_logits` 中 `softmax_scaled(z, temperature, probs[k])` 之前。
5. 仅支持默认 `aggregation=first`（`sum/mean` 需分别估算偏置，未实现，报告里备注）。

> 说明：`probs_hidden`（上一实验）与 `bopt.embeddings=1` 未被破坏；默认（无 `KEV_CALIB`）行为完全不变。

---

## 2. 实测明细（noul 混淆，最关键）

| noul 混淆 | baseline | calibrate |
|---|---|---|
| 应 True→答 True (TP) | 24 | 18 |
| 应 False→答 False (TN) | **0** | 2 |
| 应 False→答 True (FP) | 16 | 14 |
| 应 True→答 False (FN) | 0 | 6 |
| 40 题结果不变 | – | 32 |

**基线病根确认**：baseline 的 noul 全部 40 题都答 True（TP=24 / FP=16，即**从不答 False**）——"always-yes"。

**偏置实测**：中性输入下 `bias(false)=-0.17, bias(true)=+0.86`（净 ~+1.03 偏向 false）。即中性上下文本身就偏爱 true；扣减后每个题都应把分数往 false 推 ~1.03。但**只翻转了 16 个 False 题里的 2 个**，还把 6 个 True 题推成了 False（net -4）。

---

## 3. 为什么失败（机理结论）

**上下文校准修的是「答案 token / 格式/频率的系统性先验」**；它**修不了「模型误读了内容语义」**。具体：

- 0.8B 的 noul always-yes 大量集中在**否定/反义/取反**类（"不参与""未开工""不存在""未否认"），这些是模型**没读懂内容** → 无脑续 True。
- 内容本身诱导的 True 偏置远大于中性输入的 ~1.03 偏置，所以扣那 1.03 推不动边界（只翻转 2/16）。
- 即便放大 scale（4）也只是等比例放大 z′ 的相对分离，**不能产生它本来没有的 sign 翻转**，故结果与 scale=1 完全相同。
- choice/score 也没有位置/格式意义上的可校准系统性偏置（choice 18→17、score 2→1，基本打平/微降）。

**对 kevcpp 的含义**：直接用"首 token logits + 中性输入偏置扣减"的上下文校准**不值得沿用**去治 always-yes——因为根因是 0.8B 的阅读理解，不是输出 token 先验。校准在"模型其实知道答案、只是偏向写某个 token"时才有用，在"模型不懂内容"时无效。

---

## 4. 复现

```bash
# baseline
.\build\bin\Release\kevcpp.exe --model models\Qwen3.5-0.8B-Q8_0.gguf --backend cpu --port 18110 --ctx 2048 --quiet
python build\correctness_harness.py --port 18110 --model calib-baseline

# calibrated
$env:KEV_CALIB="1"; $env:KEV_CALIB_SCALE="4"
.\build\bin\Release\kevcpp.exe --model models\Qwen3.5-0.8B-Q8_0.gguf --backend cpu --port 18113 --ctx 2048 --quiet
python build\correctness_harness.py --port 18113 --model calib-on-scale4
```

产物：`build/correctness_calib-baseline.json`（51.8%）、`correctness_calib-on.json`（44.7%）、`correctness_calib-on-scale4.json`（44.7%）。

---

## 5. 建议（下一步）

1. **不用沿用「首 token + 中性输入」的上下文校准**来治 always-yes；它把根因当作了输出先验，而根因是 0.8B 内容理解。
2. 更可能有效的（基于上一份 arXiv 调研）：
   - **候选完整序列 log-prob（`aggregation=sum/mean`）**——修"只读一个 token"的表示失真；
   - **选项顺序轮换 + 投票 / 排序投票**——修 position/label bias；
   - **RAG**（把判定规则检索进 prompt）——直接补 0.8B 的理解缺口；
   - **换更大的基座（9B/35B）**——从 benchmark 看 35B 本就 68%；
   - 若必须修 0.8B 的 always-yes 且不微调：可试把 noul 的判定阈值/局部边界后处理（但这是 dev-set 调优，违背校准初衷，需谨慎）。

---

## 6. 文件清单（本次改动）

- `src/kev_model.h`：新增 `calibrate_`、`calib_cache_`、`estimate_answer_bias`、`apply_calibration` 声明；加 `<map>`。
- `src/kev_model.cpp`：`init()` 读 `KEV_CALIB`；`probs_logits` 校准开启时强制串行 + softmax 前调用 `apply_calibration`；实现 `estimate_answer_bias` / `apply_calibration`。
- 本文档 `docs/research/contextual_calibration_experiment_report.md`。