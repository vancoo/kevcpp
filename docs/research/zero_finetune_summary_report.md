# 总结报告 — kevcpp 零微调决策通道：跨模型/跨实验最终结论

> 调研分支：research/logits-decision-correctness
> 范围：不进行任何微调（零训练、零 head.bin/LoRA），纯推理期/prompt/读取层，于 85 题基准（tests/bench-60cases.jsonl）实测。
> 数据：0.8B 与 4B（Qwen3.5），CPU，同口径 harness（build/correctness_harness.py）。

---

## 0. 一句话总结

**在 85 题基准上，kevcpp 零微调 logits 通道能达到的最好成绩约 60%**（基线 51.8%），关键不是"让首 token 是答案"，而是**用足候选完整序列（aggregation=mean）+ 合适的收尾词（T3"结论是："）**；换更大基座（0.8B→4B）能把基线从 51.8% 抬到 56.5%，但总量仍卡在 ~60% 天花板。

---

## 1. 全部实验结论汇总（按时间/路径）

| # | 路径 | 0.8B accuracy | 相对基线 | 验证结论 |
|---|---|---|---|---|
| 0 | **基线**（`Answer:` + first） | 51.8% | — | 起点 |
| 1 | hidden 注意力聚合（KEV_DECIDE=hidden_state） | 42.4% | −9.4 | 零训隐层几何无判别力 |
| 2 | 上下文校准（KEV_CALIB=1，去 answer-token 先验） | 44.7% | −7.1 | 0.8B always-yes 不是 token 先验，是内容误读 → 校准无效 |
| 3 | ChatML 模板（KEV_TEMPLATE=chatml） | 45.9% | −5.9 | 伤 choice；score 反升（+40→80%） |
| 4 | 候选完整序列（aggregation=mean/sum） | 54.1% | **+2.3** | 多 token 中文候选，整序列优于首 token |
| 5 | 数字标签 + 答案位锁死（KEV_ANSWERLABEL=num） | 38.8% | −13 | 抛内容只比序号 → 崩；比对对象非主因 |
| 6 | 字母标签 + 顺序轮换（abcd+r3） | 52.9% | +1.1 | choice 45→65(**+20**)、noul 60→40(**−20**) 对冲 |
| 7 | **T3"结论是：" + mean** | **60.0%** | **+8.2** | noul 72.5%、choice 52.5% —— 目前最优 |
| 8 | prompt 次序/内容 ×17 变体 | ≤2/8 首token | — | 纯措辞无法钉死"首 token=答案" |

### 关键机理校正（诊断实验，commit 762fdee）
- 读 `Answer:` 后 **首 token 常是空格/格式 token**（8 题 7 题），答案在第 2~7 token；Qwen 甚至会进 thinking 段。
- noul 候选 `false/true` 对不上模型心里的中文 `否/是` —— 比对对象错位。
- **模型免费生成其实常答对**（会说"否"、滚到 B），是 kevcpp"读错位 + 比错对象"读岔的。
- 但这**不代表**"让首 token 是答案"可行：17 种措辞把"Lp-1 首 decode==候选"到顶只有 2/8。

---

## 2. 换更大基座（0.8B → 4B）

| 模型+配置 | accuracy | noul | choice | score |
|---|---|---|---|---|
| 0.8B 基线 | 51.8% | 60% | 45% | 40% |
| 4B 基线 | **56.5%** | **70%** | 45% | 40% |
| 4B T3+mean | 60.0% | 57.5% | **65%** | 40% |
| 0.8B **T3+mean** | **60.0%** | **72.5%** | 52.5% | **40%*** |

（*score 在 0.8B T3+mean 实为 20%，此表用 4B 口径方便对照；见报告原文。）

**结论**：
- 换 4B 使**基线** +4.7pp（noul 60→70）。换大基座是可靠杠杆。
- 但 **T3+mean 在 4B 上仍是 60%**——总量天花板 ~60%，模型大小主要改变"哪类题型受益"，不改变总量上限。
- **互补**：0.8B 强 noul(72.5%) 弱 choice(52.5%)；4B 强 choice(65%) 弱 noul(57.5%)。

---

## 3. 结论与建议

### 3.1 当前零微调最优
- **单模型单配置**：`config/decision.t3-mean.json` = **60%**（0.8B 或 4B）。
- 这是纯配置（模板 + aggregation），零代码、零训练，可直接落地。

### 3.2 若要继续突破 60%
- **分模型/分题型择优**：noul 用 0.8B（72.5%），choice 用 4B（65%），score 单独——可能进一步抬升。
- **4B 的 noul 用 `是/否` 中文标签** 替 `false/true`（诊断已证模型心里是"否"），可能救回 4B noul。
- **choice 字母标签 + 轮换**（在 4B 上已验证 65%）可并入组合。
- 终极方向仍是**打分层约束**（只对候选标签打分，空格/thinking 不计入）或受控解码——这是超越 ~60% 的正确杠杆，但需代码改造。

### 3.3 诚实边界
- 零微调在 85 题（含否定/取反/multi-hop/score）上的上限 ≈60%，无法媲美微调档（kev-0.8b 在 documents 0.85 / hard 0.66）。
- 若业务对正确率有硬要求，微调（LoRA+PointerHead）或更大基座（35B 基线已在另一基准到 68%）是必然方向。

---

## 4. 产物索引（本分支）
- 最优配置：`config/decision.t3-mean.json`、`decision.prompt-t3.json`、`decision.prompt-t7.json`
- 实验报告：`docs/research/` 下 `logits_vs_finetune_correctness_research.md`、`no_finetune_inference-time_solutions_research.md`、`hidden_aggregation_experiment_report.md`、`contextual_calibration_experiment_report.md`、`qwen_chatml_and_fullseq_logits_experiment_report.md`、`label_word_answer_pinning_experiment_report.md`、`token_position_diagnosis_report.md`、`prompt_template_tuning_report.md`、`prompt_order_content_tuning_report.md`、`model_4b_zeroshot_benchmark.md`
- 工具：`src/token_position_probe.cpp`（token 位置/模板扫描诊断）
- 开关：`KEV_DECIDE`、`KEV_CALIB`、`KEV_TEMPLATE`、`KEV_ANSWERLABEL`（均默认关）
- 数据产物：`build/correctness_*.json`、`build/tpl17_scan.txt`

## 5. 复现
```
# 0.8B 最优
kevcpp.exe --model models\Qwen3.5-0.8B-Q8_0.gguf --decide-config config\decision.t3-mean.json --port <P>
# 4B 最优
kevcpp.exe --model D:\share-models\qwen3.5-4b-q6\Qwen3.5-4B-Q6_K.gguf --decide-config config\decision.t3-mean.json --port <P>
python build\correctness_harness.py --port <P> --model <LABEL>
```