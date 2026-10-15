# 4B 基座零微调 85 题 A/B 基准

> 任务：对 4B 模型（Qwen3.5-4B-Q6_K）跑零微调 85 题 A/B，量化换更大基座相较 0.8B 的提升。
> 数据：tests/bench-60cases.jsonl（85 题：score5 / choice40 / noul40），0.8B 与 4B 同口径，CPU/ctx2048。

## 结果表

| 模型+配置 | accuracy | noul | choice | score |
|---|---|---|---|---|
| 0.8B 基线（Answer:+first） | 51.8% | 60% | 45% | 40% |
| 4B 基线（Answer:+first） | **56.5%** | **70%** | 45% | 40% |
| 4B T3+first（结论是+first） | 57.65% | 57.5% | 60% | 40% |
| **4B T3+mean（结论是+mean）** | **60.0%** | 57.5% | **65%** | 40% |
| （对比）0.8B T3+mean | 60.0% | **72.5%** | 52.5% | 20% |

## 分解分析
- **换基座本身的提升（基线口径）**：0.8B 51.8% → 4B **56.5%（+4.7pp）**，主要来自 **noul 60%→70%**。4B 基线的 choice(45%) 与 0.8B 持平。
- **4B 上 T3 模板（结论是：）的独立作用**：choice 45%→60%(+15)、noul 70%→57.5%(−12.5)。即 `结论是：` 大大救 4B 的 choice,但伤 4B 的 noul。
- **4B 上 mean 聚合的作用**：choice 再 +5（60→65%），noul 不变。score 恒定 40%。
- **跨模型对比**：T3+mean 在 0.8B 与 4B 都到 60%，但**分布相反**：0.8B 强在 noul(72.5%)、弱在 choice(52.5%)；4B 强在 choice(65%)、弱在 noul(57.5%)。

## 结论
1. **换 4B 是提升零微调正确率的可靠杠杆**：基线口径 +4.7pp（51.8→56.5%），T3+mean 口径与 0.8B 持平在 60%（但分布更均衡、choice 更强）。
2. **4B 没让总正确率突破 60%**（与 0.8B T3+mean 相同）——零微调 logits 通道的 aggregate 天花板在此方法下约 60%，模型大小主要改变"哪类题型受益"，而非总量大幅抬升。4B 的强项是 choice，弱项（noul）反而比 0.8B 差。
3. 建议：若要继续提正确率，4B/更大模型的 **choice** 已够强（65%），短板在 **noul（57.5%）与 score（40%）**；可针对性用 `是/否` 标签或分题型组合（0.8B 的 noul 表现更好，可择优模型联合/分题型）。

## 复现
- 4B: `kevcpp.exe --model "D:\share-models\qwen3.5-4b-q6\Qwen3.5-4B-Q6_K.gguf" --backend cpu --port <P> --ctx 2048 --decide-config config\decision.<cfg>.json`
- harness: `python build\correctness_harness.py --port <P> --model 4B-<label>`
- 结果文件：build/correctness_4B-{baseline,t3first,t3mean}.json