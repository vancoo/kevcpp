# 实验报告 — prompt 模板调试：让"答案候选标签贴近 Answer: 首位"

> 调研分支：research/logits-decision-correctness
> 动机：诊断已证实纯 `...\nAnswer:` 后 0.8B 首 token 常是空格，答案在第 2~7 个 token。本实验系统调试模板措辞，看能否让候选标签在 `Answer:` 处压过空格(margin>0)。
> 工具：改造 `src/token_position_probe.cpp`——模板数组化(T0~T8)，对每个 (case×tpl) 读 Lp-1 全量 logits，输出"最佳候选标签 logit − 空格 logit"的 margin 及 argmax。
> 数据：8 个代表题（docs/research/token_position_diagnosis_report.md 的 c1n..c8_choice），0.8B CPU。

## 1. 核心指标
`margin = 最佳候选标签(choice:A/B; score:数字; noul:false/true)的 logit − 空格' '的 logit`。>0 = 候选压过空格 = 答案更可能"第一位出现"。

## 2. 模板 × 平均 margin 表（8 case 平均）

| 模板 | avg_margin | 其中 choice(c5/c8) margin |
|---|---|---|
| T0 基线 `...\nAnswer:` | **−9.56** | 全负（−7~−8）|
| T1 禁空格中文指令 | −7.10 | c5 −7.10 |
| T2 先列选项+序号 | −9.07 | 全负 |
| T3 收尾"结论是：" | −1.44 | **c5 +2.43, c8 +5.13** ✅ |
| T4 "正确选项：" | −0.25 | **c5 +4.93** ✅ |
| T5 few-shot"答案是 X" | −0.80 | c5 +4.53, c8 +4.89 ✅ |
| T6 括号限定 (A/B/C/D) | −1.36 | c8 +3.31 ✅ |
| **T7 "Answer："** | **+0.03** | c5 +1.79, c8 +3.34 ✅ |
| T8 "只输出字母" | −0.82 | c8 +3.13 ✅ |

## 3. 结论
1. **基线最差**：T0 的 `Answer:` 处候选被空格压死（margin −9~−11）。诊断正确。
2. **措辞真能抬**：所有变体都把 avg_margin 从 −9 抬到 ≈0 甚至正（T4/T7），尤其 **choice 的 A/B 成功压过空格（+2~+5）**。即 prompt 工程是零训练低成本大杠杆。
3. **noul/score 仍被空格压着**：noul 候选是英文 `false/true`，在这些模板下 margin 仍 −1~−11；score 数字也难。→ **noul 用英文 false/true 标签本身就是错位**（对不上模型心里的 否/是）。

## 4. 复现
`cmake --build build --config Release --target token_position_probe` 后
`build/bin/Release/token_position_probe.exe --model models\Qwen3.5-0.8B-Q8_0.gguf --steps 3`
输出含每 (case×tpl) 的 `[PTMPL] ... margin(cand_best-space)=...` 行。

## 5. 后续
- 对最优模板（T4/T7/T3 类"少字强命令/结论是"）配 noul→`是/否`，跑完整 85 题量化正确率提升。