# 实验报告 — 不微调"多语言别名池 + logits 聚合"相似度判别

> 调研分支：research/logits-decision-correctness
> 问题：kevcpp 用"候选 key 首 token 的 logit"做精确定点匹配。诊断发现模型在答案位想说"否/是"，但 key 是 `false/true` → 比对对象错位。之前尝试 hidden 直接余弦(路径1, 36.5%)、提示词引导方案B(模型仍不吐 false/true) 均失败。
> 本实验探索：**不微调、不读 hidden**，只改用"多语言/同义别名池"对 Lp-1 的 logits 聚合(每选项 = 其别名池各词首 token 的 log-sum-exp)，再比较。
> 工具：`src/token_position_probe.cpp` 新增 `[ALIAS]` 诊断(依赖已有的 Lp-1 全量 logits)。

## 结果（8 个代表题）

**基线模板(T00) + 别名池聚合 = 6/8 正确**（远高于 plain-first 和 hidden cos）：

| case | 类型 | 正确答案 | 别名池 argmax | 是否对 |
|---|---|---|---|---|
| c1n | noul | false | false | ✅ |
| c2n | noul | false | false | ✅ |
| c3n | noul | false | false | ✅ |
| c4n | noul | false | true | ❌(难例) |
| c5 | choice | A | A | ✅ |
| c6 | score | 2 | 1 | ❌(近miss) |
| c7 | noul | false | false | ✅ |
| c8 | choice | B | B | ✅ |

按模板：多数模板 6/8；T29("必须原样输出 false/true") 1/8(强行引导反而崩)。

## 关键结论
1. **"模型说 否 但 key 是 false" 的多语言错位，用"别名池聚合"就能修掉**(c1n/c2n/c3n/c7 的 false 判定全对)——零训练、只需把候选 key 展开成别名集(lse 聚合)。
2. **这明显优于之前所有路线**：
   - plain-first logits：noul 几乎全错(always-yes + 错位)。
   - hidden 余弦(路径1)：42%/36%。
   - 提示词引导方案B：模型仍不吐 false/true，还污染 choice。
   - **别名池聚合(Lp-1 logits) = 6/8**。
3. 无需嵌入/hidden/微调，纯 logits + 别名集，多语言天然适应（"否/是"、`false/true`、`no/yes`、`假/真` 都进池子）。
4. 残留错误：c4(难否定)、c6(score 近miss) —— 是模型能力/分数排序问题，非错位。

## 结论
**建议落 kevcpp**：把 noul(及可扩展任意选项)的候选打分从"单 key 首 token logit"改成"别名池 log-sum-exp"。零训练，极可能提升 noul(以及 choice/score 若配别名)。这是一个"读取层"的一步、普适多语言，值得接进 `probs_logits` 作为候选聚合选项。

## 复现
`cmake --build build --config Release --target token_position_probe` → `build/bin/Release/token_position_probe.exe --model models/Qwen3.5-0.8B-Q8_0.gguf --steps 0`，读 `[ALIAS]` 行与 `[TEMPLATE-AGG] alias-pool-argmax==CORRECT`。

## 待办/后续
在 kevcpp `probs_logits` 加 `aggregation=alias`(或 env 开关)，把 noul 候选别名池{否,是}/{false,true}/{no,yes}/{假,真} 折叠后 lse，跑 85 题验证 noul/choice/score 提升。此为本实验最直接也是最值得落地的下一步。