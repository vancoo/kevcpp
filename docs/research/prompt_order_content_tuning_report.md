# 实验报告 — prompt 次序/内容优化：能否让"第一个 decode 就是选项"？

> 调研分支：research/logits-decision-correctness
> 任务：通过 prompt 的【次序/结构】与【内容/措辞】优化，尽量提升"Lp-1(答案位) 处模型第一个 decode(argmax token) 就是候选选项标签"的概率。
> 工具：`src/token_position_probe.cpp`（17 个变体 V0~V16，含 `{optlist}` 逐行选项），`--steps 0` 只测 Lp-1 首 decode，8 个代表题（否定noul×3/choice×2/score×1/日期noul/反义choice）。

## 核心结论（决定性）
**17 种次序/内容变体中，没有任何一个能让"Lp-1 处第一个 decode==候选标签"在 8 个代表题里可靠出现——最高仅 2/8，多数 0~1/8。**

| 变体 | first-decode==CAND (n/8) | 变体 | n/8 |
|---|---|---|---|
| V0 基线 `...Answer:` | 0 | V9 只输出标签 | 1 |
| V1 结论是 | 1 | V10 请选出一个答案 | 0 |
| V2 正确选项是 | 0 | V11 你的选择是 | 0 |
| V3 state先行+结论是 | 1 | V12 答 | 1 |
| V4 答案是 | 1 | V13 请回答 | 0 |
| V5 optlist+结论是 | 1 | V14 state先行+optlist+结论是 | 1 |
| V6 optlist+答案是 | **2** | V15 few-shot"答案:A" | 1 |
| V7 系统头判别器 | 1 | V16 答案是: | 0 |
| V8 禁空格禁解释 | 1 | | |

## 机理解释
1. **0.8B 思考型模型在纯 `Answer:`/各类收尾后,第一个 token 仍是空格/标点/thinking 居多**——纯 prompt 措辞/次序无法真正"钉死首位"。这与先前诊断一致(首 token 常是空格,答案在第 2~7 token)。
2. **真正提升正确率的不是"让首 token 成为答案",而是"读取/聚合侧"**：之前的 T3"结论是："+**aggregation=mean(候选完整序列 log-prob)** 把 0.8B 从 51.8% 抬到 **60.0%**(noul 72.5%、choice 52.5%)——这个收益来自 mean 聚合,而非"首 token 压过空格"。因为 mean 对候选整词打分,不依赖"首个 decode 就是答案"。

## 结论与启示
- **"让第一个 decode 就是选项"单靠 prompt 次序/内容调优在此 0.8B 上不可靠**(≤2/8),不建议继续在措辞上投入。
- 正确的零训练路径(已落地)是：**用足选项完整序列(mean)+ 合适的收尾词(T3)** → 已达到 60%。若还想进一步,应转向**打分层掩码(只对候选标签打分,空格/thinking 不计入)**或**受控解码**,而不是纯 prompt。

## 复现
`cmake --build build --config Release --target token_position_probe` 后
`build/bin/Release/token_position_probe.exe --model models\Qwen3.5-0.8B-Q8_0.gguf --steps 0`
读输出里 `[TEMPLATE-AGG] first-decode==CAND: n/8`。

## 递移
- 最优模板(按 85 题 accuracy)仍为 T3`结论是：`+mean: `config/decision.t3-mean.json`(60%)。
- 相关工作区改动未动其它实验产物。