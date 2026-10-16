# 实验报告 — 关闭思考（enable_thinking=false / empty-thinking block）

> 调研分支：research/logits-decision-correctness
> 任务：让 Qwen3.5 思考型模型"不思考"，使答案位首 token 尽量就是选项，并 85 题 A/B。

## 1. 机制依据（已从 GGUF 提取 chat_template 核实）
`build/chat_template.txt` 末尾（enable_thinking 开关）：
```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is true %}
        {{- ' thinking\n' }}
    {%- else %}
        {{- ' thinking\n\n response\n\n' }}
    {%- endif %}
{%- endif %}
```
→ **enable_thinking=false 时，assistant 后给一个"空思考块" ` thinking\n\n response\n\n`，模型应跳过思考直接作答。**
该模型 GGUF 特殊 token 含 `<|im_start|>/<|im_end|>`，但**无 `<|think|>/<|answer|>`**；` thinking`/` response` 是普通文本词而非特殊 token。

## 2. 实现（零训练，`KEV_TEMPLATE=nothink`，默认 off）
- `src/kev_model.cpp`：新增 `KEV_TEMPLATE=nothink` 分支，模板为
  ```
  <|im_start|>system\n你是严格按给定选项作答的判别器，只输出最终选项，不要任何思考或解释。<|im_end|>\n
  <|im_start|>user\n{instructions}\n[ {options} ]\n{state}<|im_end|>\n
  <|im_start|>assistant\n thinking\n\n response\n\n
  ```
- **关键坑**：ChatML 模板须 `parse_special=true` 才识别 `<|im_start|>`；改了 `probs_logits` 的 tokenize（`tpl_special` 为 true 时）。
- `src/token_position_probe.cpp`：加 V17 nothink 模板 + 按模板含 `<|im_start|>` 决定 parse_special。

## 3. 层1：首 token 诊断（贪心前10 token，代表题）
| 题 | V0 基线 | V17 nothink |
|---|---|---|
| c1n_noul(答否) | `' ' 否 A 公司 宣布 将 不参与...` | `no <空><空><空> assistant thinking response` |
| c5_choice(答A) | `' A' thinking response 根据 您 提供...` | **`A*` <空><空><空> assistant thinking response** |
| c8_choice(答B) | `' A' 球 在 1 号 位 ，B* 球 在` | `A <空><空><空> thinking response` |
| c2n_noul(答否) | `' ' 否 ，该 项目...` | `no <空><空><空> assistant thinking response` |

**首 decode==候选统计**：V0=0/8 → **V17=2/8**。
- 改善：nothink 确实"停下思考"（不再进 `<thinking>` 复述），且 **c5_choice 首 token 就是正确 `A*`**（rigorous: 之前基线是 `' A' thinking...`）。
- 但多数题首 token 仍是空/`no`/结构残留，模型在 `response` 后又 drifts 出 `<|im_start|>assistant/thinking/response` 片段——说明"读取位 = Lp-1(after `response\n\n`)"仍不完全落在答案 token 上。

## 4. 层2：85 题 harness（0.8B / CPU）
| 配置 | accuracy | score | noul | choice |
|---|---|---|---|---|
| 基线 first | 51.8% | 40% | 60% | 45% |
| T3+mean（既往最佳） | 60.0% | 20% | 72.5% | 52.5% |
| **nothink** | **51.8%** | 40% | 65% | 40% |

- 总体与基线持平（51.8%），未超过 T3+mean 的 60%。
- 相对基线：score 持平、noul 60→65（小升）、choice 45→40（小降）。

## 5. 结论
1. **关闭思考本身"见效"了**——模型不再进 `<thinking>` 复述，且部分题（c5）首 token 就是答案；"首 decode==候选" 0/8→2/8。
2. **但"读 Lp-1=response 后那个单 token"仍不理想**：模型在 `response\n\n` 后往往先吐空/`no`/结构残留，单点读取没把答案 nailed 住。
3. **单点 logits 通道的瓶颈不在"思考"，而在"读取位置/方式"**——即便不思考，首 token 也未必是答案(n=2/8)。这与 `aggregation=mean`(T3+mean=60%)"用候选整词 log 概率"的结论一致：单点首 token 上限有限。

## 6. 建议
- 若坚持"不思考"，正确落地应是 **nothink + aggregation=mean**（用候选整词 log 概率，而非单点首 token）——未测，值得一试；或**打分层只对候选标签打分**（空格/`no`/`response` 不计入）。
- 折中：`KEV_TEMPLATE=nothink` 留作开关，默认 off；推荐仍用 T3+mean(60%) 作为零微调基线。

## 7. 复现
```
KEV_TEMPLATE=nothink kevcpp.exe --model models\Qwen3.5-0.8B-Q8_0.gguf --port <P> --ctx 2048 --quiet
python build\correctness_harness.py --port <P> --model 0.8B-nothink
```
探针：`token_position_probe.exe --model models\Qwen3.5-0.8B-Q8_0.gguf --steps 10`（V17 nothink）。