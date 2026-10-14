# 实验报告 — kevcpp hidden 聚合判别通道（零训练，KEV_DECIDE=hidden|hidden_state）实测结果

> 调研分支：`research/logits-decision-correctness`
> 动机：验证「不用微调，改用注意力/隐层聚合替代'取首 token logits'能否提升决策正确率」。
> 实现：`src/kev_model.cpp` 新增 `probs_hidden()` / `decode_one_to_hidden()`，`bopt.embeddings=1`，由环境变量 `KEV_DECIDE=hidden|hidden_state` 切换；默认仍是 logits。
> 数据：`tests/bench-60cases.jsonl`（85 题），同机同为 `models/Qwen3.5-0.8B-Q8_0.gguf`、`--backend cpu --ctx 2048`，复用 `build/correctness_harness.py`。

---

## 1. 结论（TL;DR）

**纯推理期、零训练的「注意力/隐层聚合」通道正确率不升反降**：

| 通道 | accuracy | noul | choice | score |
|---|---|---|---|---|
| logits（现状基线，本次同 build 复测） | **51.8%** (44/85) | 60% | 47.5% | 40% |
| **hidden_state**（attention 加权聚合 state + 中心化余弦） | **42.4%** (36/85) | 45% | 42.5% | 20% |

即：**你设想的"用注意力找相关内容再 sum/mean"在未训练 0.8B 上没有提升，反而明显变差**。这是一个诚实、可复现的负结果，且原因很具体。

---

## 2. 为什么 hidden 聚合反而更差（机理）

从实测分布的诊断看，hidden 通道的特徵是 **"伪判别 / 系统偏置，而非真信号"**：

1. **noul 从"always-yes"翻成"always-false"**：
   - logits 通道：应 False 全答 True（→ 60%）。
   - hidden_state 通道：**应 True 的也几乎全答 False**（prob 全部压在 0.40–0.49，id 6/8/9/10/17/19/26/28/29/30/37-40/46-50/64/77 等 True 被答成 False）。
   - 原因：中心化后的余弦得分对"false token 的续写隐藏"系统性偏大——这是一个**恒定偏移/偏置**, 不是对文本语义的判别。
2. **choice / score 概率几乎均匀**（confidence≈0.002–0.03）：
   - 每个候选的 `cos(center_query, center_cand)` 都在一个窄区间（如 4 选 1 时 p≈0.24–0.29），softmax 后接近均匀，argmax 就退化成噪声。
3. **根因**：未训练模型的高维残差流 hidden 里，"query（决定向量/聚合向量）"与"各候选续写结尾 hidden"的差异方向**并不与"是否正确答案"对齐**。中心化只去掉了各候选共享的"通用 LM 激活方向"，剩下的差异大多是文本无关的噪声。**"哪个方向 = 正确答案"正是 PointerHead 的 `Wq`/`Wk` 通过训练学习的那部分**；没有它，任意零参数投影/聚合都点不到这个方向。

> 这与第一份调研的结论自洽：**零微调下能修掉"单位置浪费 + 读错对象（vocab 头 vs 表示）"，但补不回"隐层几何→正确选项"的对齐（那需要训练）**。实测证实了这一点。

---

## 3. 什么变了 / 没变

- ✅ `bopt.embeddings=1` 生效，`read_hidden` 可在多个 token 位置读表示向量；`probs_logits` 在 embeddings 开启下**未回归**（本 build 复测 51.8% ≈ 原 52.9%，CPU vs Vulkan 噪声内）。
- ✅ `probs_hidden` 实现了「决定向量(=attention 聚合全文) + 可选对 {state} 做软注意力加权聚合 + 候选续写 hidden + 中心化余弦」整条链路，零训练。
- ❌ 正确率没有因此提升——这是本实验的核心负面结论。

---

## 4. 可复现与后续

- 复现：`$env:KEV_DECIDE="hidden_state"; kevcpp.exe --model models/Qwen3.5-0.8B-Q8_0.gguf --port 18101`，再跑 `python build/correctness_harness.py --port 18101 --model x`。
- 数据产物：`build/correctness_0.8B-hidden_state.json`（42.4%）与 `build/correctness_0.8B-logits-embon.json`（51.8%）。
- 意义：**它把"纯推理期能补多少"这个问题的答案量化了**——对 0.8B 未训练基座，B. logits 仍是零训练里相对可靠的；hidden 聚合单独用不如 logits，更说明"要正确率必须引入被训练的投影/对齐（PointerHead / LoRA）或更强的推理期机制（校准/轮换投票/RAG）"，而不能指望靠"把 hidden 换个聚合方式"白拿。

---

## 5. 证据索引

- 实现：`src/kev_model.cpp`（`probs_hidden`、`decode_one_to_hidden`、`hidden_state_span`）、`src/kev_model.h` 声明。
- 后端能力：`src/kev_backend.h`（`read_hidden`, `KEV_BACKEND_CAP_HIDDEN_READ`）、`src/kev_backend_cpu.cpp`（`op_read_hidden` → `llama_get_embeddings_ith`）。
- llama.cpp 语义：`third_party/llama.cpp/src/llama-context.cpp`（`output_all = cparams.embeddings`；`get_embeddings_ith` 与 `get_logits_ith` 共用 `output_resolve_row`，故 embeddings 开不影响 read_logits 语义）。
- 机制结论：`docs/research/logits_vs_finetune_correctness_research.md`、`docs/research/no_finetune_inference-time_solutions_research.md`。