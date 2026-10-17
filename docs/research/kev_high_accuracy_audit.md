# kev 高准确率审计报告 —— 为什么 kev(0.8B 错5/4B 错2/9B 全对)而 kevcpp 零微调 4B 只有 71.8%

> 审计对象:E:\van\projects\kev(参考工程) 与 E:\van\projects\kevcpp(本项目)
> 方式:源码/README/model-card/PLAN/train.py 逐条取证;不跑重实验。
> 分支:research/logits-decision-correctness

---

## 0. 一句话结论

**kev 的 0.8B/4B/9B 高准确率不是"读取技巧",而是"用 ~1.25 万条标注样例 + LoRA rank16 对 backbone 做了 2 epoch 全模型微调",并且 PointerHead(Wq/Wk)与 backbone-LoRA 联合训练;kevcpp 零微调缺少的正是这一次"针对决策任务的学习"。** 且 kev 官方自述:"adapted backbone 用 LM head 读出的成绩与 pointer 相同"(`PLAN.md` L182),说明**高准确率主要来自 backbone 被 LoRA 改写了,pointer 只起读出/校准作用,并非瓶颈所在**。

---

## 1. kev 的训练配方(证据: README.md L283-299, kev/train.py, 模型卡 front-matter)

| 项 | 值 | 证据 |
|---|---|---|
| 基础训练集 | `decision-v7`:10,000 条(10 个公开数据集)+ 896 条(生成 policy)+ 1,680 条(60 个生成规则结构)≈ **1.25 万** | README L283 |
| epoch | **2** | README L283 / train 命令 `--epochs 2` |
| 适配器 | **LoRA rank=16**(train.py 默认 `--lora 16`) | README L283 / train.py `--lora default 16` |
| 学习率 | 0.8B=`1e-4`;4B/9B=`5e-5` | README L283 |
| loss | **cross-entropy**(CE;指 decision 多选/是否/评分) | README L283 |
| 训练对象 | **LoRA adapter(attention/MLP/DeltaNet 全部投影)+ PointerHead(Wq/Wk)联合训练** | `train.py` L575-576:`head_params` 单独 param-group,与 `model.trainable_parameters()` 一起 SGD;模型卡 `library_name: peft`,`base_model_relation: adapter` |
| 后续 delta | day-count 生成例、真实文档、skill 数据(0.8B/4B)各一轮 | README L285 |
| 加载 | 释放的权重就是 **peft LoRA adapter**;指针头单独 `head.pt` | checkpoint.py、模型卡 front-matter |
| H100 成本 | 0.8B 首 stage ~20 min;4B ~1 h | README L293-299 |

**关键纠正**:kev **不只是**训 Wq/Wk,它**对整个 backbone 做了 LoRA rank16 微调**。这是和"改读取"本质不同的事——它改的是模型权重本身。

---

## 2. "kev 0.8B 错5 / 4B 错2 / 9B 全对" 的来源核实

- 在 kev 工程内**未找到**对 kevcpp `tests/bench-60cases.jsonl`(85 题,19 个类别)的这份给分记录(没有 5/85、2/85、全对 的 report/json)。
- kev 官方评测都是它自己的套件(`decision-v7`/`hard-v1`/`devtools-v1`/`documents-v1`/`transfer-v4`/MMLU 等),**口径与 kevcpp 的 85 题不同**。
- 因此用户口径的这两个数字是**用户方实测/外部来源**,kev 仓库无法独立佐证。审计以"kev 训练配方 + 能力边界"为据,不背书具体 5/2/0。

**kevcpp 85 题的 19 类(证据: bench-60cases.jsonl 统计)**:
`情绪强度5 日期先后4 日期闰年1 数字比较3 三选一2 金额阈值5 事件分类5 距离阈值5 季度比较5 同比方向5 持股比例5 币种识别5 多跳5 否定反义5 三段论5 时间序列5 物体追踪5 量词5 反讽5`;qtype:noul40/choice40/score5。

---

## 3. kev 的读取结构:哪些"纯阅读"、哪些"必须训练"

读取(证据:`kev/model.py` encode L82-119, `_readout` L303-304, PointerHead L182-201):

```python
# encode: 每个 question = <state> + [<q>instr] + <o>opt</o>...(每选项) + <decide>
decide_idx=[base+len(br)-1]; opt_idx=[base+end for end in ends]   # </o> 位置
# _readout:
z = head(h[decide_idx], h[opt_idx])   # = (Wk·h_opt) @ (Wq·h_decide) * scale
```

| 组件 | 是否可"零训练抄" | 说明 |
|---|---|---|
| `<decide>` 锚 token + prompt 结构 | **可抄** | 用特殊 token 给出"在这下判断"锚点(kevcpp 起步已用 marker,方向对) |
| `</o>` 选项闭合位 hidden | **可抄** | 每个选项取 `</o>` 位置 hidden 作为"选项表示" |
| h_decide ↔ h_opt 做 cross-match 框架 | **可抄(形)** | kevcpp 的 `decide` 通道已复刻此形,零训余弦 47% |
| **Wq/Wk(Linear d→dp)投影** | **必须训练** | 决定"哪个向量方向=正确"的判别子空间;不训则余弦恒定(共线),实测 47% |
| **backbone-LoRA 改写** | **必须训练** | kev 官方:adapted backbone 经 LM head 与 pointer 同分(L182)——**判别力在训练过的 backbone,不在读取** |
| 特殊 token 的 embedding 训练(`special_embeddings`) | 可选训练 | 模型卡提到 27B 用过;0.8B/4B/9B 基础 recipe 未必须 |

**所需训练,已彻底排除是 kev 能"只抄结构"**:`PLAN.md` L182 自己写"adapt backbone 经 LM head 读 = pointer 同分",实证了判别力在 backbone-LoRA。

---

## 4. kev vs kevcpp:逐项对照(全部可审计)

| 维度 | kev | kevcpp 零微调 | 性质 |
|---|---|---|---|
| backbone | Qwen3.5-Base,经 **LoRA rank16 微调** | Qwen3.5 原样,不动权重 | kev 训练了,kevcpp 没有 |
| 训练数据 | ~1.25 万条(decision-v7)+ delta | 0 | 数量级差距 |
| 决策头 | PointerHead Wq/Wk(训练) | 别名池/LM-head 通用 logits | kev 专训,kevcpp 通用 |
| 读取位置 | `<decide>` + `</o>`(选项描述位) | `Lp-1`/别名池(候选 key) | kev 语义,kevcpp 词面 |
| 答案判定 | h↔h cross-match | token logits / 别名聚合 | — |

**kev 对 kevcpp 多出来的核心是:模型权重被训练过 + 一个专训头**。其余全是"可以抄但不带来准确率"的部分(前面已经用零训 decide 通道 47% 证过)。

---

## 5. 要复现 kev 90%+,kevcpp 需要的路径

**(a) 唯一确定的、有实证支撑的路:训练**——与用户的强约束(不训练)本质冲突,必须如实讲:
  - 用决策格式的标注数据(LoRA rank16, CE, ~1.25 万条量级, 2 epoch)微调 backbone;或全量 SFT(PLAN L166 说全量 SFT 甚至胜过 LoRA recipe)。
  - 需能接受"训练一次映射/模型权重"。这等价于 kev 的第一 stage。
  - 若要"不微调 backbone,只训线性头"——需**远超 85 题的训练样本**(前面实验:40 题 LOO 训 10240 维头过拟合到 42%)。最少几百~几千条**同类决策题**,且该头仍受"末层 hidden 共线"限制,成功率存疑。

**(b) 零训练天花板(已实证)**:kevcpp 4B combo 别名池 = **70.6%**,模型自答(nothink2+first)= **71.8%**;所有零训练读取/公式/别名校准已扫,收敛在此。**这就是零微调可复现的极限,不是 kev 的分母。**

**(c) 与"是否模型不行"的关系**:kev 自述自己的短板(PLAN L8、L9、L11)——knowledge 由 base 决定(MMLU 不因 Kev 而涨),date/长文需要格式技巧。kevcpp 85 题的"反讽/多跳/持股比例"正是这些"需要真实理解/知识"题的汇聚,**即便 kev 也在硬-v1 上只有 0.665(0.8B)**,说明 kev 对这些题的提升也不是无限,是**分布内有**提升。

---

## 6. 判断(一句话)

**kevcpp 缺的最关键一环 = "针对决策任务的权重学习"(LoRA/微调)**,它把 base 模型从"通用语言读出"提升为"能在第一印象里把答案放在可读位置";而 kevcpp 零微调在这环上为 0,所以被锁在 LM-head 通用投影能解出的 ~71%,无法触达 kev 的 90%+。**kev 的高准确率不是读取技巧能白拿的——它是一次明确的监督训练。**

## 附:关键文件索引(取证用)
- kev: `README.md` L283-299, `kev/train.py` L373-415,575-585, `kev/model.py` L82-119,182-201,303-304, `PLAN.md` L166,178-199, 模型卡 `docs/model-cards/kev-0.8b.md`(front-matter: peft/adapter)
- kevcpp: `tests/bench-60cases.jsonl`(85 题,19 类), 本次实验 json 见 `build/correctness_*`