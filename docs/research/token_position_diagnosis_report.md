# 诊断报告 — "答案 token 出现在输出序列第几位 / 首 token 为什么常错"(零训练令牌位置诊断)

> 分支: research/logits-decision-correctness。
> 工具: 新增 `src/token_position_probe.cpp`(直接链接 vendored llama, 不用 llama-cli), target `token_position_probe`。
> 模型: models\Qwen3.5-0.8B-Q8_0.gguf(Qwen3.5-0.8B, Q8_0, n_vocab=248320, 思考型 qwen35 arch)。
> 模板: kevcpp 默认 "{instructions}\n[ {options} ]\n{state}\nAnswer:"。
> 完整输出: build\probe_full10.txt。复现: `token_position_probe --model models\Qwen3.5-0.8B-Q8_0.gguf --steps 10`。

---

## 1. 结论(TL;DR) — 一个决定性发现

**kevcpp"读 `Answer:` 后第一个 token 的 logit"这个做法,机理上就不成立:**

- 8 道代表题里,**7 道的 Lp-1 位置 top1 token 是 `' '(空格)`**——一个纯格式 token,不是答案,不是选项。
- 真实(贪心)生成时,正确答案常出现在**第 2 个(noul 的 `否`)甚至第 7 个(choice 的 `B`)token**才出现,前面被 空格/换行/标签/`thinking`/逐字复述 占据。
- 结论: **不是 0.8B 太笨,而是"读取位置(首 token)+比对对象(false/true 英 token vs 模型心里的 否/是)"双错位。**

---

## 2. 方法

- 直接链接 vendored llama(有 llama.lib)写单文件工具,避免缺 llama-cli 的硬限制。
- 对每题:
  1. **层次1**: decode 全 prompt,读 Lp-1(Answer: 末位)的全量 vocab logits,扫 top-8 并 detokenize,标出是否命中候选 key。
  2. **层次2 贪心滚动**: 取 Lp-1 的 argmax token 追加回序列,再 decode 读下一位 argmax,重复最多 10 步——即"模型真会生成的 token 序列"。
- 8 道题覆盖: 否定/反义 always-yes 重灾(noul c1–c4)、choice(c5/c8)、score(c6)、日期(noul c7)。
- 全程在 decode 前 `llama_memory_clear(...,true)` 避免 M-RoPE X<Y 冲突(每步全量从 pos 0 解码)。

---

## 3. 逐题关键输出

### 3.1 Lp-1 top1 是什么(8 题)

| 题 | qtype | 正确答案 | Lp-1 top1 token | top1 是否=答案? |
|---|---|---|---|---|
| c1 | noul | false | `' '(空格)` | 否(格式 token) |
| c2 | noul | false | `' '(空格)` | 否 |
| c3 | noul | false | `' '(空格)` | 否 |
| c4 | noul | false | `' '(空格)` | 否 |
| c5 | choice | A | `' A'` | 是(但见 3.2: 生成后续进 thinking 段) |
| c6 | score | 2 | `' '(空格)` | 否 |
| c7 | noul | false | `' '(空格)` | 否 |
| c8 | choice | B | `' A'` | 否(正确是 B) |

→ **8 题中 6 题 top1 是纯格式/空格,非答案**;choice 题 top1 是字母标签,但 c8 正确是 B 而 top1 是 A。

### 3.2 贪心滚动(模型真会生成的 token 序列)

- **c1(c2,c4,c7 同型,否定 noul)**: `' ' → '否' → 换行 → ...`。**模型确实想说"否"(正确!)** 但先输出了一个空格,第二个 token 才是 `否`。
- **c5(choice)**: `' A' → 换行 → '<|im_start|>thinking' → '<|im_start|>response' → '根据'→'您'→'提供的'→'数据'...`。**暴露了 Qwen 思考型模型在 plain `Answer:` 后实际进入 thinking/response 推理段**(不是直接给 A)。
- **c6(score)**: `' ' → '客户'→'有多'→'生气'→'？'...`。**在逐字复述题干**,完全没在答题。
- **c8(choice)**: `' A' → '球'→'在'→'1'→'号'→'位'→'，'→'B'`。**正确答案 `B` 出现在第 7 步**,前面先说了错 A 再修正。

### 3.3 为什么 logits 会 always-yes

c1–c4 的 Lp-1 top-8 里,模型给 `yes`(id 9542)的 logit 普遍略高于 `no`(id 874)。kevcpp 拿 `false/true` 这组英 token 在该位置的 logit 做 argmax → 偏向 true → always-yes。但模型"心里想的答案"是中文 `否`(id 96994),与 kevcpp 比对的 `false/true` **根本不是同一套 token**。

---

## 4. 结论与对后续实验的指引

1. **首 token 常是格式/空格 token(7/8 题)**,不是答案 → 印证主会话用户判断: "取 Answer: 后第一个 token 大概率与答案无关"。
2. **真实答案常在更靠后**(noul 第2个 token 的 `否`;choice 可到第7个)。只读 Lp-1 单位置本身存在"读错位"问题。
3. **比对对象错位**: kevcpp 用英文 `false/true`,`模型用中文 `否/是`。即便位置对,比对不了。
4. **思考型模型确实会进入 `<|im_start|>thinking/response` 段**(c5),plain `Answer:` 没把它钉到"给答案"。这解释了 ChatML 实验为何差,也说明真正可控做法是**受控解码到"答案必须紧跟是/否/选项标签"**。
5. **免费生成的语义层面其实常答对**(c1 会说 `否`、c8 会到 `B`),但 logits 单位置比较把它读错了——**不是模型的语义不行,是 kevcpp 的读取方式把它们全读岔了**。

## 5. 据此下一步建议(零训练)

- **把候选 token 对齐到模型真实输出**: noul 用 `否/是`(或 `Yes/No`)而非 `false/true`;choice 用 `A/B` 标签 token。同时**把读取位置从 Lp-1 首 token,改为"跳过第一个空格/格式 token 之后的答案 token"**(或受控到"首 token 必是答案标签")。
- 或更彻底: **真正的受限生成+解析**,在锁死"答案段"后读(避免读到空格/thinking)。
- 这些都仍零训练、纯 prompt/读取层。建议派子代理实测"noul 候选 key 改 `否/是` + 读取点顺延一个 token"。

## 6. 附: 工具产物与复现
- 新增: `src/token_position_probe.cpp` + CMake target。构建 `cmake --build build --config Release --target token_position_probe`。
- 原始输出: `build\probe_full10.txt`。