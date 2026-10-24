# kev 官方评测集接入报告 —— 同卷对比的第一步

> 分支：`research/logits-decision-correctness`
> 背景：此前 kev 声称的 93.8%(0.8B) / 97.6%(4B) / 100%(9B) 一直**无法验证**，因为我们手中没有 kev 的考卷，只能用它自己的 85 题做横向对比。本报告解决这个问题。

---

## 1. 结论：kev 的官方测试集就在本地 kev 仓库里

`E:\van\projects\kev\evals\` 下包含全部官方评测套件（jsonl 格式，`state + questions{key:{type,instructions,criteria,label}}`）：

| 套件 | test 规模 | 题型分布 | 说明 |
|---|---|---|---|
| **hard-v1** | 700 条 / **1088 题** | choice 826 / noul 226 / score 36 | 编程式标签、留出模板（kev 0.8B 模型卡主报表：1,088 questions） |
| **documents-v1** | 574 条 / **936 题** | choice 936 | CFPB 投诉叙述阅读理解 |
| **decision-v7** | 1176 条 | – | 冻结套件，10 个公开数据源 × 1000（`evals/v7/decision-v7`） |
| **transfer-v2** | 668 条 | – | 从未训练过的源（泛化测试） |

hard-v1 的 7 个族：`tradeoff`(200) / `long_policy`(200) / `judge`(188) / `probability`(158) / `multi_hop`(125) / `temporal_numeric`(117) / `ambiguous`(100)。

---

## 2. 一个关键发现：kev 的题比我们的 85 题难得多

### 2.1 选项数量差异（结构性难度）

| 选项数 | kev hard-v1 (choice 826 题) | 我们的 85 题 (choice 40 题) |
|---|---:|---:|
| 2 | 0 | 12 |
| 3 | 138 (16.7%) | 14 |
| 4 | 124 (15.0%) | 14 |
| **5** | **551 (66.7%)** | 0 |
| 6 | 13 (1.6%) | 0 |

**kev 的 choice 题 2/3 是 5 个选项**（随机盲猜基线 20%），我们最多 4 个选项（平均 2.9 个，盲猜基线 ~33%）。

### 2.2 题目类型差异

- kev 含 `long_policy`（200 条，state 最长 18,226 字符 ≈ 3,925 token）、`probability`（需精确概率计算）、`multi_hop`（多跳推理）。
- 我们的 85 题偏"抽取式判断/比较"，单条 state 通常是 1-2 句。

### 2.3 实测对比（35B-A3B + combo + `正确选项：`）

| 评测集 | 题量 | accuracy |
|---|---:|---:|
| 我们的 85 题 | 85 | **87.1%** |
| kev hard-v1 短题（不含 long_policy） | 38 | **36.8%** |

> **同一模型、同一配置，跨到 kev 的考卷上掉 50pp。**
> 这直接说明：**我们此前 87.1% 与 kev 的 97.6%/100% 完全不可比** —— 题目难度不在一个层级。

---

## 3. 接入过程中修复的 3 个真实 bug

### 3.1 `criteria` 值为 `null` → 服务器 422

kev 部分数据集（如 `tradeoff`）写 `criteria: {quarry_sql: null, tidedb: null, ...}`，语义是"选项描述就是键本身"。
我们的服务器要求非空字符串描述 → 422。**修法**：转换时把 `null` 替换为键名（与 kev 判定语义一致：label 是键）。

### 3.2 上下文容量不足 → 422 `request too long`

服务器并发 8 流，`--ctx 2048` 时每流仅 **256 token**；`--ctx 16384` → 1280/流；`--ctx 49152` → 3584/流。
kev `long_policy` 最长需 **3,925 token** → 必须 `--ctx 65536`（验证通过）。

### 3.3 （最关键）harness 超时导致服务端**并发槽位泄漏** → 大批 422 `no free concurrent sequence slot`

- 现象：跑着跑着，大量请求返回 422，且错误从 long_policy 扩散到 judge/temporal_numeric 等短题。
- 根因：harness 默认 `timeout=120` 秒。CPU 上 35B 处理 3-4k token 文档需数分钟 → 客户端超时中断，但**服务端未释放该请求占用的并发槽位** → 槽位泄漏 → 后续全部 422。
- **修法**：`post()` 超时改为可配，默认 **3600 秒**（`KEV_HARNESS_TIMEOUT`）。
- **验证**：修复后短题 smoke 的 **HTTP 错误从 26 → 0**。

> 这个 bug 值得记录：**任何长输入 + 短客户端超时的组合都会触发槽位泄漏**，是生产环境的隐患。

---

## 4. 新增工具（可复用）

| 文件 | 作用 |
|---|---|
| `build/convert_kev_suite.py` | kev 评测集 → kevcpp harness 格式转换（支持 `--limit`；null-criteria 修正） |
| `build/correctness_harness.py` | 新增 `--cases` / `--out`，可跑任意用例集；超时改为可配 |

用法：
```powershell
python build\convert_kev_suite.py E:\van\projects\kev hard-v1 test build\kev-hard-v1-test.jsonl
.\build\bin\Release\kevcpp.exe --model <gguf> --backend cpu --port 18260 --ctx 65536 --quiet --decide-config config\decision.best.json
python build\correctness_harness.py --port 18260 --model 35B-kev-hardv1 --cases build\kev-hard-v1-test.jsonl --out build\correctness_35B-kev-hardv1.json
```

---

## 5. 意义与下一步

**意义**：终于可以做**同卷公平对比**。kev 的 1,088 题 hard-v1 是它自己声明的主报表考卷，我们的引擎在同一份题目上的成绩才是真实对标。

**下一步**：
1. 跑完 35B 全量 1,088 题（进行中）—— 得到真实锚点。
2. 跑 4B 全量（更快）—— 判断"零微调小模型"在 kev 难度上的上限。
3. 按族（family）拆解错误：预期 `long_policy`/`probability`/`multi_hop` 是主要失分点，需要针对性改进（而非继续调模板）。
4. 若需要，再测 `documents-v1`(936 题，CFPB 阅读) 与 `decision-v7`(1176 题) 以覆盖 kev 的全部宣称口径。
