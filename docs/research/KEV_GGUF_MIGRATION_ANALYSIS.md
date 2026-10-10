# kevcpp 从「基座+LoRA+head」迁移到「原版开源 GGUF + 零 LoRA/零 head」的 duediligence 实现方案

> 分析模式：只读、只分析、只输出方案。未改任何源码（本文档落在 `docs/research/`，不触碰 `src/`）。
> 依据统一标注 `文件:行号`；读不到 / 不确定的明确写「待核实」。
> 所有源码结论均来自实读 `E:\van\projects\llama.cpp`（最新参考，含 decision 支持）与 `E:\van\projects\kevcpp\src`。
> 生成时间：由 kev-gguf-migration-analyst 子代理产出，供主会话审阅决策。

---

## Part 1 — llama.cpp decision 通道的确切契约

### 1.1 KEV 决策的完整触发条件

核心判定函数 `common_get_decision_type(const llama_model*)`（`E:\van\projects\llama.cpp\common\common.cpp:1186-1196`）：

1. 读 `general.architecture` 字符串，取不到则返回 `COMMON_DECISION_TYPE_NONE`（:1188-1190）。
2. 拼键 `"<arch>.decision.type"`，读不到则返回 `NONE`（:1191-1194）。
3. 用值在 `COMMON_DECISION_TYPE_NAMES`（:1165-1175）里反查；`"kev"` → `COMMON_DECISION_TYPE_KEV`（:1168）；未知值 → `COMMON_DECISION_TYPE_UNKNOWN`。

枚举定义在 `common.h:957-969`（`:961` 为 `COMMON_DECISION_TYPE_KEV`，注释：dot product of hidden states of last token and of one end token per option）。

**结论：KEV 唯一必需元数据键就是 `<arch>.decision.type="kev"`，GGUF STRING 类型。** 缺此键 → 退化 `NONE`（决策功能不启用）。另有一个 `common_get_decision_type(fname)` 重载（:1198-1231）做不加载模型的文件级校验：文件/架构缺失 → `UNKNOWN`，键缺失 → `NONE`，值非 STRING → `UNKNOWN`。

另外 `common_init_result` 构造（:1287-1296）对 KEV/LAYA/CLEF/LFM2_D1_OMNI 强制：
- `params.embedding = true; params.pooling_type = LLAMA_POOLING_TYPE_NONE`
- `cparams.embeddings = true; cparams.pooling_type = LLAMA_POOLING_TYPE_NONE; n_outputs_max = n_batch; n_outputs_max_per_seq = 1`

即 **KEV 依赖 embeddings 输出 + pooling NONE（逐 token 输出隐层）**。

### 1.2 `server_decision_context::init(model)` 读哪些字段 / 默认值

`E:\van\projects\llama.cpp\tools\server\server-decision.cpp:33-141`：

| 字段 | 来源 | 默认/行为 |
|---|---|---|
| `type` | `common_get_decision_type(model)`；`NONE` → 直接 return（:36-39） | `COMMON_DECISION_TYPE_NONE` |
| `tmpl` | `llama_model_chat_template(model, "systemone")`，**必需**，无则抛 `std::runtime_error`（:46-50） | `nullptr`（抛错） |
| `temperatures` | 扫所有 `prefix+"temperature.<type>"` / `prefix+"temperature.<type>.<bucket>"` 键（分桶如 `choice.2`/`choice.3_5`，见 :52-67 与 :1010-1020） | 空 map → `get_temperature` 返回 **1.0**（:1028） |
| KEV 专属 `token_marker` | `common_tokenize(vocab, "<|box_end|>", false, true)`，必须恰好 1 个 token，否则抛异常（:100-107） | `LLAMA_TOKEN_NULL` |
| `n_options_max` | KEV 分支设为 **255**（:107） | `255` |
| LAYA 专属 `token_marker`/`token_sep` | `llama_vocab_mask`/`llama_vocab_sep`（:109-110） | — |
| `max_head_tokens` | 读 `prefix+"max_head_tokens"`（仅 LAYA，:116-120） | — |
| `pointer`/`markers`/`column` | **不在这里读**，在 `fill_task` 阶段写入 `server_task.decision` | — |

**KEV 需要的 GGUF 元数据极简：`<arch>.decision.type="kev"` + `systemone` chat template。`temperature.*` 全部可选。**

### 1.3 KEV prompt 怎么组（fill_task）

`server-decision.cpp:652-705` + `render()`（:581-650）：

- `render()`：把 `{id, type, instructions, state, options}` 喂给 **`systemone` jinja 模板**（:645-649）。
- KEV 时 `state`/`instructions` 走 `decision_kev_text()`（正则把 `<|x|>` 改写为 `<¦x¦>`，:395-398，应用于 :616-619）；每个 option 的 `key`/`description` 也经 `decision_kev_text()`（:564-569）。
- 模板产出的**文本**再 `common_tokenize(vocab, prompt, false, true)`（parse_special=true，:688）。
- `fill_task` 里 KEV 分支（:692-703）：
  - 扫描 token 流，凡等于 `token_marker`（`<|box_end|>`）的位置塞进 `task.decision.markers`；
  - `markers.size()` 必须等于 option 个数，否则抛「unexpected layout of the decision prompt」（:699-701）；
  - `task.decision.pointer = tokens.size()-1`（:702）—— **question 取 prompt 最后一个 token 的隐层**。

> **关键对照**：kevcpp 现有布局是 `[<|fim_prefix|> state] 每问 [<|fim_middle|> instr (<|box_start|> opt <|box_end|>)* <|fim_suffix|>]`，`<|fim_suffix|>`（decide 收尾）在 branch 末、《|box_end|> 是每个 opt 的收尾（`kev_encode.h:44-45`、`kev_encode.cpp:79-80,107`）。这与 llama.cpp KEV 的「option 读 `<|box_end|>`、question 读 `tokens.size()-1`」**位置契约一致** —— kevcpp 现有 marker 布局天然能映射到 llama.cpp KEV 的 markers/pointer 语义。`text_marker`/`token_sep` 是 **LAYA** 用（:116-120、:719 起校验 sep），KEV 不使用。

### 1.4 `send_decision()` 的 KEV 打分矩阵

`E:\van\projects\llama.cpp\tools\server\server-context.cpp:2328-2407`，KEV 走 embeddings 分支（无 labels）：

- 取本 batch 中 `seq_id==slot.id && output` 的所有位置到 `idx`（:2359-2364）。
- `n_embd_out = llama_model_n_embd_out(model_tgt)`（:2381）；**`n_pointer = n_embd_out / 2`**（:2382）。
- question：`embd_q = llama_get_embeddings_ith(decision.pointer)`，即最后一个 token 的隐层（:2383，取值逻辑 :2366-2369）。
- 每 option（:2386-2401）：
  ```
  dot = Σ_{i=0..n_pointer-1}  embd_q[i] * embd[marker][n_pointer + i]
  score = dot / sqrt(n_pointer)
  ```
  即 **隐层对半切 `[q|k]`，question 用前一半 q、option 用后一半 k，点积再除以 √(n_pointer)**。
- `temperature` **不在 send_decision 里用**，而在 `format_answer` 的 softmax 步 `exp((s_i - score_max)/temperature)`（`server-decision.cpp:1078-1088`）；`get_temperature`（:1010-1029）无键默认 1.0。

> 尺寸：Qwen3.5-0.8B `n_embd=1024`、`n_embd_out=1024`（kevcpp `tests\kab_default_err.txt:96-98` 佐证），⇒ `n_pointer=512`。**注意这与 kevcpp PointerHead 的投影维 `dim_proj=256`（`kev_head.h:4,59`）完全不同。**

### 1.5 这些是否都在 server 层？有无 common/gguf 可复用 API？

| 能力 | 位置 | 可复用性 |
|---|---|---|
| KEV 类型判定 | `common.cpp:1186-1231` + 自动开 embeddings/pooling NONE（`common.cpp:1287-1296`） | ✅ **common 层库级可复用**（`libllama-common`），不依赖 server |
| KEV prompt 组装/模板 render/markers/pointer | `server-decision.cpp` fill_task/render | ❌ 仅 `tools/server`，非库 API |
| KEV 打分（半边点积） | `server-context.cpp:2381-2401` | ❌ 仅 `tools/server` |
| 温度 softmax / confidence 公式 | `server-decision.cpp:1010-1133` | ❌ 仅 `tools/server`（公式简单，可在 kevcpp 复刻） |

**结论**：KEV 的「prompt 编排 + 打分 + 温度」没有任何位于 common/gguf 的库级 API 可直接调用，全在 llama-server 进程内。要走原生通道，要么升级 vendored 并启用 server/common/tools 构建（改动大），要么让 kevcpp 作为 llama-server 的 HTTP 客户端（旁挂）。

---

## Part 2 — 两条改造路线的具体实现方案

### 现状基线（kevcpp vendored 与机制）

- vendored llama.cpp = **0.5.0-dev**（`E:\van\projects\kevcpp\third_party\llama.cpp\CMakeLists.txt:6-8`），**无任何 decision/KEV 支持**（vendored 全树 grep 未见 decision 枚举/逻辑）。构建经 `add_subdirectory(third_party/llama.cpp)`（kevcpp `CMakeLists.txt:71`），非 standalone 时 common/tools/examples/tests 都不构建（:41-45）。
- kevcpp 走 llama-free 后端 C-ABI：`src/kev_backend.h` ops 表 + `src/kev_backend_cpu.cpp`；`read_hidden` 经 `llama_get_embeddings_ith`（`kev_backend_cpu.cpp:206`）。
- `KevModel::init`：`bopt.lm_head=0`（`kev_model.cpp:112`），`head_.dim_in()!=0 && !=model_n_embd` 校验（`kev_model.cpp:118-121`，head 硬编码 1024）。
- `rows_from_slot` 里 `probs[qi]=head_.probs(hd.data(), optp)`（`kev_model.cpp:222`）——**唯一喂 PointerHead 的位置**。
- marker 词表 `SPECIAL[5]=<|fim_prefix|>,<|fim_middle|>,<|box_start|>,<|box_end|>,<|fim_suffix|>`（`kev_encode.h:44-45`）。**Qwen3.5 原版词表本身就含这些特殊 token**（kev 当初据此选用），所以原版 GGUF **无需改词表**即可产出 KEV 布局 —— 这是两方案可行性的前提。

### 路线 A：kevcpp 内部自实现 KEV 打分（改动最小，不升级 vendored）

**核心思想**：保留 0.5.0 vendored、保留 backend 插件机制、保留 encode/prefix/LRU，只把 `head_.probs` 换成「llama.cpp KEV 半边点积」的无权重打分器。

#### A-1 新增打分模块
**新增 `src/kev_kev_head.h/.cpp`**（仿 `kev_head.h/.cpp` 结构，但**去掉权重**）：
```cpp
// score_option(h_q, h_opt, n_embd, temperature) -> float
//   n_pointer = n_embd / 2;
//   dot = Σ_{i<n_pointer} h_q[i] * h_opt[n_pointer+i];
//   return dot / sqrt(n_pointer);
// probs: softmax(scores / temperature)
```
（公式取自 `server-context.cpp:2381-2401` + `server-decision.cpp:1078-1088`。）

#### A-2 改 `src/kev_model.cpp`
- **:118-121 dim 硬编码解绑**：不再依赖 `head_.dim_in()==1024`，改为一律用 `backend_->model_n_embd()` 作为 KEV 维度 `n_embd`（即 1024）。删掉 head.bin 加载分支（:115-117）与对 `head_` 的依赖。
- **:222 `head_.probs(...)`** → 调用新的 KEV 打分器（`probs_kev(hd, optp, /*n_embd=*/dim)`）。`hidden_at_output`/`read_hidden` 读取路径**整体复用不变**（`kev_model.cpp:150-153,213-220`）。

#### A-3 改 `src/kev_model.h`
`:95-96` 的 `dim_in()/temperature()` 改为：`dim_in()=model_n_embd()`、`temperature()`=配置注入或默认 1.0；成员 `head_`（:119 `PointerHead head_`）改 `KevKeVHead head_` 或去掉。`init` 不再要求 `head_path`。

#### A-4 改 `src/main_server.cpp`
- `--head` 改为可选/删除，不再要求 head.bin（:74,83,97,133 起的 KevModelOptions head_path）。
- 启动日志 `model.dim_in()`（:141）、`model_card` 的 `temperature`（:68）改为 `model_n_embd` / 配置或默认。prewarm（:150-170）、warm_graph（:176-183）无需改。

#### A-5 lm_head 是否仍需为 0？
**仍置 0，没问题**。KEV 读的是 embeddings 输出层（lm_head 之前的隐层），`bopt.lm_head=0`（`kev_model.cpp:112`、`kev_backend_cpu.cpp:115`）正好只输出隐层；当 `n_embd_out==n_embd`（Qwen3.5 成立）半边点积维度 = `n_embd/2` 正确。

#### A-6 prompt marker 是否需用户配置？是否注入 GGUF 元数据？
- 原版 Qwen3.5 自带全部 5 个 special token，**默认可用、无需配置/无需改词表**。但建议在配置里提供「option 结束 marker」（默认 `<|box_end|>`）与「question=最后一个 token」两个可覆盖项，以备换用不含这些 token 的开源模型。
- **路线 A 不需要注入 GGUF decision 元数据**：打分完全在 kevcpp 侧自算，绕过 llama.cpp decision 机制；`<arch>.decision.type="kev"` 只对 llama-server 有意义，对 kevcpp 无害但**无作用**（可注明不做）。

#### 路线 A 改动文件清单

| 文件 | 操作 | 说明 |
|---|---|---|
| `src/kev_kev_head.h` / `.cpp` | **新增** | 无权重 KEV 半边点积打分 + 温度 softmax |
| `src/kev_model.cpp` | 改 | 去 head.bin 加载与 1024 硬校验；:222 换 KEV 打分；维度用 model_n_embd |
| `src/kev_model.h` | 改 | 成员 PointerHead→KEV 打分器；dim_in/temperature 语义 |
| `src/main_server.cpp` | 改 | `--head` 可选化；日志/卡片读 model_n_embd |
| `CMakeLists.txt` | 改 | m3_bench/main_server 源列表加 `kev_kev_head.cpp` |
| `tests/` 对拍工具 | 可选 | 加 KEV 打分与 Python 参考的对拍 |

> 不改 vendored llama.cpp、不改 backend ABI、不动 encode/prefix/LRU。

### 路线 B：kevcpp 走 llama.cpp 原生 server 通道

#### B-1 升级 vendored llama.cpp 到含 decision 版本（改动面大）
- vendored 目前非 standalone 不构建 common/tools/server；而 KEV 全在 server 层，需启用 common/tools 构建、把 `server-decision.cpp`/`server-context.cpp` 及其依赖拉进 kevcpp 构建。
- 上游 0.5.0 → 最新有大量 API/结构重构，需重做 backend ABI 对 `llama_get_embeddings_ith`/`n_embd`/`n_embd_out` 的适配（`kev_backend_cpu.cpp`）。
- **不推荐首选**：改动面大、风险高、收益低 —— kevcpp 的「编排 + 前缀缓存 + LRU + 批量 branch decode」本就是 llama-server 决策流程的替代物，硬塞 server 通道反而牺牲现有架构。

#### B-2 kevcpp 作为 llama-server 的 HTTP 客户端（旁挂）
- 部署：独立一个 `llama-server` 进程加载**注入过决策元数据**的 GGUF；kevcpp `main_server` 变成 `POST /v1/systemone` 的**代理/转发**（收到请求 → 转发 → 取 answers 原样返回，保留 auth/日志/usage 包装）。
- 该路线对「任意开源 GGUF」的工作方式：
  1. **注入元数据（必须）**：用户给一份轻量配置（YAML/JSON），用 `gguf` 工具往原版 GGUF 写 `<arch>.decision.type="kev"` + `systemone` 模板；或让 kevcpp 加载时经 gguf API 临时注入（不落盘）。**原版 Qwen3.5 没有这两项，不注入则 `common_get_decision_type`→NONE、且 `init` 因无 systemone 模板抛异常（server-decision.cpp:46-50）。**
  2. 打分/接口成本全交 llama-server。
- 优点：几乎零 C++ 改动（只改 main_server 转发 + URL/超时配置）；缺点：多进程/端口部署、多一跳延迟、需注入工具链；且 KEV 打分的 V3 风险同样存在，但在 kevcpp 侧**难插探针对拍**。

> B 两条写法二选一：B-1 改动面大不推荐；B-2 改动面小但引入「注入元数据 + 旁挂 server」运维成本。

---

## Part 3 — 给用户的推荐结论

### 3.1 A vs B 对比

| 维度 | 路线 A（kevcpp 内自实现 KEV） | 路线 B（走 llama-server 原生） |
|---|---|---|
| C++ 改动量 | 小（~4 文件 + 1 新增模块） | B-1 大；B-2 小（仅转发） |
| 是否动 vendored llama.cpp | ✗ 不动 | B-1 动（大风险）；B-2 不动 |
| 是否需注入 GGUF 元数据 | ✗ 不需要 | **B-2 必须**（`decision.type`+`systemone` 模板） |
| 复用现有 decode/前缀缓存/LRU | ✅ 完全复用 | 放弃（交给 llama-server） |
| 对「任意开源 GGUF 免微调」支持 | ✅ 只要自带 marker token，否则需配置覆盖 | ✅ 但要配注入管线 + 旁挂进程 |
| 可对拍/可插探针（验证 V3 风险） | ✅ 高可观测 | 中/难 |

### 3.2 关键风险（V3）

**对未经 kev 训练的原版 Qwen3.5-0.8B 基座直接用「半边点积 / √n」打分，是否真能正确决策 —— 是高风险的待验证项，必须在实现后第一步验证。**

依据：llama.cpp 的 KEV `dot/√n`（server-context.cpp:2396-2400）是一个**无任何可学习参数**的启发式相似度；而 kevcpp 现有 PointerHead 是**训练过的 1024→256 线性层 q/k**（`kev_head.h:3-8`，`kev_head.cpp:25-37,98-99`）。原版 Qwen3.5-0.8B 从未按 KEV 目标微调，其上 `[q|k]` 半切点积与「要选的选项」之间没有保证对齐 —— 可能输出近似均布或随机偏置。**llama.cpp 的 KEV 通道本身是为经过对应对齐/评分的决策模型设计的，把未训练基座硬套上去是其定义外的用法。**

### 3.3 实现完成后第一步应做什么验证

先用 **路线 A 的最小原型**（复用现有管线、改动最小、可插探针）做一次「基座裸 KEV」可行性实验，勿急于上生产：

1. 用原版 `models/Qwen3.5-0.8B-Q8_0.gguf` 跑一个已知答案的判断题/选择题集（gold labels），对拍：
   - 路线 A 的裸 KEV 半边点积 top-1；
   - 现有 PointerHead 基线（临时用带 head 的旧路径跑同样 case）；
   - 记录 accuracy、概率是否畸变（恒近 1/K、方差过小等）。
2. 仅当裸 KEV 达到可接受精度才继续（可试小 temperature 档、不同 option 描述写法）。
3. 若精度不达标：B 原生通道（同一 `dot/√n` 公式，数学等价）**也不会更准**；此时向用户如实报告「零 LoRA 裸基座决策不可行」，再考虑微调/回退。
   - 注：路线 A 与路线 B 的 KEV 打分**数学上是同一公式**（同 half-split + dot/√n + temperature softmax），因此 A 的验证结果可外推到 B 的结论；B 独有差异仅模板/注入。

> **推荐：先用路线 A 做可行性原型与量化验证（低成本、高可观测）→ 验证通过再决定是否推进路线 B 的「旁挂 llama-server + GGUF 元数据注入」生产形态。B-1（升级 vendored 拉 server 构建）改动面大、收益低，不建议。**

### 3.4 「配置」在这个路线里承担什么

一份轻量 **YAML（或 JSON）** 决策配置，两路线各承担：

```yaml
model:
  path: models/Qwen3.5-0.8B-Q8_0.gguf   # 原版基座
  decision_type: kev                      # 路线B: 注入 <arch>.decision.type="kev"
  systemone_template: |                   # 路线B: 注入 GGUF 的 "systemone" chat 模板
    ...
kev:
  option_end_marker: <|box_end|>          # 路线A: option 结束 marker（默认值）
  question_at_last_token: true            # 路线A: question 取最后一个 token
  temperature: 1.0                        # 两路线共用（可选分桶）
backend:
  name: cpu                               # 或 gpu/vulkan
  ctx: 8192
```

- **路线 A**：配置只承担「温度 + marker 覆盖」等推理调参，**不含注入**；缺省也能跑。
- **路线 B-2**：配置承担「从 config → 往 GGUF 注入元数据」的驱动（`decision_type`/`systemone_template` 写进 GGUF 或运行时临时注入），外加 `llama-server` 的地址/超时。
- 原版 Qwen3.5 自带 marker token，故**无需为单词表配 token 表**；换不含这些 token 的开源模型时才需要配置覆盖 marker。

---

## 附：证据索引备忘

- KEV 枚举/判定：`common.h:956-975`、`common.cpp:1165-1231`
- KEV 自动开 embeddings/pooling NONE：`common.cpp:1287-1296`
- `server_decision_context` 结构/`init`：`server-decision.h:32-151`、`server-decision.cpp:33-141`
- KEV prompt 组装：`server-decision.cpp:581-650, 652-705`
- `task.decision` 结构（labels/markers/pointer/column/order）：`server-task.h:178-202`
- KEV 打分矩阵：`server-context.cpp:2328-2407`（尤其 2381-2401）
- 温度 softmax / confidence / `get_temperature`：`server-decision.cpp:1010-1133`
- vendored 版本：`third_party/llama.cpp/CMakeLists.txt:6-8`；无 decision 支持（vendored 全树 grep）
- kevcpp head：`kev_model.cpp:112,115-121,222`；`kev_head.h:3-8`；`kev_head.cpp:25-37,98-99`；`kev_model.h:95-96,119`
- marker 布局：`kev_encode.h:44-45`、`kev_encode.cpp:49-114`
- HTTP 层：`kev_api.cpp:197-261`、`main_server.cpp:74-147,202-243`
- backend hidden 读取/dim：`kev_backend_cpu.cpp:102,114-115,206,252-254`
- llama.cpp 最新参考（含 decision）源码：`E:\van\projects\llama.cpp\...`（已直接实读核实）

**待核实/未验证标注**：
- `llama_model_n_embd_out` 在 Qwen3.5 上是否恒等于 `n_embd`（当前 test 文件佐证=1024，未在运行时对 Qwen3.5-0.8B 实测）——路线 A 用 `model_n_embd` 即可规避。
- 原版 Qwen3.5-0.8B 是否恰含全部 5 个 `SPECIAL` token（`kev_encode.h:44-45`）——kev 当初选用暗示存在，需 init 时用 `special_token_id`/GGUF 词表实测确认。
- KEV 半边点积在未训练基座上的决策质量：**V3 未验证，必须实验**。
- 路线 B 注入元数据时 `gguf` 工具的具体 CLI/API 名称：未核。