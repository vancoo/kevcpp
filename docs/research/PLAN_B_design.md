# 方案 B 落地设计文档：零微调下载开源 GGUF + 轻量决策配置，走 llama.cpp 原生 KEV decision 通道

> 保存路径：`E:\van\projects\kevcpp\docs\research\PLAN_B_design.md`
>
> 作者：kevcpp 设计子代理 | 上级主会话：`session-818fb7ea-4072-4fd2-a5b1-dea9ec83`（以实际为准）
> 状态：**设计文档（只调研只设计，未改任何源码、未运行任何改写命令）**

---

## 0. 本设计基于的已核实事实（来源标注）

**上游 llama.cpp 事实（子代理用 web_fetch 直连 llama.cpp master 源码核实）**

- ✅ **已核实**：llama.cpp master（`ggml-org/llama.cpp`，2025 之后）的 `common/common.h` 中 `enum common_decision_type` 明确包含：
  ```cpp
  COMMON_DECISION_TYPE_NONE,
  COMMON_DECISION_TYPE_OPENJEV,  // logits of one label token per option
  COMMON_DECISION_TYPE_LEV,      // + noul rating scale
  COMMON_DECISION_TYPE_KEV,      // dot product: last-token hidden · one end token per option
  COMMON_DECISION_TYPE_NIMBLE,
  COMMON_DECISION_TYPE_LAYA,     // score of one marker token per option, embeddings output
  COMMON_DECISION_TYPE_CLEF,     // all questions in one prompt
  COMMON_DECISION_TYPE_PPLX_DECIDER,
  COMMON_DECISION_TYPE_LFM2_D1, COMMON_DECISION_TYPE_LFM2_D1_OMNI,
  COMMON_DECISION_TYPE_UNKNOWN,
  ```
  并提供 `common_get_decision_type(const llama_model*)` / `const std::string & fname`。**支持 `/v1/systemone` 与 KEV 决策通道已确认在 master 中存在**。
- ✅ **已核实**：`tools/server/server-decision.h` 的 `server_decision_context` 支持 `init(const llama_model*)`，注释原文：`read the "<arch>.decision.*" metadata, type stays NONE if the model has none`。其私有字段：
  - `temperatures`（`std::map<std::string,float>`，键为 `<type>` 或 `<type>.<n_options bucket>`）
  - `noul_true_first`、`choice_sorted`
  - `labels` / `label_texts`（OPENJEV/LEV/NIMBLE/PPLX_DECIDER 用）
  - `token_marker` / `token_sep` / `text_marker` / `max_head_tokens` / `max_option_tokens`（LAYA/KEV 用）
  - 决策问题有 `type ∈ {CHOICE, SCORE, NOUL}`，可 `can_share_prompt()`（KEV 为 true）、`is_joint()`（CLEF）、`format_answer()`。
- ✅ **已核实**：master 中决策打分有两条路径（子代理沿用主会话当日直连 `server-context.cpp` 的结论）：(a) `decision.labels` 非空 → 候选 token logits 打分；(b) KEV 指针头——把每 token 的 embedding 对半切，前半当 q、后半当 k，`score = dot(q,k)/sqrt(n_pointer)`，question embedding 取 `decision.pointer` 位置，每个 option 取 `decision.markers` 位置。
- ✅ **已核实**：llama.cpp 支持 `-hf user/model[:quant]` 直接下载开源 GGUF 运行。

**kevcpp 本地事实（子代理直读工作区核实）**

- ⚠️ **vendored llama.cpp 版本**：`third_party/llama.cpp` 为 **0.5.0**（`LLAMA_VERSION_MINOR=5`），HEAD commit `068d0902`，分支 `build-portable-backends`。子代理对该目录 grep `COMMON_DECISION_TYPE|/v1/systemone|server_decision` **零命中**，且该目录**没有** `common/` 目录（`include/` 下只有 `llama.h` / `llama-cpp.h`）。→ **当前 vendored 版本不含 decision 支持**，走 B1 必须做较大幅度上游同步（见 §6-a）。
- ✅ 编排在 `src/kev_model.cpp`：`init()` 中 `bopt.lm_head = 0`（kev_model.cpp:112，hidden-only serving）；`head_.dim_in() != backend_->model_n_embd()` 时拒绝（:118-121，即硬编码 1024）。`probs()` / `rows_from_slot()` 走 `head_.probs(hd, optp)`（kev_head.cpp 的 PointerHead）。
- ✅ 已有 llama-free C-ABI 后端抽象 `src/kev_backend.h`（`kev_backend_ops` 固定函数指针表，含 `read_hidden`、`decode_seq`、`decode_batch`、`tokenize`、`special_token_id`、`model_n_embd` 等），`src/kev_backend_factory.cpp` + `kev_backend_cpu/gpu/vulkan.cpp` 构成插件机制（`BackendFactory_load`）。
- ✅ HTTP 在 `src/main_server.cpp` + `src/kev_api.cpp`，响应 `{model, answers, usage, latency_ms}`。

---

## 1. 目标与范围

### 1.1 方案 B 的定义
> **下载一个未做过任何 kev 训练的通用开源 GGUF 模型 + 一份轻量决策配置文件**，**零 LoRA 微调、零独立 head 权重文件**，直接复用 llama.cpp 原生「KEV decision 通道」做 system-one 决策推理（一次前向、不生成 token，输出每个候选选项的打分/概率）。

即：**模型权重完全来自开源仓库，决策逻辑完全由 llama.cpp 原生 KEV 指针头完成**。本地唯一新增的是「描述如何组装 prompt / 每个选项的标记 token / 温度 / 映射」的配置，不训练、不加载 `head.bin`、不叠 LoRA。

### 1.2 方案 B 做到什么
- 交付一份**可评审、可拆解实施**的落地设计：两条接入路线、决策配置 schema 与示例、数据流、对 kevcpp 的改动点清单、风险与验证方案。
- 明确如何在「承诺普通开源模型也能出有判别力的 KEV 分数」这一**待验证正确性假设**成立的前提下，把 KEV 决策通道接入 kevcpp 导出路径。

### 1.3 方案 B 不做什么（边界）
- **不做代码实现**。本文档是设计稿；任何 `src/*`、`third_party/llama.cpp/*` 的改动都是后续实施阶段的事（本子代理铁律：不改源码、不跑改写命令）。
- **不做可行性承诺**。KEV 对未训练开源模型的判别力属于**待验证科学假设**，本设计只给出最小成本的验证方案（§7），不上诉成功。
- **不评估现有内置逻辑的废弃**。kevcpp 现有 `kev-merged-q8.gguf + head.bin` 的 Rank-1/LoRA-fold 路径保留不动；方案 B 是并行新增通道，不触碰它。
- **不做配置文件的完整字段语义逆向**：`<arch>.decision.*` 的**全部**合法键与取值语义（如 `text_marker` 具体作用、`n_options_bucket` 命名规则）在面向 master 源码直读时未 100% 穷举确认，涉及处标注**待验证**。

---

## 2. 两条接入路线对比

### 路线 B1：直接升级/使用 llama.cpp 原生 `/v1/systemone`

**形态**：把 vendored llama.cpp 同步到含 decision 支持的 master，让 kevcpp 由两种方式提供决策：(i) 依赖一个独立的 llama-server 进程暴露 `/v1/systemone`，kevcpp 只是 client（最省事）；(ii) 把 `server_decision_context` 直接内嵌进 kevcpp 进程复用其逻辑。

**适用场景**
- 想最快吃到 llama.cpp 官方对 KEV/OPENJEV/LAYA/CLEF/PPLX_DECIDER 等多种决策类型的持续维护，只需下载开源 GGUF（如 HIT-TMG/kev 系或任何自带 `<arch>.decision.type=kev` 元数据的模型）。
- 可接受「决策服务独立进程 / 或 kevcpp 与 llama.cpp 高度同构」。

**改动面（当前 vendored = 0.5.0 不含 decision，这点决定 B1 成本高）**
- `third_party/llama.cpp`：从 0.5.0 + 本地 `build-portable-backends` 分支**大幅同步到 master**。需评估：kevcpp 的 CPU/Vulkan 后端是通过 `llama-cpp.h` C ABI 封装的，同步后 `llama-cpp.h` / `llama.h` 的 public API 若变（多 token 位置、MROPE 位置数组、embd 携带等，master 中 `common_batch::token` 已含 `std::array<llama_pos, GGML_MROPE_SECTIONS> pos`），`kev_backend_*.cpp` 的封装层要跟着改。
- 若走「独立 llama-server」形态：kevcpp 主程序**几乎不动**，只加一个 HTTP client 到 `/v1/systemone`，把响应转成 `{model, answers, usage, latency_ms}`。改动集中在 `src/kev_api.cpp` / `src/main_server.cpp` 加一个路由。
- sum: 路线 B1 的**正确性由 llama.cpp 官方背书**，但**改 vendored llama.cpp 的风险是全部路线里最高的**（见 §6-a）。

### 路线 B2：复用 kevcpp 现有 backend 插件机制，新增 `kev-backend-openkoai` 插件

**形态**：写一个新后端插件 dll（对照 `kev-backend-cpu.cpp` 的 C-ABI 实现），内部：
- 照常调 llama.cpp 读取 hidden（`llama_get_embeddings_ith` / 对应 read_hidden 语义），**直接复用现有 CPU/Vulkan 后端**，不必为它重写推理核心；
- 但 `probs`/打分这一步**在 dll 内实现 KEV 点积分**（把每个 token 的 embedding 对半切开，q=前半，k=后半，score = dot(q,k)/sqrt(n_pointer)，contiguous softmax）；
- 决策所需的「配置」由 dll 加载一份独立轻量配置文件（§3）。

**适用场景**
- 想让 **kevcpp 主程序基本不动**（`kev_model.cpp` / `kev_api.cpp` 只做最少接线），把「KEV 解读 + 组装 + 打分」全部隔离在插件内。
- 不想升级 vendored llama.cpp（保持 0.5.0），规避 B1 的最大风险点。
- kevcpp 已有 `BackendFactory_load` 插件加载与 `KEV_BACKEND_ABI_VERSION=1` 版本机制，加插件对存量插件零影响。

**改动面（显著小于 B1）**
- **新增**：`src/kev_backend_openkoai.cpp`（+ .h 或直接对照现有插件样板），CMake 增加该插件目标。
- **新增**：一份决策配置文件（YAML/JSON，§3）。
- **主程序改动**：仅在 `kev_model.cpp::init()` 的硬编码判别处（:118 `head_.dim_in()!=model_n_embd()`）对「openkoai 模式」跳过 head 校验，并让 `probs` 走插件自身打分而非 `head_.probs`；`kev_api.cpp`/`main_server.cpp` 加一个暴露 KEV 决策结果的响应字段（或复用现有 answers 形态）。**主程序主体（编排、批处理、LRU、HTTP 框架）不动。**

> 结论：**B2 是「最小引擎改动、风险可控」的达成方案 B 的推荐路径**；B1 适合作为「验证正确性假设」的探针（用官方 llama-server，§7），或在后续确认 upstream 稳定后作为长期归宿。

| 维度 | 路线 B1（升级 llama.cpp 原生 systemone） | 路线 B2（新 backend 插件） |
|---|---|---|
| vendored llama.cpp 改动 | 必须大同步到 master（当前 0.5.0 无 decision） | **不动**（继续用现有 CPU/Vulkan 插件的前向） |
| kevcpp 主程序改动 | 独立 server 形态≈不动；内嵌形态较大 | 极小（跳过 head 校验 + 一个响应字段） |
| KEV 打分实现 | llama.cpp 官方 | 插件内自实现（简单：切半+点积+softmax） |
| 决策配置消费 | GGUF `<arch>.decision.*` 元数据 | 独立配置文件（§3） |
| 正确性背书 | 官方 | 需自己验证（§7） |
| 维护成本 | 高（追踪 upstream API 变更） | 低（隔离在插件内） |
| 适用 | 探针验证 / 长期完整方案 | **推荐**，最小改动达成方案 B |

---

## 3. 轻量决策配置的格式设计

方案 B 的「轻量决策配置」用 YAML（亦可 JSON），作用是：**在下载的开源 GGUF 身上复刻 kev 型决策所需的全部 prompt/打分参数**。它在 B2 由 openkoai 插件加载；在 B1 则把它翻译成 GGUF 的 `<arch>.decision.*` 元数据（或直接引导一个带该元数据的 GGUF）。

### 3.1 Schema（YAML 草案）

```yaml
# kev decision config —— 方案 B 轻量决策配置
decision:
  type: kev                    # kev|openjev|lev|nimbble|pplx_decider|layA...（B1 需与 llama.cpp 枚举一致；B2 目前只实现 kev）
  shared_mode: true            # B1: can_share_prompt；B2: question 间共享前缀缓存

  # ---- KEV 指针头打分参数（对应 llama.cpp server_decision_context 的 KEV 分支）----
  pointer:
    column_ratio: 0.5          # embedding 对半切开位置：前 0.5 当 q，后 0.5 当 k（LLM_KV decision pointer 语义，待验证）
    n_pointer: 256             # 对半后单侧维度 = floor(model_n_embd * column_ratio)，用于 score = dot(q,k)/sqrt(n)
    question_token_selector: pointer   # 取最后一个 token 的 hidden；decision.pointer 位置（待验证：master 用 pointer 字段定位）
    option_token_selector: markers     # 每个 option 取其 end marker 位置的 hidden；decision.markers

  temperatures:                # map：键为 "<type>" 或 "<type>.<n_options bucket>"
    kev: 1.0
    # kev.2: 1.1              # 示例：2 选项时的覆盖
  noul_true_first: false
  choice_sorted: false

  # ---- marker token 定义（模型词表里的特殊 token，B2 用 backend 的 special_token_id 查询）----
  token_marker: "<|opt0|>"     # 选项起始标记（待验证：KEV 分支是否真的用它定位 option 起点）
  token_sep:    "<|sep|>"      # 问题/选项分隔（待验证）
  text_marker:  "<|q|>"        # 文本标记（text 或特殊 token，待验证确切割分语义）
  max_head_tokens: 0           # question + options 上限（0=不限）
  max_option_tokens: 48

  # ---- 候选类别 → 选项文本 token 映射（本地语义层，非 GGUF 原生，B2 用）----
  categories:
    - key: "a"                 # 该选项在请求/返回里的键（answers 用它）
      label: "Yes"
      prompt_text: "Yes"       # 拼进 prompt 显示给模型的选项文本
      marker: "<|opt0|>"       # 该项的 option end marker token
    - key: "b"
      label: "No"
      prompt_text: "No"
      marker: "<|opt1|>"
    - key: "conf_total"        # NOUL 或评分类选项
      label: "Confidence"
      prompt_text: "confidence_high"
      marker: "<|opt2|>"

  # ---- prompt 模板变量（B2 组 prompt 用；B1 由 server_decision_context::render + chat template 处理）----
  prompt_template: |-
    {system}
    Question: {question}
    Options:
    {for option in options}
    {option.key}: {option.prompt_text}{sep}{option.marker}
    {end}
    Decide: {question.marker}

  # 模板占位符（可选，用于自定义渲染）：
  #   {question} 单题文本，{options} 候选列表，{sep}/{marker} 分隔标记，
  #   {labels} 候选 key 序列（choice_sorted=true 时按此排）
```

### 3.2 示例（真实可跑的最小 `decision.yaml`）

```yaml
decision:
  type: kev
  shared_mode: true
  pointer: { column_ratio: 0.5 }
  temperatures: { kev: 1.0 }
  noul_true_first: false
  choice_sorted: false
  token_marker: "<|opt|>"
  categories:
    - { key: "yes", label: "Yes",  prompt_text: "Yes",  marker: "<|opt|>" }
    - { key: "no",  label: "No",   prompt_text: "No",   marker: "<|opt|>" }
```

### 3.3 配置如何被两条路线消费

- **B1 消费**：`server_decision_context::init(model)` 只认 GGUF 的 `<arch>.decision.*` 键。因此 B1 需要一个「配置 → GGUF 元数据」的翻译器：要么先生成/改装一个含这些元数据的 GGUF 再 `-hf` 下载/加载，要么在本地对下载的 GGUF 做 KV 注入（写 GGUF 需谨慎，属实施阶段）。`labels`/`token_marker`/`temperatures` 直接对应 server-decision.h 字段。
- **B2 消费**：openkoai 插件直接读这份 YAML。`type/prompt_template/categories/marker` 决定 prompt 组装与每个 option 的 hidden 读取位置，`pointer.column_ratio/n_pointer/temperatures` 决定点积分与 softmax，`categories` 决定返回答案的 key/label。

> 统一原则：**配置是 kevcpp 侧的唯一事实源**。B1 是「把配置送给 GGUF 元数据」，B2 是「配置直接进插件」。两者共用同一份 schema，语义即 3.1。

---

## 4. 数据流

### 4.1 路线 B1（独立 llama-server 形态 / 内嵌形态）

```
HTTP 请求 (POST /v1/systemone)
   │
   ▼
[kevcpp main_server/kev_api]  转发 /v1/systemone
   │  (独立 server 形态：仅做 HTTP client 转发；内嵌形态：直调 server_decision_context)
   ▼
[llama.cpp server]
   1 config = server_decision_context.init(model)   ← 从 GGUF <arch>.decision.* 读
   2 parse_questions(body)             ← CHOICE/SCORE/NOUL 题 + options
   3 fill_task(...) / fill_task_joint  ← 按 shared_mode 组 prompt（拼 question + options + markers）
   4 一次 forward（无 token 生成），读取 hidden（question=pointer 位置，option=markers 位置）
   5 打分：labels 空 → KEV 指针头
        embedding 对半 → q(前), k(后) each option；score_i = dot(q,k_i)/sqrt(n_pointer)
        （labels 非空 → 候选 token logits 打分）
   6 format_answer → {choice, score(s), noul}
   │
   ▼
[kevcpp] 归一化为 {model, answers, usage, latency_ms} 响应
```

### 4.2 路线 B2（openkoai 插件）

```
HTTP 请求 (POST .../decision)
   │
   ▼
[kevcpp main_server/kev_api] 解析请求（题 + 候选 categories）
   │
   ▼
[KevModel::probs]  (主程序几乎不动；openkoai 模式跳过 head 校验)
   │  经 BackendFactory_load("openkoai") 得到 openkoai 后端实例
   ▼
[openkoai 插件]
   1 加载 decision.yaml（type=kev、categories、markers、temperatures、prompt_template）
   2 组 prompt：模板变量填充 + 每题 append token_marker…… + 各 option 的 prompt_text + 各自 end marker
   3 调 llama.cpp 前向（复用 CPU/Vulkan 后端 decode），读每个 option end-marker 位置的 hidden + question 位置 hidden
   4 KEV 点积打分 n_input = model_n_embd
        half = n_input/2
        q  = hidden[question][0:half]
        for each opt: k=hidden[opt][0:half]; z_i = dot(q,k)/sqrt(half)
        probs = softmax(z / temperature)   (noul 等按 noul_true_first)
   5 返回 {key→prob|score} 到上层
   │
   ▼
[kevcpp] 包装 {model, answers, usage, latency_ms} 响应
```

---

## 5. 对 kevcpp 现有代码的具体改动点清单

（本设计仅列清单，不实施。路径相对工作区根 `E:\van\projects\kevcpp`。）

### 5.1 走 B2（推荐）：主程序是否基本不动 —— **是，基本不动**

| 动作 | 文件 | 内容 |
|---|---|---|
| **新增插件** | `src/kev_backend_openkoai.cpp`（新建） | 实现 `kev_backend_ops` 全套，复用现有 llama 前向/hidden 读取；`probs`/打分在插件内做 KEV 点积；加载 §3 决策配置 |
| **新增配置** | `config/decision.yaml`（新建，示例见 §3.2） | 决策 schema 实例 |
| **硬编码解绑①** | `src/kev_model.cpp:118-121` | 当前 `head_.dim_in()!=model_n_embd()` 失败即拒。对 openkoai 后端（或 opt.head_path 为空 + 配置 type=kev）**跳过该校验**，改用配置/插件的 `pointer.column_ratio` 而非固定 1024 |
| **硬编码解绑②** | `src/kev_model.cpp:112` `bopt.lm_head=0` | 已是 0，Kev 决策本就不需 lm_head 输出，**无需改**；确认 openkoai 也走 hidden-only |
| **打分解耦③** | `src/kev_model.cpp` `probs/rows_from_slot` | openkoai 模式下不再调 `head_.probs(hd, optp)`，改为调后端暴露的「KEV 打分」入口（对比 `kev_head.cpp`）；`kev_head.cpp` 自身**不改**（保留给现有 Rank-1 路径） |
| **响应字段** | `src/kev_api.cpp` / `src/main_server.cpp` | 在 `{model, answers, usage, latency_ms}` 外扩展决议结果（或多一把 `POST .../decision` 路由），复用 kev 既有 http 解析/序列化 |
| 后端工厂注册 | `src/kev_backend_factory.cpp` | 让 `BackendFactory_load("openkoai")` 能发现新插件（现有机制应只需按目录扫 dll，**待验证**） |

> B2 的 net：**新增 1-2 个文件 + 改 3-4 行判定/接线**，编排核心（LRU、批处理、PrefixLRU、KV 序列管理、HTTP 框架）零改动。

### 5.2 走 B1：改动取决于形态
- **独立 llama-server 形态**：kevcpp 只加 `/v1/systemone` HTTP client 转发 + 响应归一化（`kev_api.cpp` / `main_server.cpp`）。主程序不动。
- **内嵌形态**：需要把 vendored llama.cpp 同步到 master 并提供 `server_decision_context` 能力 → 触及 `kev_backend_cpu/gpu/vulkan.cpp` 的 llama API 封装（因 master 的 `llama_pos` 多槽位 / MROPE / batch API 变化），改动大。
- 所有 B1 形态仍需配置 → GGUF 元数据注入或直接用带 decision 元数据的 GGUF。

### 5.3 硬编码如何解绑/绕过（汇总）
- `dim==1024`：不来自 kev_model 硬编码而是在 head 校验处。B2 用 `pointer.column_ratio`（相对 model_n_embd）替代，天然适配任意 embedding 维的开源模型；B1 由 GGUF `pointer` 配置决定。
- `lm_head=0`：保留，KEV 用 hidden 不依赖 lm_head。
- `kev_head.cpp` 的 PointerHead（双线性 q/k 线性层 1024→256 + temp）：这是**训练出来的 head**，方案 B 不加载它；KEV 原生指针头是**无学习权重的切半+点积**（上游已确认）。两者都要在文档里区分清楚，避免混淆。

---

## 6. 风险与待验证项

### (a) vendored llama.cpp 是否含 decision 支持 —— ❌ 当前不含（已核实），这是 B1 最大风险
- 核实结论：vendored `third_party/llama.cpp`（0.5.0，commit `068d0902`，分支 `build-portable-backends`）中 grep 不到 `COMMON_DECISION_TYPE` / `/v1/systemone` / `server_decision`，也无 `common/` 目录。**当前版本不含 decision 通道。**
- **风险**：B1 必须同步至含 decision 的 master。同步将引入 master 的 public API 变更（`llama_pos` 多槽位数组、MROPE、`common_batch` `decision_order` 等），可能导致 kevcpp 的 CPU/Vulkan 后端封装（`kev_backend_cpu/gpu/vulkan.cpp`）编译/行为变化，且 kevcpp 是 vendored + 本地分支改动，合并冲突面大。
- **缓解**：优先 B2（不动 vendored）；B1 仅作为独立进程探针（不改 vendored，直接下载官方 master llama-server 二进制做验证）。

### (b) KEV 对「普通开源模型」是否真能出有判别力的点积分数 —— ⭐ 最需要验证的正确性假设
- 本质：KEV 打分 = 把「最后 token 的 hidden」切半当 query，把「每个 option end-marker 的 hidden」切半当 key，做 `dot/√n`。这**隐式假定模型在 marker 位置的表示与在 question 位置的表示构成可比的 q/k 点积空间**。
- 对 kev 训练过的模型，这个几何性质是被训练强化的；对**从未见过 KEV prompt 形状的通用开源模型**，`question hidden` 与 `option end-marker hidden` 的余弦关系是否与「答案正确性」相关，是**未证实的假设**。可能的结果：分数有判别力（基线可用）、无判别力（接近均匀）、甚至反向。
- **结论**：方案 B 的成败完全押在此假设上。§7 的验证方案就是为此设计最小成本探针。**在本假设被验证前，方案 B 不应进入实现**。

### (c) prompt / 选项措辞对准确率的影响 —— 高影响，需实验
- KEV 打分完全依赖「正确选项的 hidden 与 question hidden 在点积空间更接近」。选项文本用「Yes/No」还是完整句子、marker token 位置的表示、问题的语义密度都会改变对齐程度。`max_option_tokens`、`prompt_template`、`text_marker` 语义都影响结果。
- **待验证**：多组措辞的消融对比（§7）。

### (d) 是否需要 llama.cpp 输入的「特殊 marker token body 控制」
- master 的 `fill_task_laya` / KEV 分支会往 token 流里插 `token_marker` 等（对 LAYA/KEV 用，见 server-decision.h 注释）。**待验证**：这些 marker 是必须作为原词表特殊 token 注入（`parse_special=true` 走 `special_token_id`），还是可以退化为普通文本 token；普通开源模型若词表没有该类 marker，A/B 路线如何处理（B2 可自行注入控制 token；B1 需要 GGUF 元数据 + 词表支持，否则受限）。

### (e) 准确率需实测区间
- 无法静态给定。**待验证**区间估算：若 (b) 假设成立，KEV 决策在通用模型上的准确率远低于 kev 专用微调模型（后者经专门训练可到高准确率）。合理预期仅为「baseline 探针」，上限待实测。诚实标注：**不承诺任何具体准确率数字**。

---

## 7. 验证方案（最小成本，先验证可行性再动 kevcpp）

**核心思想**：**不动 kevcpp 任何代码、不改 vendored llama.cpp**，先用官方 llama.cpp `llama-server` 作独立探针验证 (b) 正确性假设。

1. **下载官方 llama-server**（含 decision 支持的 master 二进制或自编译），不要碰 vendored。
2. **选模型**：下载 1-2 个通用开源 GGUF（如近 1-2B 的指令模型，CPU 即可跑），也可顺带测一个官方 kev 系示例 GGUF 作对照（区分「方法对但模型不熟」vs「方法本身不行」）。
3. **注入/使用 decision 元数据**：用 llama.cpp 自带工具或脚本把 §3 配置写成 GGUF 的 `<arch>.decision.type=kev` 及 `token_marker/temperatures/labels…`（或用已带元数据的官方示例 GGUF）。
4. **调用**：`POST http://127.0.0.1:8080/v1/systemone`，body 含 `questions:[{id,type:"choice",options:[{...key,description}]}]`。
5. **判读**：看返回的 `score`/`choice` 对「已知标签的小样本集（如 20-50 道 Yes/No / 多项选择）」是否显著偏离均匀、方向是否符合答案。指标：top-1 命中、类内 score 间隔、多次变体稳定性。
6. **对照实验 (c)**: 换几套 prompt/选项措辞，观察分数间隔与命中变化。
7. **裁定**：
   - 若分数有判别力且命中率可接受 → 进入实施，选 **B2**（kevcpp 改动最小）接入。
   - 若无判别力 → 方案 B 的 (b) 假设不成立，**放弃 / 需微调**，此时只有现有 kevcpp 微调路径可用。

> 该验证完全在 kevcpp 之外进行，零风险、最小成本，即可决定方案 B 是否继续。

---

## 8. 待验证项汇总表

| # | 待验证项 | 状态 | 验证方式 |
|---|---|---|---|
| V1 | vendored llama.cpp 是否含 decision 通道 | ❌ **不含**（0.5.0，已核实） | 已核实，代入设计 |
| V2 | master llama.cpp 含 `/v1/systemone` + `COMMON_DECISION_TYPE_KEV` | ✅ **含**（已核实源码） | 已核实 |
| V3 | (b) KEV 对未训练开源模型给出有判别力点积分数 | ⭐ **未验证（最关键）** | §7 独立探针 |
| V4 | `<arch>.decision.*` 全部合法键/取值语义（text_marker、n_options_bucket、column/pointer 精确含义） | 未 100% 穷举 | 直读 master 源码/官方文档（实施期补） |
| V5 | prompt/选项措辞对准确率影响 | 未验证 | §7-6 消融 |
| V6 | 是否需要词表特殊 marker token body 控制，普通开源模型如何处理 | 未验证 | §7-4/5 看是否用普通文本可行 |
| V7 | 准确率实测区间 | 未验证（不承诺数字） | §7-5 小样本实测 |
| V8 | 配置文件 → GGUF decision 元数据注入的可靠工具链 | 未验证 | 实施期选用 llama.cpp 写 GGUF KV 能力 |

---

*本设计仅调研与设计，未修改任何 kevcpp 源码，未运行任何会改写文件的命令。所有标注「✅已核实」的为子代理直连上游源码核实；其余均标注「待验证」。*