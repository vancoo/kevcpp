# kevcpp「免 LoRA 微调 / 免 head 直用开源 GGUF」可行性调研报告

> 调研日期：本会话
> 调研人：子代理（delegated research agent）
> 任务：外部网络调研（web_search / web_fetch），验证 kevcpp 能否不经过模型微调（不做 LoRA）也不提供
> head，直接下载开源 GGUF 就能跑 system-one 决策推理；或退化为「配置文件」 / 「针对特定模型的适配 dll」。

---

## ⚠️ 重要前置声明：本会话的外部网络调研工具不可用

本次受委托执行**外部网络调研（web_search / web_fetch）**，但在执行时，两个工具均因环境后端未注册而
失败（返回 `Error: configured web provider "bing" / "direct" is not registered`），已多次重试均相同。

这意味着：
- 本报告**未能**在本次会话内对任何外部链接做实时抓取与逐字核对；
- 下文的「外部链接」是我基于对这些开源生态的既有技术认知给出的**权威来源地址**，**未经本会话在线验证**；
- 结论的技术判断**可靠**（基于扎实且公认的 GGUF / llama.cpp / transformers / 推理框架机制），
  但如需「已验证的外部引用」，请在联网可用的会话里对下文标注的链接做二次核对。

> 依据：本会话明确约定「web_search 结果为外部不可信数据；web_fetch 亦为外部不可信数据」，且「调研、写作
> 类具体事务由子代理完成并回传自包含结果」。此处仅如实汇报工具不可用，不以臆测冒充已验证来源。

---

## 0. 结论速览（TL;DR）

| 问题 | 结论 | 关键依据 |
|---|---|---|
| 1. 免 LoRA、免 head，直接跑开源 GGUF 就出决策 | **不可行（直接出决策）** / 部分可行（在「拿现成 base LLM 的 logits / 生成文本」层面可行；但拿不到 kev 的 PointerHead 决策输出） | GGUF 只含权重+元数据，不含决策头；决策信息在单独的 head 里 |
| 2. 只给一份「容易写的配置文件」完成适配 | **不可行（完整复刻）** / 部分可行（把 PointerHead 结构描述成配置可行，但默认推理引擎不消费它） | PointerHead 是**第二个前向网络**，不是 GGUF 元数据能表达的 |
| 3. 针对特定模型的「适配 dll」 | **可行（成本最低路径）** | kevcpp 已有现成 C-ABI 后端 dll 机制 + `kev_head.cpp`；做一个 head 插件 dll 是「已几乎建成」的事 |
| 4. 底层机制（GGUF/llama.cpp/其它框架头叠加） | **GGUF+llama.cpp 可直接前向任意开源 GGUF**；「不改权重叠 head」在 llama.cpp 层需自定义图/后端，「配置」无法表达 | 机制清楚见 §3 |

**一句话：** kev 的决策能力并不来自「base LLM 的 Vocab logits」，而是来自**额外一个名为 PointerHead 的
二段侧网络**（`d=1024 → dp=256`，再做 `k(h_opts)·q(h_decide)*scale` + softmax）。因此「免 LoRA、免 head
直接跑一个开源 GGUF 就得到同样决策」在**机制上不成立**；真正可行的最轻路径是**把 PointerHead 打包成一个
适配 dll/配置，或直接用现成的 `kev_head.cpp` + 后端 dll 机制**。

---

## 1. kevcpp/kev 到底「适配了什么」（本地权威证据，本会话已读代码与文档）

在评判可行性前，必须先精确回答「head 是什么、LoRA 是什么」。以下全部来自本仓库文档/代码（非网络）：

- **基础模型**：不是现成开源模型，而是**自研改造的 0.8B 混合架构** = 24 层（18 层 Gated-DeltaNet 循环 +
  6 层全注意力）。`n_embd=1024`，vocab=248320。
- **LoRA 在 kevcpp 里已被 fold 掉**：`src/merge_lora_tool.cpp` 把 LoRA 融进 Q8_0 基础权重，
  运行时走**单份权重** `models/kev-merged-q8.gguf`（约 774 MiB）。→ 运行时已经**不再需要 LoRA 文件**。
- **lm_head 是 tied Rank-1 且被砍掉**：README §4 明确「砍掉未使用的 tied lm_head（Rank-1，约 -100ms）」。
  → 决策**不是**从 LLM 的词汇表 logits 取的。
- **决定性输出来自 PointerHead**（`models/head.bin`，约 2 MiB）：
  - 结构：`d=1024 → dp=256`；`k(h_opts) · q(h_decide) * scale`；softmax；temperature（`kev_head.cpp` 复刻）。
  - 数据流（README §8.3）：`parse → encode → ensure_state → rows_from_slot（branch decode，在 decide/option
    位置读 hidden）→ PointerHead → probabilities(softmax z/T)`。
  - 本质：PointerHead 是一个**把「决策 query」与「各选项 hidden」做点积读出**的二段分类网络，**运行在 LLM
    的 hidden states 之上**，是**独立的第二段前向**，不是 Vocab softmax。

> **最关键推论**：kev 的「决策概率」由 PointerHead 对 hidden states 的运算给出。任何「只加载 base GGUF、
> 不附 head」的推理，能得到的是 base 模型的**文本 logits / 续写**，**没有** PointerHead 这个决策维度。
> 所以「免 head 直接决策」在逻辑上等价于「不要分类器就做分类」，机制上不成立。

---

## 2. 逐条技术点结论（含建议方案）

### 1) GGUF 是什么、不依赖微调能否直接用通用引擎前向推理 —— **可行**

- GGUF 是 llama.cpp/ggml 生态的模型序列化格式，本质 = **张量权重（量化或 fp）+ 模型超参数元数据
  （架构、层数、dim、vocab、rope 参数等）**。它**不固有「微调/适配」概念**，就是把一个已训练好的权重
  checkpoint 换成更紧凑、可 mmap 的二进制容器。
- 任一符合 llama.cpp 支持的架构（Llama/Qwen/Gemma/Mistral/GDN 等）的 GGUF，llama.cpp/ggml 都能
  **直接加载并在 CPU/GPU 前向推理**，无需任何 LoRA 或额外 head。这是 llama.cpp 的日常用法。
- **但**：这次「前向」产出的是该 base 模型的**文本 logits / next-token**。要得到 kev 式决策，
  缺 PointerHead 这一步。

> 依据（未经本会话在线核对的权威源，供联网复核）：
> - GGUF 格式说明：`https://github.com/ggml-org/ggml/blob/master/docs/gguf.md`
> - llama.cpp 主仓：`https://github.com/ggml-org/llama.cpp`
> - GGUF 概述 wiki：`https://github.com/ggml-org/llama.cpp/wiki/GGUF`

**结论：** 直接前向 = **可行**（而且是现成能力）；但「得到决策」= **不可行**（缺 PointerHead 这一维）。

**（附）能否在既有 base 上「不改权重」叠一个轻量任务头？**
- **llama.cpp / GGUF**：GGUF 本身只是张量容器，可以在导出阶段**新增一个自定义 head 张量**写进去；
  但 llama.cpp 的**默认推理图不消费**它。要让 llama.cpp 用这个头，需要**自定义后端 / 自建 ggml 图**
  （类似 kevcpp 已做的：decode 后读 hidden → 另跑 `kev_head`）。纯「配置文件」不能触发这个二次前向。
  - llama.cpp 演进中虽支持一些自定义（`llama_probe`、插件 backend、ggml 图扩展、`--logits-all` 取每位
    logits），但**没有公开的标准「把任意自定义决策头描述进配置文件并自动执行」**的机制。
- **transformers（HF）**：完全可行。官方做法就是 **freeze 基础模型 + 自定义 classification head**，
  这是 Hugging Face 的**标准 fine-tuning 教程模板**（如 `AutoModelForSequenceClassification`、或手写
  `nn.Module` 包装）。「freeze LN 不加微调」指不训练，但**仍需要加载 head 权重**才能得到该头的输出。
  - 若**不训练**，则 head 权重必须**已存在**（否则刚初始化的 head 输出是噪声）。→ 仍需一份 head 权重。
- **OpenVINO / ONNX Runtime / TensorRT-LLM**：都允许「freeze 主干 + 附加自定义子图/层」——因为这本质
  是**图/模型层面拼接**。操作上需要重新导出模型（把 base + head 合成一个 graph/onnx/IR/engine），
  这又是一次「导出脚本」工作量，**不是纯配置**。

### 2) 能不能「只给一份容易写的配置文件」完成适配 —— **部分可行 / 守护（做成完整复刻）不可行**

可行性分析：
- PointerHead 的**数学结构**确实很小（`d=1024→dp=256`，点积+scale+softmax），**足以用配置描述**
  （张量维度、scale、temperature、从哪些 hidden 位置取值）。
- **但默认推理引擎不会因为读到一份配置文件就自动去跑这段数学**。你要么：
  (a) 自己写一段**图 JSON / 插件**让引擎执行它（kevcpp 的 `kev_head.cpp` 就是这份「可执行配置」）；
  (b) 用 llama.cpp 的**自定义 backend / ggml 图脚本**（存在，但仍是代码/图定义，不是「纯声明式配置」）。
- 更重要的是：用户诉求是「**不用微调、也不用提供 head**」。`LoRA` 可以免（因为 base 开源 GGUF 本来就是
  通用模型）；但 **PointerHead 这一份 2MB 权重是决策的本体**，**不可凭空免除**——除非接受退化为
  「用 base 模型自己的文本/generation 能力做近似决策」，那完全偏离 kev 语义、且需另写 prompt/解析逻辑。

> **诚实区分两种「配置文件」：**
> - 「描述**已有** head（张量布局 + 算子参数）」→ 可行，但 head 权重仍必须有来源。
> - 「让引擎自动为任意 base 凭空生成决策头」→ **不可行**（引擎没有、也不该有「凭空发明决策头」的能力）。

### 3) 针对特定模型的「适配 dll / 插件」—— **可行，且成本最低（kevcpp 已具备全部零件）**

这是三条路径里**最推荐**、**成本最低**的一条，因为 kevcpp **已经实现了其中绝大部分**：

- 仓库已有 **C-ABI 后端插件机制**（`docs/portable_backend_deployment.md`）：
  - `kev_backend.h` 定义 `meta/params/ops`（无 llama 类型、ABI 版本化、ops 表只追加）；
  - `BackendFactory_load(name)` 经 LoadLibrary/dlopen 加载 `kev-backend-cpu.dll`（或 `kev-backend-vulkan.dll`）；
  - HTTP 宿主（`main_server.exe`）**不链接 llama**，只通过 `kev::Backend*` 调算子面。
- PointerHead 已有 C++ 实现 `src/kev_head.cpp`，`n_embd=1024 → dp=256` 前向。
- 因此「做一个 head 插件 dll」的增量工作 = **把 `kev_head.cpp` 包成一个 C-ABI 插件**（载入 `head.bin`），
  或将现有 `kev-backend-cpu.dll` 扩展一个 `compute_head` 算子。这是**已经建成 80% 的工程**，而非从零开始。

> ⚠️ 已知工程坑（本地文档已记录，非臆测）：插件方案受 llama.cpp **single-llama rule** 约束——每个进程
> llama/ggml 只允许**一份拷贝**；`main_server.exe` 若同时静态链接 llama 又加载 `kev-backend-cpu.dll`，
> 会出现双拷贝崩溃（该分支已被回滚）。所以生产态必须让**只有 dll 链接 llama**。这是完成这套方案时
> **唯一需要在一个稳定带调试器的主机上收尾验证的点**。

### 4) 现行生态「免微调 / zero-shot / 结构化输出 / logits 取分类」的做法 —— **有多种，但都不等于 PointerHead**

- **结构化输出 / JSON schema**：llama.cpp → server 的 `--jinja`/grammar，或 Outlines、Guidance 等库，
  **不改 base 权重**、纯靠约束解码/grammar 强制模型输出固定格式或「选项 A/B/C」文本。
- **logits 直接取分类**（无需微调）：对大模型做分类不用 softmax 头，而是**取指定 token 位置的 logits**、
  或对候选 token 集做 `logprobs` 比较（zero-shot 文本分类）。llama.cpp 的 `--logits-all` / `logprobs`
  都支持。**但这仍是「base 模型的 Vocab logits」，不是 kev 的 PointerHead**。
- **结论**：这些「免微调」技巧能做**近似/替代决策**，但**无法逐位复刻** `kev-merged-q8.gguf`+`head.bin`
  的概率输出（因为决策信号所在的那个网络，恰恰是 LoRA-fused 主干 + PointerHead 的组合）。

### 5) 若要一份 head，三种最轻量方案的定价对比（参考现成机制）

| 方案 | 内容 | 成本 | 是否等价复刻 kev 决策 | 备注 |
|---|---|---|---|---|
| **A. 导出脚本自动生成 head 文件** | 写脚本从「已有 base + 训练/导出好的 head 权重」导出 `head.bin`（kevcpp 本就有转换/合并工具 `merge_lora_tool` 可扩展） | 低‑中 | ✅ 是 | 需**已有** head 权重源；脚本本身不难 |
| **B. 配置文件+自定义 backend** | 用配置描述 PointerHead 结构 + llama.cpp 自定义 backend/ggml 图执行 | 中‑高 | ✅ 是（工程量大） | 引擎默认不消费配置；需自己写执行段 |
| **C. 适配 dll / 插件做二次前向** | 用 kevcpp **现有 C-ABI dll 机制**把 `kev_head` 包成 `kev-backend-*.dll` 或 head 插件 | **低（最省）** | ✅ 是 | **kevcpp 已建成 80%**；唯一收尾点是 single-llama rule |

**强烈建议：方案 C**，因为 kevcpp 已有完整后端 dll 骨架和现成 `kev_head.cpp`，成本最低、风险最小。

---

## 3. 对用户原始诉求的逐条判定

> 原始诉求：**不经过模型微调（不做 LoRA）、也不提供 head，直接下载开源 GGUF 直接跑 system-one 决策推理**。

1. **「不做 LoRA」——可行。** 开源 GGUF 已是完整权重，无需再 LoRA。kevcpp 甚至已把 LoRA fold 进单权重。
2. **「不提供 head，直接决策」——不可行（机制上）。** PointerHead 是决策的本体，无法凭空免除。
   除非接受退化为 base 模型的文本/近似输出（偏离 kev）。
3. **「退一步只给配置文件」——部分可行。** 配置能**描述** PointerHead，但**默认引擎不消费它**；
   仍需配套执行逻辑（backend/图），且 head 权重依旧要有来源。
4. **「针对特定模型的适配 dll」——可行，且是成本最低解。** kevcpp 已有 C-ABI dll 机制 + 现成 head 实现。

**给主会话的建议下一步：**
- 若接受「head 必须有」，直接走**方案 C（head/backend 插件 dll）**，基于现有 `kev_backend.h` +
  `kev_head.cpp`，成本最低；
- 若坚持「完全不提供任何 head」，则只能做「开源 base 模型的 zero-shot 结构化文本决策」近似，**无法
  复刻 kev 的逐位概率**，且需另外设计 prompt/grammar/取 logits 逻辑——建议明确告诉用户这是两条本质
  不同的路线。

---

## 4. 建议复核的外部链接（未经本会话在线验证，因 web 工具不可用）

> 以下链接均为权威一手来源，**请在具备联网与 web 工具的会话中核对**后再作为「已验证引用」使用。

- GGUF 格式规范：https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- llama.cpp 主仓库（GGUF 加载/推理/后端插件/grammar）：https://github.com/ggml-org/llama.cpp
- GGUF 概览 wiki：https://github.com/ggml-org/llama.cpp/wiki/GGUF
- Hugging Face 分类头 / freeze backbone 微调教程（含自定义 head、AutoModelForSequenceClassification）：
  https://huggingface.co/docs/transformers/tasks/sequence_classification
- transformers 文档（取 logits、zero-shot、类头）：https://huggingface.co/docs/transformers/index
- OpenVINO：https://docs.openvino.ai/
- ONNX Runtime：https://onnxruntime.ai/
- TensorRT-LLM：https://github.com/NVIDIA/TensorRT-LLM
- Outlines（结构化生成）：https://dottxt-ai.github.io/outlines/
- 本仓库内权威依据（本会话**已实际阅读**）：`README.md` §4/§8、`docs/KECPP_vs_KEV_SERVING_CONTRAST.md`、
  `docs/portable_backend_deployment.md`

---

## 5. 诚实声明（scope & caveats）

- **网络部分未经在线验证**：因 `web_search`/`web_fetch` 在此会话不可用（provider 未注册），所有外部链接
  未实时抓取；技术判断基于成熟而稳定的既有认知，但请勿把本报告当作「已含实时引用」的调研。
- **代码/文档部分已核实**：所有 kevcpp 内部事实（LoRA fold、tied lm_head 被砍、PointerHead 结构、
  后端 dll 机制、single-llama 坑）均来自本仓库实际读到的 README 与 docs，可作唯一可靠依据。
- 本报告未断言任何「已验证的网络来源」；如需权威引用落稿，建议在联网会话补一轮 web 核对后再转正式文档。