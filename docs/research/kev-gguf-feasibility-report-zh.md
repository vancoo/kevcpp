# kevcpp / kev：免 LoRA、免 head 直接加载开源 GGUF 跑 System One 决策推理 —— 技术可行性调研报告

> **调研状态与重要声明（必须先读）**
> 本报告是**纯技术可行性调研**的目标产物。但受限于本次子代理会话的执行环境，**`web_search` 与 `web_fetch` 两个外部网络工具均因 provider 配置问题（错误：`configured web provider "..." is not registered`）完全不可用**，无法进行实时抓取与逐条链接验证。
>
> 因此本报告正文基于已确证的、长期稳定的领域技术知识整理，**每条结论都附上了我确知应当存在的权威来源链接（GGUF 规范、llama.cpp 文档、HuggingFace 教程等）**，但**这些链接均标注为"候选路径（待验证）"而非"本会话已实时抓取验证"**。请把本报告视为**含链接待验证清单的可行性初稿**，并在主会话具备 web 权限的环境中对每个链接做一次打开确认，再决定是否对外引用。
>
> 我不会为凑链路而编造内容；凡牵涉到精确版本号、具体 API 签名、社区里最新的提交行为等"时效敏感"事实，均建议二次确认。

---

## 结论速览（TL;DR）

| # | 技术点 | 结论 | 成本 | 建议方案 |
|---|--------|------|------|----------|
| 1 | GGUF 是什么 | **可行（事实成立）**：GGUF 是 ggml 生态的模型容器 = 权重重张量 + metadata，本身就是"开箱可推理"的格式 | 无 | 直接用现成 GGUF |
| 2 | llama.cpp / ggml 能否直接推理任意 GGUF | **可行**：llama.cpp 加载任意 GGUF 原生在 CPU/GPU 前向推理，天然不要求 LoRA/head；加载后就是标准 logits 分布 | 无 | `llama-cli` / `llama-server` / shared library |
| 3 | GGUF 能否"改权重加/换 head 张量" | **部分可行**：GGUF 结构允许加/改名张量，但 llama.cpp 只执行它"认识的架构"，无法凭空新增不认识的 head 层 | 中～高 | 优先级低，见方案二/三 |
| 4 | 免微调 + 免 head 直接跑 system-one | **可行（在正确实现下）**：system-one 决策可落到"在既有权重上取 logits"，即可在不改权重、不加 head 的前提下做 | 低 | 方案一（logits 直取分类）+ 方案三（配置映射） |
| 5 | 若必须 head 的轻量方案 | 导出脚本/配置文件/dll 各有取舍，见方案二/三/四 | 中 | 优先配置文件，其次导出脚本，最后才 dll |

**一句话核心判断**：只要 kev 的"system one 决策推理"在算法上可以被冻结（freeze）基础权重 + 读取输出 logits 来表达（即 zero-shot / zero-shot 分类 / 结构化标量决策），那么**不需要 LoRA、也不需要单独 head 文件**，直接下载开源 GGUF 即可推理，成本几乎为零（方案一）。只有当 kev 内部明确要求"新增一个专用决策头/专用输出 token 层且必须在权重文件里"时，才需要额外方案。

---

## 1. GGUF 是什么？不依赖微调能否直接前向推理？

### 1.1 事实
- GGUF（`ggml` model format）是 llama.cpp 生态自 ggml 的 `ggml_model_loader`（继承自 GPT-GGML/GGML 早期格式）演化而来的**单一文件模型格式**。
- 它的内容分两大部分：
  1. **metadata（元数据）**：键值对（kv store），记录 `general.architecture`、层数、`block_count`、`head_count`、`rope` 参数、`tokenizer` 词典、上下文大小等。
  2. **张量数据（weight tensors）**：按 `layer.*.weight` 命名的权重张量，以 ggml 支持的各种量化格式（`ggml_type`，如 `Q4_0`、`Q8_0`、`BF16`、`F16` 等）存储。
- GGUF **本身就是为"直接加载推理"而设计的**：`general.architecture` 告诉 loader 用哪套图构建逻辑，张量与临时（工作）张量一起在 ggml 计算图上前向。**GGUF 不内嵌也不要求 LoRA 权重**——LoRA 是运行时附加的（`-l primer` 选项），它与"基础 GGUF 本身可推理"是两件正交的事。
- **结论**：拿到一个开源 GGUF（如 Llama、Qwen、Mistral、Gemma 的无微调 base 或已发布 GGUF），只要 llama.cpp/ggml 认识它的 `architecture`，就能直接前向推理，**不需要、也不涉及 LoRA**。LoRA 只是可选的增量微调手段。
- 依据链接（候选，待验证）：
  - ggml GGUF 规范：`https://github.com/ggml-org/ggml/blob/master/docs/gguf.md`
  - llama.cpp 官方 README（`-l/--lora` 选项证明 LoRA 是可选运行时项）：`https://github.com/ggml-org/llama.cpp`
  - 第三方格式说明（GGUF 结构）：`https://huggingface.co/docs/hub/en/gguf`

### 1.2 判定与依据
- **可行。** 一个 GGUF = 权重 + metadata，本就不含也不要求微调/LoRA/head，通用推理引擎按其 architecture 直跑前向。

---

## 2. llama.cpp / ggml：加载任意 GGUF 直接推理，是否天然不要求 LoRA/head？

### 2.1 事实
- llama.cpp 的核心目标是"在消费级硬件上以最小依赖跑 LLM 推理"。它通过 `llama_model`（`llama_model_load_from_file`）加载 GGUF，然后在 ggml 计算图上执行前向，产出**标准 logits 张量**（shape `[n_vocab]` 或 `[n_tokens, n_vocab]`），再经采样/解码出 token。
- 推理**走的是官方同款权重的原始前向**。LoRA 不是必须的：只有当你需要"改变模型行为/给它注入新能力"时才叠加 LoRA。这里是"决策推理"需求，要看 kev 希望模型"生成什么"。若决策可表达为**在原始 logits 上选 top-k / 在某类 token 上取 softmax**，则零附加。
- GGUF 里没有"自定义逻辑头（专用于决策/分类的输出 token 层）"这样的内置概念。它没有开放的"在权重里新增一个决策头算子"的机制。head 若需要，是由**调用方**（宿主程序）在 logits 上自己加，或由其他框架实现。
- **结论**：
  - 加载任意 GGUF 并 CPU/GPU 推理：**完全可行**，天然不要求 LoRA。
  - 要求"GGUF 支持在权重文件里内置自定义决策/分类 head"：**不可行/不支持**（原生 llama.cpp 只跑它架构里已知的层）。
- 依据链接（候选，待验证）：
  - `https://github.com/ggml-org/llama.cpp`
  - `https://github.com/ggml-org/ggml`
  - llama.cpp 构建/使用文档：`https://github.com/ggml-org/llama.cpp/blob/master/README.md`

### 2.2 判定
- 加载推理：**可行**。内置自定义 head：**不可行（原生）**。

---

## 3. 能否"不改权重"在既有基础模型上叠加轻量任务头做决策？

### 3.1 llama.cpp / gguf 侧
- GGUF 张量可被 gguf 工具/脚本重打包（改写 kv、改名/追加张量均可做到，因为本质是序列化容器）。但 **llama.cpp 的 loader 只初始化它"认识的 architecture"里列出的张量名**。你往文件里塞一个陌生名字张量，llama.cpp 不会自动把它当"决策头"跑——它没有在计算图里接一层你自定义层的逻辑。
- 因此"**通过 gguf 编辑器在权重文件里附加/替换 head 张量并让 llama.cpp 直接执行它**"= **不可行**（除非做二次开发改 llama.cpp 的图构建）。这属于源码级扩展，而非配置文件级。
- llama.cpp 的扩展机制是做"自定义 backend 层 / ggml op"并在图构建代码中注册——那已经等同于方案四（编译 dll/插件），且要动其核心 SchedGraph，成本高、维护难。

### 3.2 其它推理后端：可加轻量分类头且不改基础权重
这一点是支持"方案一/方案三"的关键。
- **HuggingFace `transformers`**：加载基础模型后用 `model.freeze()`/工作量在训练时；对**纯推理**，直接取 `model(...).logits`、或用一个 `nn.Linear(hidden, n_classes)` 挂载在 `hidden_states` 上做分类头——**基础权重不动**，head 就是一个未训练/轻量加载的线性层。很多分类任务（zero-shot 用 `AutoModelForSequenceClassification` 搭配基础 transformer）就是这么做的。
- **ONNX Runtime / OpenVINO**：把冻结好的"base + 分类头"导出成单个 ONNX/IR，推理时输入文本输出类别 logits。head 可以是任意 ONE 前的算子图；基础权重冻结不动。**可行**。
- **TensorRT-LLM**：同样支持加载基础权重并构建含额外输出头的图进行推理。
- **共同点**：这些框架的"head"是**图的一部分/调用方代码**，不是放在 GGUF weight 里的东西，也**不要求对 base 做 LoRA 微调**；head 完全可以是"只读已冻结权重 + 一个轻量映射"。若 kev 只要求"输出 logits 映射到决策类别"，甚至不需要新增任何可训练参数。
- 依据链接（候选，待验证）：
  - transformers AutoModelForSequenceClassification：`https://huggingface.co/docs/transformers/model_doc/auto`
  - ONNX Runtime：`https://onnxruntime.ai/`
  - OpenVINO：`https://docs.openvino.ai/`
  - TensorRT-LLM：`https://github.com/NVIDIA/TensorRT-LLM`

### 3.3 只靠"配置文件"加载开源 GGUF 并映射输出为决策类别（不加 LoRA、不加单独 head 文件）
- **可行（只要 head 是纯 logits 后处理）**。做法：配置文件声明
  - model_path（某个开源 GGUF）
  - 决策逻辑（如：取 `logits[last_token]` 中指定"决策类别 token id"集合的 softmax 作为各类别概率 / 取 top-k 作为决策）
  - 类别 ↔ token 的映射表（如 `类A → token id [..]`）
  宿主程序据此加载 GGUF、跑前向、在 logits 上做映射。整个过程**无 LoRA、无额外权重 head 文件**。
- 前提：kev 的 system-one 语义必须能落到"对 base 模型 logits 做确定性/统计决策"（zero-shot / in-context 决策 / 指定类别 token 的 logits 直取）。若 kev 的 head 是**可训练的**（需要学一个新决策平面），则"纯配置文件"不够，需要至少一个轻量 head（方案二）或在 base 上做 prompt 工程（方案一 shadow）。

### 3.4 判定汇总
| 子项 | 可行性 | 成本 |
|------|--------|------|
| llama.cpp GGUF 内置/替换 head 张量 | 不可行（原生） | 高（要改源码） |
| transformers/ORT/OV/TRT-LLM 冻结 base + 轻量分类头 | **可行** | 中 |
| 纯配置文件（不加 LoRA/不加 head 文件）加载 GGUF + logits→决策映射 | **可行（在 logits 化语义下）** | 低 |

---

## 4. 开源生态"免微调 / zero-shot / 结构化输出 / logits 直取分类"

这是支撑"不需要 LoRA、不需要 head"的生态证据：
- **zero-shot 使用 base 模型**：GLUE/BoolQ/ARC 等推理任务在 controller 时代普遍用**不微调的 base 模型** + 条件 prompt + 在 logits 上取指定类别 token 概率（例如 Winogrande、BoolQ 的"是/否"分类）。这就是"logits 直取分类"的标准范式。
- **结构化/受控解码**：GGML 生态（`grammars`、`json_schema`、`constrained decoding`）可在**不改权重、不加 head**下限制输出为 JSON/类别 token——用于把 LLM 输出钉到决策类别集合上。
- **取出最后一层 logits / hidden states**：`llama.cpp` 及各种绑定（`llama-cpp-python`、`node-llama-cpp`）都暴露 `logits`；也可取最后一 token 的 embedded / logits 作为"隐式 head 输入"。这正是"用指定 logits 位置（最后一个 token）模拟 head 输出"。
- **结论**：生态存在成熟做法，等于"用基础 GGUF 的 logits + 配置映射"来模拟 head，**不需要训练 head，也不留存 LoRA**。
- 依据链接（候选，待验证）：
  - llama-cpp-python（暴露 logits）：`https://github.com/abetlen/llama-cpp-python`
  - node-llama-cpp（logits）：`https://github.com/withcatai/node-llama-cpp`
  - GGML 受控解码 / grammar：`https://github.com/ggml-org/llama.cpp/blob/master/grammars/`

---

## 5. 确需 head 时的几种最轻量方案与成本评估

以下按成本从低到高排列：

**方案一（最理想）：logits 后处理，免 head、免 LoRA**
- 做法：宿主读 GGUF 前向 → 取 `logits[最后 token]` 或指定虚拟 token 位置 → 在类别 token id 集合上 softmax/top-k → 决策。
- 成本：**极低**。仅需一段后处理代码 + 一份类别↔token 映射配置。
- 建议：若 kev 的 system-one 允许这种语义，**直接采用此方案，彻底不碰 LoRA 与 head**。

**方案二：导出脚本，从基础模型自动生成 head 文件（轻量冻结 head）**
- 做法：一个 Python/ONNX 脚本 `freeze_and_attach_head.py`：载入 base GGUF 的权重（转为 safetensors/onnx 或保留原权重），`freeze` 之，在其 `hidden_states` 后挂一个 `nn.Linear(hidden, n_classes)`（随机或从某预置投影初始化），导出为一个推理图/权重包（非完整 LoRA，只是一个 head 权重或一个独立 ONNX）。
- 优点：不改 base 权重；缺点：需要一个推理后端（ORT/OV/TRT 或 llama.cpp+自定义 op）能跑"base+head"张量；`nn.Linear` 若未经训练，等价随机基线，需与方案三结合（在会话里给它 few-shot/embedding 初值）。
- 成本：**中**（写脚本 + 选后端）。

**方案三：手写配置文件（最务实）**
- 做法：YAML/JSON 描述 model_path、决策 token 映射、logits 取位、温度/采样、受约束解码 grammar。宿主引擎按配置加载 GGUF 并映射。**无 head 文件、无 LoRA**。
- 成本：**低**。若日志证明确实需要"一个新输出层"，可把方案三作为"逻辑 head"（纯在 logits 上做的决策映射），而把方案二作为"实在要可学习 head"时才上。

**方案四：编译 dll / 插件做二次前向（成本最高，尽量不用）**
- 做法：针对 transformer 权重 + ggml，写一个 `decide_head` 的自定义 ggml op / 自定义 backend，或用 llama.cpp 的 C API 在 C++ 侧直接访问 `llama_model` 的最后一个 hidden state 并跑自定义 `head`（一个独立小网络），编译为 dll 供 kevcpp 加载。
- 优点：可完全自控 head 逻辑并紧耦合 C++ 宿主；缺点：**要维护框架源码级别的扩展/二次前向**，跨平台 dll ABI、模型/版本兼容都要自己管，成本**高**。
- 建议：仅在"必须新增可学习的决策头且必须嵌入本地推理内核"时才选；否则优先方案一/三。

| 方案 | 是否免 LoRA | 是否需要单独 head 文件 | 成本 | 适用 |
|------|------------|----------------------|------|------|
| 一：logits 后处理 | 是 | 否 | 极低 | kev 允许 logits 语义 |
| 三：手写配置 + 映射 | 是 | 否 | 低 | 最通用、推荐作默认 |
| 二：导出脚本生成冻结 head | 是 | 是（轻量 head/ONNX） | 中 | 需要可学习/可部署 head |
| 四：编译 dll 插件二次前向 | 是 | 是（dll） | 高 | 必须自控 head 且本地内嵌 |

---

## 总体建议

1. **首选方案一 + 方案三**：只要 kev 的 system-one 决策能表达为"冻结基础权重上取 logits + 类别映射/受控解码"，就**直接下载开源 GGUF、不微调、不提供 head、只给一份类别映射配置文件**即可跑通。这满足用户"免 LoRA、免 head、只要配置文件"的全部诉求。
2. 若 kev 内部硬编码要求一个"可学习输出 head"，再上**方案二（导出冻结 head）**；不轻易选**方案四（dll）**。
3. **上线前的必要动作**：在主会话具备 web 权限的环境里，对下方"候选链接"逐条打开确认后再引用；若 kev 的工程文档里写明了它与 base 模型的 head/LoRA 契约，请先读它的 `docs/` 与推理入口源码，把 kev 的"head 语义"与上面对齐，避免我这里的"可行性"与 kev 实际设计脱节。

---

## 关键外部链接（候选，待验证 — 本会话因 web 工具不可用未能实时抓取）

> 以下是我确知应当存在、与该结论高度相关的权威来源；请在可用网络环境逐一打开确认后引用。

- GGUF 规范（ggml）：`https://github.com/ggml-org/ggml/blob/master/docs/gguf.md`
- llama.cpp（GGUF 加载 + LoRA 可选运行时 `-l`）：`https://github.com/ggml-org/llama.cpp`
- ggml（sampling / 计算图前向）：`https://github.com/ggml-org/ggml`
- HF GGUF 说明：`https://huggingface.co/docs/hub/en/gguf`
- llama-cpp-python（logits / python 绑定）：`https://github.com/abetlen/llama-cpp-python`
- node-llama-cpp（logits）：`https://github.com/withcatai/node-llama-cpp`
- transformers AutoModel/SequenceClassification（冻结 base + 分类头）：`https://huggingface.co/docs/transformers/index`
- ONNX Runtime：`https://onnxruntime.ai/`
- OpenVINO：`https://docs.openvino.ai/`
- TensorRT-LLM：`https://github.com/NVIDIA/TensorRT-LLM`
- 受控/结构化解码 grammar 示例：`https://github.com/ggml-org/llama.cpp/tree/master/grammars`