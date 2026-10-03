# Kev —— 纯 C++（CPU）重写实施计划

> 目标：用纯 C++ 复刻 Kev 决策模型，**CPU 版**，从「最小可验证版本」逐步推进到「完整 /v1/systemone 服务」。
> 关键洞察：Kev 是**纯 prefill 模型**（从不生成 token），只在 `<decide>` 和每个 `</opt>` 位置读取 hidden state 并过一个小 `PointerHead`。
> 基座（Qwen3.5-0.8B 混合架构 Gated DeltaNet + attention）与 LoRA 全部由 **llama.cpp** 承载；我们自己写读出头 + 编码层 + HTTP 壳。

---

## 0. 环境事实（已实测确认）

| 项 | 值 |
|---|---|
| 基座（HF） | `D:\share-models\qwen3.5-0.8b`（Qwen3.5-0.8B，混合架构，hidden_size=1024，24 层，full_attention_interval=4，head_dim=256） |
| GGUF 基座 | `D:\share-models\Qwen3.5-0.8B-Q8_0.gguf`（774 MB，llama.cpp 原生加载） |
| llama.cpp 源码 | `E:\van\projects\llama.cpp`（已实现 `LLM_ARCH_QWEN35`、Gated DeltaNet、`llama_get_embeddings_ith`、LoRA） |
| Kev Python 参考 | `E:\van\projects\kev`（`model.py` / `checkpoint.py` / `predictors.py` / `mlx_model.py` / `serve.py`） |
| C++ 工作区 | `E:\van\projects\kevcpp`（本目录） |
| 编译器 | **MSVC 2022 Community**，cl.exe = `d:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe`（dev 环境：`vcvars64.bat`） |
| CMake | 4.4.0（`CMake 4` 支持） |
| 生成器 | 用 MSVC 的「Visual Studio 17 2022」生成器（CMake 自动）；或加装 ninja 用 Ninja 生成器 |
| 辅助 | git、python 3.11、micromamba（`proxy` 环境） |

> 注意：g++/cl/ninja **不在 PATH**。构建 llama.cpp 时必须先 `call vcvars64.bat`（或 CMake 里指定 VS 生成器），cl.exe 才会可用。

---

## 1. 架构总览（从 Python 到 C++ 的映射）

```
Kev Request (JSON: state + questions)
   │  layer2 encode()  【C++ 复刻】 特殊 token + <|x|>→<¦x¦> + row form
   ▼
Token seq (ids, pos, seg, opt, decide_idx, opt_idx)
   │  layer0 llama.cpp  【0 代码】 prefill 读 hidden state
   ▼
hidden states @ <decide> 和 @ 每个 </opt>
   │  layer1 PointerHead 【C++ 移植 ~80 行】 q/k 线性层 + 点积 + softmax / T
   ▼
per-question probabilities  →  输出 JSON（noul/choice/score）
```

- **row form**（kev 对混合架构的原生路径，`model.py:rows_of`）：state 算一次，每个 question 作为独立因果行继续 → 这**正好等于** llama.cpp 的原生 prefill + KV/recurrent cache 复用，**无需自定义注意力掩码**。
- 前缀缓存 = llama.cpp 的 KV cache + Gated DeltaNet 反复式状态（`llama-memory-recurrent.cpp`）自然延续，我们只需做 LRU 索引。

---

## 2. C++ 工程结构

```
E:\van\projects\kevcpp\
├─ IMPLEMENTATION_PLAN.md      # 本文件
├─ CMakeLists.txt
├─ third_party/                # llama.cpp 作为 git submodule（或直接引用源码目录）
│   └─ llama.cpp
├─ src/
│   ├─ main_cli.cpp            # M1-M3 的命令行 scorer（对拍 & 调试）
│   ├─ kev_encode.{h,cpp}      # layer2 encode：user_tokens / rows_of / SPECIAL 常量
│   ├─ kev_head.{h,cpp}        # layer1 PointerHead + softmax / 温度
│   ├─ kev_model.{h,cpp}       # 组装：edd Model = llama_context + head + prefix cache
│   ├─ kev_prefix.{h,cpp}      # layer3 LRU 前缀（KV 索引）缓存
│   ├─ kev_api.{h,cpp}         # layer4 /v1/systemone 请求/响应 schema、noul/choice/score 语义
│   └─ main_server.cpp         # M4 HTTP 服务入口
├─ tests/
│   ├─ par1ty_python.py        # 用 kev Python 版生成 .jsonl 对拍基准（golden）
│   └─ test_head.cpp           # PointerHead 单测（数值 vs numpy/torch 参考）
├─ models/                     # 指向 GGUF / 转换后 LoRA 的软链或说明
└─ scripts/
    ├─ build.ps1               # vcvars64.bat + cmake + 构建
    └─ convert_lora.ps1        # 调用 llama.cpp convert_lora_to_gguf.py
```

---

## 3. 里程碑与验证闸门

### M0 —— 环境与最小加载验证
- [ ] `build.ps1`：`call vcvars64.bat` → `cmake -S . -B build -DLLAMA_CURL=OFF -DGGML_NATIVE=ON` → 构建 llama.cpp 静态库。
- [ ] 最小 C++ 程序加载 `Qwen3.5-0.8B-Q8_0.gguf`，`llama_decode` prefill 一句文本，`llama_get_embeddings_ith()` 取出某 token 的 1024 维向量，打印 L2 范数（非 NaN、非全零）。
- **验收**：程序正常输出模型信息（`llama_model_desc`）、n_embd=1024、hidden state 范数合理。

### M1 —— 最小可验证版本（MVP）★ 最重要的一步
- 目标：**用最少的 C++ 复刻出与 Python kev 概率一致的读出头**，证明整条链路可行。
- 做法：
  1. 用 Python kev 侧（可临时用 `kev.predictors.LocalPredictor` 或手动拼 encode）生成若干**固定请求**的 `ids` / `<decide>` 索引 / `<opt>` 索引，导出为 golden JSON。
  2. C++ `main_cli.cpp`：输入同样的 token 序列 → llama.cpp prefill 取向量 → 移植的 `PointerHead`（先用**随机/占位权重**跑通链路，再加载真实 `head.pt` 权重）。
  3. 数值对拍：`max|Δp|` 和 argmax 一致率。
- **交付物**：`kev_head.{h,cpp}` + 一个把 `head.pt` 权重读入 C++ 的小工具。
- **验收**：相同输入下，C++ 版单问概率与 Python 版在 `1e-2` 量级内一致（Q8_0 扰动容许），argmax 全一致。

### M2 —— 完整 encode 逐 token 对齐
- 复刻 `encode()`（`kev/model.py:82`）：SPECIAL 分隔、`user_tokens` 的 `<|x|>`→`<¦x¦>` 改写、row form `rows_of` 拆分、`decide_idx`/`opt_idx` 计算。
- 分词：直接用 GGUF 自带分词器（`llama_tokenize`），与 HF 快速分词器核对（注意 `split_special_tokens` 差异）。
- **验收**：对同一请求，C++ 生成的分词结果、`decide_idx`/`opt_idx` 与 Python `encode()` **逐 token / 逐索引一致**（`tests/parity_python.py` 断言）。

### M3 —— 前缀缓存 / KV 复用
- 实现 `kev_prefix` LRU：同 state 复用 llama.cpp KV + DeltaNet 状态（等价 kev `prefix()`/`probs_with_prefix()`）。
- 实现 state 一次性编码、多 question 分批出 hidden state。
- **验收**：同一 state 连续多次请求，第二次起耗时显著下降；概率不变。

### M4 —— 完整 `/v1/systemone` HTTP 服务
- 用轻量 C++ HTTP（`cpp-httplib` 或 `drogon`）复刻 `serv.py` 的 TypeSafe 接口。
- 补齐 `noul/choice/score` 三种问题语义、`/v1/models`、`x-typesafe-request-id`、bearer auth、`temperature`。
- **验收**：能通过 `kev` 的 `tests/test_api.py`（TypeSafe SDK 直接打本服务）。

---

## 4. 关键技术风险与对策

### R1：LoRA 目标张量命名差异
kev（PEFT）adapter 的 DeltaNet 目标是 `in_proj_qkv / in_proj_z / in_proj_a / in_proj_b / out_proj`；llama.cpp qwen35 用 `wqkv / wqkv_gate(z) / ssm_beta / ssm_alpha / ssm_out`。
- **对策**：用 `convert_lora_to_gguf.py` 转换时按张量语义映射；逐个核对 A/B 矩阵、`alpha/r`（kev `lora_alpha=2r`，见 `mlx_model.py:37` merge 公式）。M1 阶段可**先不要 LoRA**（仅 base + 未训练/随机 head 验证链路），M2 后再接真实 LoRA，最大程度解耦。

### R2：Q8_0 量化对概率/温度的影响
kev 的温度 T（0.8B=2.35）是在 **bf16/原始 logits** 上标定的。Q8_0 量化会引入扰动。
- **对策**：MVP 阶段实测 `max|Δp|`；若温度敏感，提供 `--temperature` 可覆盖，必要时在 C++ 侧重标定温度（与 `scripts/calibrate_checkpoint.py` 同思路）。

### R3：hidden-state 精度
llama.cpp embedding 输出为 **fp32**（`res->t_embd`），对 PointerHead 有利。Q8_0 的主要扰动在权重，属可接受范围。

### R4：逐 token 对齐（分词差异）
Qwen tokenizer 的 fast/legacy 分词、`<¦x¦>` 改写、`add_special_tokens=False` 都要对齐。
- **对策**：M2 用 golden 测试钉死；必要时用 llama.cpp 分词器 + 显式特殊 token id 映射。

---

## 5. 建议执行顺序（先跑通再优化）

```
M0 (build llama.cpp + load GGUF)
  → M1 MVP (读hidden+PointerHead+对拍)   ← 门槛，证明可行
  → M2 encode 对齐
  → M1 接入真实 head.pt 权重 + convert LoRA
  → M3 前缀缓存
  → M4 HTTP 服务
```

关键原则：**先用 base+随机 head 跑通链路，再接真实权重和 LoRA**，把「架构可行性」与「精度对齐」分开验证，避免一次引入过多变量。

---

## 6. 结论
- **纯 C++ CPU 版完全可行**：llama.cpp 原生支持架构 / hidden state / LoRA；我们只需写读出头 + 编码层 + HTTP。
- **GGUF 基座直接可用**：`Qwen3.5-0.8B-Q8_0.gguf`（774 MB）省内存，适合 CPU。
- **建议从 M1 MVP 起步**，以「与 Python 对拍概率一致」作为每个里程碑的硬验收。