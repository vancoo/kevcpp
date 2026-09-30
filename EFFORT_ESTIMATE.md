# Kev(C++) 各阶段工作量评估

> 前置约束（已确认）：**kevcpp 把 llama.cpp 需要用到的代码复制进本项目再修改使用，不直接改动 `E:\van\projects\llama.cpp`。**
> 本文件据此给出两种 vendoring 策略下的工作量。**全线单人，CPU 版，纯 prefill。**

---

## 0. 一句结论

llama.cpp 编译产物就是两个现成静态库 target：`llama`（glob 所有 `src/*.cpp` + `models/*.cpp`）和 `ggml`（`add_subdirectory(ggml)`），kevcpp 只需 `add_subdirectory` 一份复制 + `target_link_libraries(kev PRIVATE llama)`。**因此采用「全量复制」策略时，llama.cpp 部分的"搬运成本"几乎为零**，各里程碑的真正工作量集中在 kev 自身的读出头 / 编码层 / 接线 / HTTP 上。总量约 **6–11 人天**（不含测试打磨）。

---

## 1. 两种 vendoring 策略（工作量大头在选哪条）

| 策略 | 做法 | 搬运成本 | 优点 | 缺点 |
|---|---|---|---|---|
| **A. 全量复制（推荐）** | 把 llama.cpp 整棵 `src/ ggml/ include/ vendor/` 复制进 `third_party/llama.cpp`，保留其根 `CMakeLists.txt`，用 `add_subdirectory` 引入 `llama` / `ggml` 两个静态 target | **≈0.25–0.5 人天**（复制+固定版本+建 CMake 骨架） | 零裁剪回归、能直接拿 `llama_x` C API、后续改 qwen 只在 own 拷贝内改 | 磁盘/体积大（~数百文件）；但 CPU 版无额外依赖 |
| **B. 最小子集裁剪** | 只复制 CPU+gguf+qwen35 实际链接的 ~30–50 个源文件，去掉其它后端/采样器/多模态 | **≈2–4 人天**（死代码裁剪 + 头文件剪枝 + CMake 重写 + 每步回归） | 仓库极简 | 裁剪/回归工程量大，性价比低 |

> **强烈建议策略 A**。理由：(1) llama.cpp 的 `llama` target 是 `file(GLOB models/*.cpp)`，去掉一个带依赖的后端极易编译失败；(2) CPU 版没有 OpenCL/Vulkan/Metal 等后端编译负担；(3) 全量复制才符合"改了 own 拷贝不碰上游"的隔离意图。策略 B 的节省不值得那 2–4 人天。

---

## 2. 里程碑工作量细化（策略 A，单人）

| 阶段 | 内容 | 自研(kev)代码 | 搬运/现成 | 估算 |
|---|---|---|---|---|
| **M0 构建+加载** | `vcvars64.bat`+cmake 建库；复制 llama.cpp；最小程序 load GGUF→prefill→`llama_get_embeddings_ith` 取 1024 向量 | `main_cli` 骨架 ~100 行 | llama/ggml target 现成 | **0.5–1 人天** |
| **M1 MVP 读出头** | 移植 `PointerHead`（q/k 双线+点积+softmax/温度）| ~80–120 行 | `llama_get_embeddings_ith` 现成 | **1.5–2.5 人天** |
| **M2 encode 对齐** | 复刻 `encode()`：SPECIAL 分隔、`<\|x\|>`→`<¦x¦>`、`rows_of`、decide/opt 索引；分词对拍 fast tokenizer | `kev_encode` ~250 行 + 对拍脚本 | GGUF 自带 tokenizer | **1–2 人天** |
| **M3 前缀缓存** | LRU 索引复用 llama.cpp KV+DeltaNet 状态（`llama_get_kv_cache` + 反复式状态）| `kev_prefix` ~200 行 | KV 复用原语现成 | **1–2 人天** |
| **M4 HTTP** | `cpp-httplib`/`drogon` 复刻 `/v1/systemone`、`/v1/models`、auth、`x-typesafe-request-id` | `kev_api`+`main_server` ~400 行 | cpp-httplib 单头文件 | **1–1.5 人天** |
| **接线（插到 M2/M4 之间）** | 真实 `head.pt` 读取 + LoRA → GGUF 转换 + 张量映射 + Q8 温度校准 | `torch-pt` 解析 + convert 脚本 | `convert_lora_to_gguf.py` 现成 | **1–2 人天** |
| | | | **合计** | **6–11 人天** |

### 关键拆分（降低单点风险）
- **M1 拆两步**：① base + 随机 head 跑通链路（0.5–1 人天）；② 接真实 `head.pt`（需 .pt 解析）。可避免"链路没通就先纠缠权重格式"。
- **head.pt 读取（M1 第二步 / 接线）**：是唯一"地基"级风险。若引 libtorch 依赖巨大；建议**自写 minimal torch-`zipfile`/Uber (PYTORCH_FORMAT) reader 只读若干张量**（kev 的 q/k/温度，外加 LoRA A/B），或用 Python 先把 head.pt 导出成 `.bin`/JSON golden，C++ 只管加载——这样把「解析 torch 格式」从 C++ 任务里抽走，M1 更稳。

---

## 3. 各阶段验收「过/不过」的门槛（工作量不确定性来源）

| 阶段 | 过关标准 | 若不过，多花的变量 |
|---|---|---|
| M0 | 能读 hidden state 且范数合理 | 低（MSVC/CMake 版本问题，一般 1 轮内修）|
| M1 | base+随机head跑通；真实head时 max\|Δp\|<1e-2、argmax 全一致 | **中**：torch 格式解析、fp32 vs bf16 位对齐 |
| M2 | 逐 token / 逐索引与 `encode()` 一致 | **中**：QWEN 分词器 fast/legacy 差异、`<¦x¦>` 边界 |
| M3 | 同 state 复用后耗时下降且概率不变 | 低–中（ssm/反复式状态复制语义）|
| M4 | 通过 `kev` 的 `tests/test_api.py`（TypeSafe SDK）| 低–中（JSON schema 细节）|
| 接线 | 量化后温度不漂移（的 argmax 稳定）| 中（Q8_0 扰动 + 温度校准）|

> 一句话量化风险：**M1（+接线/head 解析）占到总工作量近 40%，也是唯一可能爆工期的地方**；其余各阶段较规整。

---

## 4. 投入节奏建议（里程碑-工作量）

```
M0  0.5–1 人天   打通构建+取向量        ← 今晚可完成
M1  1.5–2.5 人天 读出头+对拍             ★ 门槛
M2  1–2 人天     encode 对齐
接线 1–2 人天     head.pt + LoRA + 校准
M3  1–2 人天     前缀缓存
M4  1–1.5 人天    HTTP 服务
─────────────────────────────
总计 6–11 人天
```

**最短路径（砍 range）：** 若要快点见结果，可把「接线(真实权重)」延后到 M4 之后，先交付一个「base+随机head + 完整链路 + HTTP」的可运行版本，再补真实权重与精度——这样 4–6 人天就能有一个端到端可跑的 C++ kev。

---

## 5. 与「直接改 llama.cpp」相比的额外成本

- 采用复制策略的唯一持续成本是**版本固化/同步**：若要跟随上游，每次用 `rsync`/git subtree 拉一次，改动集中在 own 拷贝 → 每次 0.25–0.5 人天，一次性，不重复。
- 因为这些改动本来要落在 llama.cpp 里，现在落 own 拷贝，**工作量不增加**，只是把维护边界从上游挪到仓库内。