# kevcpp tests (M1 parity)

对拍目标：C++ 版 PointerHead（`src/kev_head.{h,cpp}`）与 Python/numpy 参考在**同一隐藏向量 + 同一权重**下概率一致。

## 运行
```bat
REM 1. 构建 (kev + headbench)
powershell -ExecutionPolicy Bypass -File ..\scripts\build.ps1

REM 2. 对拍 (需 proxy 环境 python; 会自动调 C++ headbench.exe)
D:\anaconda3\envs\proxy\python.exe parity.py
```
脚本（`parity.py`）工作流（全走文件，不用 subprocess 管道捕获，规避沙箱 EPERM）：
1. 生成随机 head -> `golden/head.bin`（C++ 读）+ `golden/head_weights.npz`（numpy 副本）。
2. **headbench（纯 PointerHead 数学对拍）**：随机构造 8 个 `(decide, opts)` 隐藏向量用例，
   numpy 按 kev PointerHead 公式算参考概率 -> `golden/cases.json`；调 C++ headbench 用同一
   head+向量算概率并与参考比较，报 `max|dp|`/argmax。
3. **readout（端到端）**：HF fast tokenizer 按 kev `encode()` 布局构造一个小请求
   （1 个 state + 2 个 question、不同 option 数），得到精确 ids 与 decide/opt 绝对索引 ->
   `golden/req.json`；调 C++ headbench `--mode readout`：llama.cpp prefill -> 在 decide/opt 位置取
   hidden 1024 维 -> 过同一 head 得 `golden/probs.json`，并导出 `golden/hidden.json`；
   用 numpy 在同一 head + 这些 hidden 上独立重算参考概率，与 `probs.json` 比较，报 `max|dp|`/argmax。

## 实测结果（随机权重，T=1.0）
- headbench: max|dp| = 0.0000000000, argmax 8/8
- readout  (2 questions): max|dp| = 0.000e+00, argmax_all_match = True

## 接入真实 head.pt
当前仓库内**没有真实训练权重**（kev 目录无 *.pt），故用随机权重跑通全链路。
拿到真实 `head.pt` 后导出并重跑即可：
```bat
D:\anaconda3\envs\proxy\python.exe gen_head_weights.py --head <head.pt> --out golden\head.bin
D:\anaconda3\envs\proxy\python.exe parity.py      # 直接用真实 head 重算参考概率并比对
```
`gen_head_weights.py` 读取 `head.pt` 中的 `q.weight/q.bias/k.weight/k.bias + temperature`
（`kev.checkpoint.Meta` 布局），写出 C++ 的 KEVH .bin。

## 命令直达
- `headbench.exe --mode headbench --head golden\head.bin --cases golden\cases.json`
- `headbench.exe --mode readout --model <gguf> --req golden\req.json --head golden\head.bin [--probs-out p.json] [--hidden-out h.json]`

## 备注
- PointerHead 参考实现：kev/model.py:182-200。`z=(k(h_opts)@q(h_decide))*scale`，`scale=1/sqrt(256)`，
  概率 = softmax(z/T)，T=1.0 原始。nn.Linear：y=x@W^T+b。
- 对拍用 llama.cpp 的 hidden（M0 已证有效）作为端到端基础；本轮不依赖 torch 主干前向
  （kev 的 transformers 5 因 `LinearAttentionCacheLayerMixin` 缺失无法 import，待真实链路时再补）。

---

# M2 —— encode 逐 token 对齐

`src/kev_encode.{h,cpp}` 复刻 kev.model 的 `user_tokens`/`encode`/`rows_of`：
- `SPECIAL=["<|fim_prefix|>","<|fim_middle|>","<|box_start|>","<|box_end|>","<|fim_suffix|>"]`，最后一个是 `<decide>`。
- `user_tokens`：`<|x|>` -> `<¦x¦>` 后用 `llama_tokenize(add_special=false, parse_special=true)`。
- `encode()` 产出 `ids/seg/pos/opt/decide_idx/opt_idx`（`opt` 用 `OPT_NONE=-1`/选项 span j/`OPT_DECIDE=-2`）。
- `rows_of()` 把打包形式拆成 `state + per-question branch rows`（decide/opts 为 row 内相对偏移）。

## 分词器对拍（关键前置，已确认）
`tests/tokenizer_probe.py`：对含数字/标点/`<|x|>`/非 ASCII 的 8 类样例，
把 (kev 改写后) 文本分别用 HF fast tokenizer 与 `llama_tokenize` 分词对比。
实测：`llama_tokenize`(parse_special=true) == HF fast tokenizer，RESULT: PASS。
因此 C++ 用 GGUF 自带 tokenizer 即可，无需引入 HF。
（特殊分隔 token 的 id 用 `parse_special=true` 对 `"<|name|>"` 分词取最后一个 id，规避 pre-token 前缀伪 token。）

## encode 对拍（验收）
`tests/parity_encode.py`：参考实现逐行复刻 kev.model 的 encode/rows_of（仅用 HF tokenizer，
不经 kev.model，规避 transformers 5 的 mixin 缺失），与 C++ `headbench --mode encode` 输出比较。
覆盖 2 个请求（含 `<|x|>` 改写、多问、数字、"B" 单字符选项等）。
实测：`ids/seg/pos/opt/decide_idx/opt_idx` + 每行 `decide/opts` 逐 token/逐索引一致，ALL: PASS。

## 运行
```bat
powershell -ExecutionPolicy Bypass -File ..\scripts\build.ps1
D:\anaconda3\envs\proxy\python.exe tests\tokenizer_probe.py   # 分词对拍
D:\anaconda3\envs\proxy\python.exe tests\parity_encode.py     # encode 逐 token 对拍
```

## 途经
- `headbench.exe --mode encode --model <gguf> --req req.json`   # 打印 packed + rows JSON

---

# M3 —— 状态前缀缓存 / KV 复用

`src/kev_prefix.{h,cpp}`(状态 LRU 索引) + `src/kev_model.{h,cpp}`(组装层) 复刻 kev 的
`prefix()`/`probs_with_prefix()`/`probs_and_prefix()` 与 row form 的状态复用：
- 同一 state 只 prefill 一次(缓存进一个 seq slot 的 KV + Gated DeltaNet 反复式状态)。
- 每个 question 作为 continue row：`llama_memory_seq_cp(slot -> scratch)` 把 state 前缀(含反复式状态)
  复制到 scratch seq，只解码 branch tokens(pos 从 state 末尾继续)，在 decide/`</opt>` 位置取 1024 hidden，
  过 PointerHead 得概率。`prefix_min_tokens=0`(hybrid 路径)。
- LRU(默认 2 个 state slot)：命中直接用 slot；未命中把新 state prefill 到 LRU 淘汰的 slot。
- `KevModel::probs()` 一次请求 = ensure_state(命中=仅查索引 / 未命中=state prefill) + 各问 branch decode+head，
  并记录 t_state / t_rows / t_total。

## 运行
```bat
powershell -ExecutionPolicy Bypass -File ..\scripts\build.ps1   # 产出 kev/headbench/m3_bench
build\bin\Release\m3_bench.exe --model <gguf> --head tests\golden\head.bin --req tests\golden\req_m3.json --lru 2
```

## 实测（Qwen3.5-0.8B Q8_0, lru=2; head 随机权重）
```
pass1(cold) : state_hit=no  state=210.4ms  rows=496.4ms  total=707.2ms
pass2(same) : state_hit=YES state=0.0ms    rows=496.7ms  total=497.1ms   <- 不重跑 state
pass3(other): state_hit=no  state=151.9ms  rows=501.0ms  total=653.3ms   <- 不同 state 未命中(驱逐并重 prefill)
pass4(orig) : state_hit=YES state=0.0ms    rows=498.4ms  total=498.7ms   <- 原 state 再次命中
probs: pass1 vs pass2 max|dp|=0.000e+00 ; pass1 vs pass4 max|dp|=0.000e+00
speedup(hit vs cold total) = 1.42x
RESULT: PASS
```
验收：同 state 第二次 hits 不重跑 prefill(state 210ms -> 0ms)，总耗时 707 -> 497ms；概率完全不变(0.000e+00)。

---

# M4 —— /v1/systemone HTTP 服务 (TypeSafe 兼容)

`src/kev_api.{h,cpp}`(请求/响应映射: render/option_text/question_keys/to_record/to_answers/confidence，
复刻 kev/api.py) + `src/main_server.cpp`(cpp-httplib HTTP 服务器)。复用 M1 PointerHead、M2 encode、M3 前缀缓存。

## 从冷启动到发一个完整 systemone 请求
```bat
REM 1) 构建 (kev / headbench / m3_bench / main_server)
powershell -ExecutionPolicy Bypass -File scripts\build.ps1

REM 2) 启动服务器 (无认证; 端口默认 8008; --lru 状态缓存 slots)
build\bin\Release\main_server.exe --model D:\share-models\Qwen3.5-0.8B-Q8_0.gguf ^
    --head tests\golden\head.bin --host 127.0.0.1 --port 8008 --lru 2
REM    日志末尾: "Kev CPU server on 127.0.0.1:8008 (auth=off ...)"

REM 3) 发一个含 noul/choice/score 三类问题的请求
curl.exe -s -X POST http://127.0.0.1:8008/v1/systemone ^
    -H "Content-Type: application/json" ^
    -H "x-typesafe-request-id: demo-1" ^
    -d "{\"state\":\"Qwen3.5 is a hybrid causal decision model.\",\"questions\":{\"is_valid\":{\"type\":\"noul\",\"instructions\":\"Is it true?\"},\"best\":{\"type\":\"choice\",\"instructions\":\"Pick best.\",\"criteria\":{\"A\":\"x\",\"B\":\"y\"}},\"level\":{\"type\":\"score\",\"instructions\":\"Rate 0-2.\",\"criteria\":[\"low\",\"mid\",\"high\"]}}}"

REM 4) 可选认证: 设环境变量 KEV_API_KEY=xxx 后再启动, 请求需带 Authorization: Bearer xxx
REM     GET /v1/models 也返回双模型卡 kev-latest / jev-latest
```

## 端点
- `POST /v1/systemone`：请求 `{state, model?, questions:{qid:{type:noul|choice|score, instructions?, criteria}}}`
  → 响应 `{model, answers:{qid:{...}}, usage:{input_tokens,output_tokens}, latency_ms}`。
  - noul → `{type,noul}`; choice → `{type,choice,confidence,probabilities}`; score → `{type,score,confidence,legend,probabilities}`。
- `GET /v1/models`：`{models:[{name,description,release_date,...}]}`，含 kev-latest 和 jev-latest 两张卡。
- 可选 bearer auth(KEV_API_KEY)；`x-typesafe-request-id` 回显或生成；`server-timing` 头。

## 对拍/验收（已实际跑）
`tests/run_test_api.py`：复刻 kev/tests/test_api.py 的非 SDK 断言(httpx 直打本服务)：
choice_basic / five_questions_and_null_descriptions / structured_instructions_and_criteria /
noul_score_and_object_state / optional_instructions_and_one_level_score / validation_422 /
packed_equals_separate / models —— **8/8 全过 (RESULT: ALL PASS)**。
说明：proxy 环境缺 pytest 与 typesafe_sdk，故用 httpx 复刻同一批断言(接口字节级一致；
typesafe_sdk 客户端命中同一端点)。HTTP schema/Auth/request-id 均已实测(401/200)。

## 途经
- `tests/run_test_api.py`（服务器在线时运行: python tests\run_test_api.py）

---

# M5 —— 接入真实 Kev-0.8B LoRA adapter（官方概率对齐）

把 `jaredpalmer/kev-0.8b` 的真实 PEFT LoRA adapter(r=16, alpha=32, dropout=0.05)折进 base GGUF，
以 llama.cpp 原生 LoRA 机制做成 GGUF adapter 加载，配合真实 head(在 models/head.bin, temp=2.351)，
使服务输出官方量级的真实概率。

## 权重来源
- 已下载：`models/adapter/adapter_config.json` + `adapter_model.safetensors`（372 张量，43MB）。
  经 hf-mirror.com 镜像 `urllib` 下载(`models/download_adapter.py`)——直连 HF/官方超时，镜像 OK。
- 真实 head：`models/head.pt` → `models/head.bin`（`models/build_real_head_bin.py`，temp=2.351, dim_in=1024, dp=256）。

## 张量名映射（core，已核对）
`base_model.model.layers.{N}.<mod>` → llama.cpp GGUF `blk.{N}.<t>`（LoRA 张量名 = 基名含 `.weight` + `.lora_a`/`.lora_b`）：
- DeltaNet 层(0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22)：`linear_attn.in_proj_qkv→attn_qkv`、
  `in_proj_z→attn_gate`、`in_proj_a→ssm_alpha`、`in_proj_b→ssm_beta`、`out_proj→ssm_out`。
- full-attention 层(3,7,11,15,19,23)：`self_attn.q/k/v_proj→attn_q/k/v`、`o_proj→attn_output`。
- 全 24 层 MLP：`mlp.gate/up/down_proj→ffn_gate/ffn_up/ffn_down`。
依据：`llama.cpp src/models/qwen35.cpp` 的 GGUF 张量名 + transformers modeling_qwen3_5.py
(`in_proj_b→b, beta=b.sigmoid()` => ssm_beta；`in_proj_a→a, g=softplus(a+dt)` => ssm_alpha)。
合并公式 = `mlx_model.merge_lora`：`W + (B@A)*alpha/r`，r=16 alpha=32 ⇒ 因子 2.0。

## 生成 + 验证（已实际跑）
```bat
set PYTHONPATH=E:\van\projects\llama.cpp\gguf-py
REM 1) adapter -> GGUF lora (KV: general.type=adapter, adapter.type=lora,
REM    adapter.lora.alpha=32, general.architecture=qwen35, 32 字节对齐, LITTLE_ENDIAN)
D:\anaconda3\envs\proxy\python.exe models\build_lora_gguf.py        # -> models\kev-lora-qwen35.gguf (43MB, 372 tensors)
REM 2) 独立校验: GGUF lora 值与 adapter 逐位一致 + delta 幅值
D:\anaconda3\envs\proxy\python.exe models\verify_lora.py            # 186 对, 值 0 误差; delta(O(1e-3))
REM 3) base vs base+LoRA 对拍 (real head)
build\bin\Release\m5_bench.exe --model D:\share-models\Qwen3.5-0.8B-Q8_0.gguf --head models\head.bin --req tests\golden\req_m5.json [--lora models\kev-lora-qwen35.gguf] [--lora-scale f]
REM 4) 服务器带真实 LoRA 起:
build\bin\Release\main_server.exe --model ...--head models\head.bin --lora models\kev-lora-qwen35.gguf --port 8008
```

## 验收数据（real head temp=2.351）
```
llama_adapter_lora_init_impl: loaded 372 tensors from lora file   (全部命中 base 张量, 无 "does not exist")
verify_lora: 186 对 tensor 值 0 误差; delta=(B@A)*2 每张量 max|.| min=1.1e-3 median=3.8e-3 max=9.9e-3
m5_bench req_m5 (state + 3 问):
  q0(no,  yes):  base 0.521 0.479   |  +LoRA 0.072 0.928
  q1(4 选项)    :  base 0.150 .115 .345 .391 | +LoRA 0.929 .018 .028 .024   (argmax 0.345(q2)->0.929(q0))
  hidden l2     :  base ~122 -> +LoRA ~36
lora-scale 扫描(连续生效, 线性正确): s=0  与 base 逐位一致; s=0.5/1.0/2.0 单调移动
服务器 +LoRA systemone: noul=0.954, choice confidence=0.051-0.31, score 合理 —— 官方量级(非 one-hot)。
```
验收口径说明：proxy 环境 transformers 5.3.0 缺 `LinearAttentionCacheLayerMixin`，无法 import kev.model 跑完整
torch 前向做端到端参考；故按任务允许的方式，用「同一 llama.cpp forward 下 base vs base+LoRA 的 hidden/概率差异」
作为 delta 生效证明（差异巨大，且 scale=0 逐位等于 base、scale 单调 → 机制正确非噪声）。GGUF lora 值与 PEFT 逐位
一致 + 张量名全命中 base + 合并公式 == mlx_model.merge_lora，共同支撑映射与算术正确。

## 途经
- `models/build_lora_gguf.py`(adapter→GGUF lora) / `models/verify_lora.py`(值/幅值校验) /
  `models/build_real_head_bin.py`(head.pt→head.bin) / `models/download_adapter.py`(镜像下载) /
  `src/m5_bench.cpp`(base vs +LoRA 落盘对拍) / `src/main_server.cpp` --lora

---

# M6 —— 精简服务日志（仅保留服务状态 + 请求时间）

`src/main_server.cpp` 用 `llama_log_set(llm_log_filter, nullptr)`（在任何模型动作之前调用）接管
llama.cpp/ggml 日志，回调里只透传 `GGML_LOG_LEVEL_ERROR`，过滤 INFO/WARN/DEBUG
（消灭 `llama_model_loader metadata dump` `- kv ...`、`create_tensor: loading tensor ...`、
`llama_graph_n_input_tensors ...`、`sched_reserve:`、`~llama_context: CPU compute buffer size is` 等刷屏）。

保留输出：
1. 启动：`kevcpp server loading model: <path>` → `kevcpp server model ready (...)` → `kevcpp server listening on host:port`。
2. 每请求一行：`[HH:MM:SS] METHOD /path <status> <ms>ms input=<n> output=<m>`（含 200/400/401/422）。
3. 错误仍清晰：`kevcpp error: init failed: <msg>`（加载失败/端口占用由 listen 返回等）。
4. 可选 `--quiet`：连请求/启动行也隐藏（只留错误与 listening）。

验证（real head + LoRA, port 8008）：
```
# 启动(默认)
build\bin\Release\main_server.exe --model D:\share-models\Qwen3.5-0.8B-Q8_0.gguf --head models\head.bin --lora models\kev-lora-qwen35.gguf --port 8008

kevcpp server loading model: D:\share-models\Qwen3.5-0.8B-Q8_0.gguf
....................................................................
kev_model: lora adapter loaded (E:\van\projects\kevcpp\models\kev-lora-qwen35.gguf, scale=1.00)
kevcpp server model ready (n_embd=1024 temp=2.351 head=...head.bin lora=...kev-lora-qwen35.gguf)
kevcpp server listening on 127.0.0.1:8008  (POST /v1/systemone, GET /v1/models; auth=off)
[15:45:07] POST /v1/systemone 200 832ms input=81 output=163
[15:45:07] GET /v1/models 200 0ms input=0 output=0
[15:45:07] POST /v1/systemone 422 0ms input=0 output=0
```
（完整示例见 tests/golden/m6_trim.log：before 有千行刷屏，after 仅 5 行启动 + 每请求一行。）

## 途经
- `src/main_server.cpp`(llama_log_set 过滤 + 请求计时行 + --quiet)；`tests/golden/m6_trim.log`(精简输出记录)