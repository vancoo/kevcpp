# kevcpp

> kev's native C++17 implementation (based on llama.cpp), re-implementing kev (Python) inference/serving behavior in C++.

kevcpp is a pure-CPU (currently), llama.cpp (https://github.com/ggml-org/llama.cpp) based kev inference service.
It uses the Python reference implementation kev (`https://github.com/jaredpalmer/kev`) as its behavioral baseline and
re-implements the `DecisionModel` / state-prefix cache `/v1/systemone` serving semantics in C++/llama.cpp.

The goal is: **provide a download-and-run decision inference engine**, avoiding kev's development-environment setup,
and to some extent rework/improve inference performance.

> Due to limited conditions, it has only been run on the development machine (AMD AI MAX 395 PRO+); more people are
> welcome to help improve and test it.
> Roadmap:
> 1. Add GPU backend support: split the backend into a standalone DLL, loaded by command-line option.
> 2. Add model support: pin different model support into concrete DLLs, loaded on demand.

---

## Table of Contents

0. [Quick Start](#0-quick-start)
1. [Current Status](#1-current-status)
2. [Based on llama.cpp and kev (What We Reference)](#2-based-on-llamacpp-and-kevwhat-we-reference)
3. [Environment & Validation](#3-environment--validation)
4. [Advantages over kev](#4-advantages-over-kev)
5. [Limitations (Honest)](#5-limitations-honest)
6. [Usage, Dependencies, Requirements](#6-usage-dependencies-requirements)
7. [Architecture](#7-architecture)
8. [Model / HuggingFace Hub](#8-model--huggingface-hub)

---

## 0. Quick Start

> This section is for users who "just want to run it": how to get the binary and model, how to start `main_server`,
> and how to craft a request message to quickly verify.
> If you need to build from source yourself, jump to [§6 Usage, Dependencies, Requirements](#6-usage-dependencies-requirements).

### 0.1 Where to download the binary and model files

**Binary files (Release executables)**

The repository currently does not ship a prebuilt CI release package (`main_server` is still distributed as source; see §6).
Available ways to obtain it:

| Source | Description |
| --- | --- |
| **Build locally** | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release`; the artifact lands at `build/bin/main_server.exe` (Windows) / `build/bin/main_server` (Linux). Recommended. |
| **GitHub Releases** | Windows 0.1.0 has been released to the Releases page. It targets AMD CPUs; Intel CPUs are untested, and the Linux build has not been compiled due to a missing environment. More contributors are welcome. |

**Model files**

The 0.8B model used by this project is **not an off-the-shelf Qwen-0.8B**, but a customized variant of this project
(see §8 for the structure). You need two files:

| File | Approx. size | Description |
| --- | --- | --- |
| `kev-merged-q8.gguf` | ~774 MiB | Main model weights (Q8_0 quantized); corresponds to the `--model` option |
| `head.bin` | ~2 MiB | PointerHead readout head weights; corresponds to the `--head` option |

Download location:

- **Hugging Face Hub**: `https://huggingface.co/vanncoo/kevcpp-qwen3.5-0.8b` (repository ID
  `vanncoo/kevcpp-qwen3.5-0.8b`). The upload will include both the GGUF weights and the PointerHead. **If it has not
  been uploaded yet, rely on the `models/` directory in the repository.**
- **Local `models/` directory**: the repo's bundled `models/kev-merged-q8.gguf` and `models/head.bin` (use them directly if present).

> 💡 Full model spec and HF Hub notes are in [§8 Model / HuggingFace Hub](#8-model--huggingface-hub).

### 0.2 How to start the service

After placing the model files in `models/`, run (example assumes the model is under `models/`):

```bash
# Windows / Linux (equivalent)
./build/bin/main_server \
    --model models/kev-merged-q8.gguf \
    --head  models/head.bin \
    --port  8008 \
    --threads 16
```

- Seeing the following logs means it started successfully and is listening:

```
kevcpp server model ready (...)
kevcpp using 16 threads ...
kevcpp warmed decode graph (state+branch) before listen
```

- It listens on `127.0.0.1:8008` by default. Health check (or model info):

```bash
curl http://127.0.0.1:8008/v1/models
```

- If the environment variable `KEV_API_KEY` is set, every request must carry the
  `Authorization: Bearer <KEV_API_KEY>` header (otherwise you get `401`).

### 0.3 How to quickly build a test request message

The endpoint is **`POST /v1/systemone`**; the body is JSON with two fields:

- `state`: business context/raw state text (a string; any text works).
- `questions`: an object whose key is the question id and whose value is `{ "type": <type>, "criteria": <criteria> }`.
  - `type` supports `noul` (binary pick), `choice` (pick one of many), and `score` (rating level).
  - `criteria`: for `noul`/`choice` pass a `{ "option_key": "option_description" }` object; for `score` pass an array of level descriptions.

A minimal example `request.json`:

```json
{
  "state": "You are the CEO of a small manufacturing company. Capacity is near its ceiling and cash is ample.",
  "questions": {
    "q_expand": {
      "type": "choice",
      "criteria": {
        "expand_capacity": "Invest to expand production line capacity",
        "invest": "Increase R&D or marketing spend",
        "hold": "Keep the status quo"
      }
    }
  }
}
```

Then send the request (`curl` works on both Windows and Linux):

```bash
curl.exe -s -X POST http://127.0.0.1:8008/v1/systemone \
    -H "Content-Type: application/json" \
    --data-binary @request.json
```

Example response (summary: `answers` gives each question's conclusion, `conf` is the confidence, `usage` carries token stats):

```json
{
  "answers": { "q_expand": "expand_capacity" },
  "conf": 0.5,
  "probs": { "q_expand": { "expand_capacity": 0.6, "invest": 0.3, "hold": 0.1 } },
  "usage": { "input_tokens": 100, "output_tokens": 10 }
}
```

> Compatibility note: passing the full `state` verbatim yields bit-identical results with the reference implementation.

### 0.4 CLI option reference

| Option | Description |
| --- | --- |
| `--model <gguf>` | Model file path (**required**, `kev-merged-q8.gguf`) |
| `--head <bin>` | PointerHead `.bin` path |
| `--host <ip>` | Listen address (default `127.0.0.1`) |
| `--port <p>` | Listen port (default `8008`) |
| `--threads <n>` / `-t <n>` | Compute threads (`0` / unset = auto-detect, capped at 32) |
| `--ctx <n>` | Context size (default `8192`; `2048` is typical for short states) |
| `--lru <n>` | State cache slots (default `2`) |
| `--prewarm <file>` | Prewarm a state at startup (read raw state text, prefill into LRU; optional) |
| `--lora <gguf>` | Additionally load LoRA weights (optional) |
| `--lora-scale <f>` | LoRA scale factor (paired with `--lora`; optional) |
| `--quiet` | Suppress per-request log lines |
| `--help` / `-h` | Print usage and exit |

> `--backend` (pluggable CPU/GPU backend) exists only on the archived branch `build-portable-backends`; it is not
> enabled in the current mainline build, and GPU backends (`gpu`/`cuda`/`vulkan`) are not implemented yet. See §5.6.

---

## 1. Current Status

- **Stable mainline: `master` @ `dd51ed7`**. Steady-state latency ≈ **~327 ms** (short requests), with output
  bit-identical to the reference. Includes three landed optimizations: removing the unused tied lm_head
  (Rank-1, `267300f`), default ctx 8192 with a clear error for over-long state (`dd51ed7`), and a threaded/vectorized
  `ggml concat_f32` (`2607590`).
- **Portable / GPU backend: ARCHIVED on branch `build-portable-backends` @ `71d2c64`**. The pluggable backend design
  (`kev::Backend` C-ABI + `--backend` + GGML_CPU_ALL_VARIANTS portable SIMD scheme) is committed as source, but **has not
  yet been integrated into a final shared-library build**; when distributing you **must strip the VNNI DLLs** (see §5).
- The current CPU hard-requires **AVX-512** (Zen4+/Sapphire Rapids level), because the mainline build is `GGML_NATIVE`
  (see §5, §6).

---

## 2. Based on llama.cpp and kev (What We Reference)

kevcpp stands on two upstreams:

### 2.1 llama.cpp (vendored, `third_party/llama.cpp`)

llama.cpp is vendored as a full copy under `third_party/llama.cpp` (introduced via `add_subdirectory` in `CMakeLists.txt`,
unmodified beyond this repo's changes to `llama-graph.cpp`/`llama-context.cpp` — the vendored layer stays read-only).
Parts referenced/reused:

- **GGUF / GGML**: models load as GGUF (`models/kev-merged-q8.gguf`, Q8_0 quantized); `llama`/`ggml` provide tensor loading
  and math.
- **CPU compute kernels**: Q8_0 tiled GEMM (`MUL_MAT`), `SSM_CONV`, `GATED_DELTA_NET`, `FLASH_ATTN_EXT`, `RMS_NORM`, etc.
- **Hybrid model builder**: `src/models/qwen35.cpp` builds **24 layers = 18 Gated-DeltaNet recurrent layers + 6 full-attention
  layers**; the attention layers sit at layers 3, 7, 11, 15, 19, 23.
- **`llama_decode`**: the core forward entry point; kevcpp does exactly one 151-token `llama_decode` per steady-state request.
- **KV / recurrent state cache**: the actual storage/reuse mechanism for the KV cache and Gated-DeltaNet recurrent state.
- **Hierarchical backend / loading**: `llama_model_loader` + `ggml_backend` loading and (in the portable scheme) automatic CPU
  variant selection.

### 2.2 kev (Python, reference implementation)

kev is the Python serving implementation (`model.py` / `serve.py` / `api.py`) in the adjacent repo `E:\van\projects\kev\kev`.
kevcpp **re-implements kev's inference and serving behavior** as its reference:

- **`DecisionModel`** (`model.py`): `probs` / `encode` / state-prefix cache interface. kevcpp's `KevModel`
  (`kev_model.cpp/h`) implements a subset of its interface (probs / encode / prefix cache).
- **Serving `serve.py`** (with state-prefix cache): HTTP serving and the same-state second-request hit semantics.
- **`systemone` request/response schema** (`api.py`'s `render` / `option_text` / `question_keys` / `to_record` /
  `to_answers` / `confidence` semantics): replicated in `kev_api.cpp` and exposed as `/v1/systemone`.
- **PointerHead**: the PointerHead class in `model.py` (head.bin weights) — `kev_head.cpp` ports its forward math
  (`k(h_opts) · q(h_decide) * scale`, softmax, temperature).

In one sentence: **kevcpp faithfully re-implements kev's inference/serving behavior in C++17 on top of llama.cpp.**

---

## 3. Environment & Validation

### 3.1 Development environment

| Item | Value |
| --- | --- |
| OS | Windows (MSVC 2022, Visual Studio Community) |
| Compiler | MSVC 2022, `/arch:AVX512`, C++17 |
| Build | CMake ≥ 3.21, **Release / x64** |
| CPU | AMD Ryzen AI MAX+ PRO 395 (Zen5, 32 logical cores, full-width AVX-512 + AVX-VNNI `vpdpbusd`) |
| Model | `models/kev-merged-q8.gguf` (Q8_0, ≈ **774 MiB**) + `models/head.bin` (≈ 2 MiB) |

Linux is also an acceptable target (GCC/Clang toolchain, see §6).

### 3.2 Passed validation (byte-faithful)

Strict byte-faithful protocol (`curl.exe --data-binary @<file>`, each run confirms sides' `input=… output=81`):

- **Short request** `tests/tmp_a_company.json` (`input=189`, state 38 + branch 151).
- **Long request** `tests/tmp_long_state_request.json` (`input=263`, over-long state).
  (The above `tests/*.json` request fixtures are in the original kevcpp repo's `tests/` directory.)
- **Known-correct probability tuple** (bit-identical to the kev reference, treated as a red-line):

| Category | Awarded | Capacity | Other | Plant | Invest | Land |
| --- | --- | --- | --- | --- | --- | --- |
| Probability | 0.0026 | **0.3974** | 0.0272 | 0.1731 | 0.396 | 0.0037 |

- Short-request conf **0.2769**, long-request conf **0.3691**; both choices are **Capacity**.
- Note: `Capacity` (0.3974) and `Invest` (0.396) are a genuine near-tie (gap 0.0014) — a selection that various
  approximate optimizations would flip at the near-tie point (see §5.3).

- **Landed optimizations and results** (master @ `dd51ed7`):
  - Removed the unused tied lm_head (Rank-1, ≈ -100 ms) → steady-state fork forward **~440 ms → ~343 ms**.
  - Fixed default ctx to 8192 + clear error on over-long state.
  - Threaded/vectorized `ggml concat_f32`.
  - Combined steady-state fork forward **~440 ms → ~327 ms** (short-request median 327.5 ms; long-request median 333.8 ms).

- **Pre-optimization baseline (authoritative)**: steady ~440 ms (`logs/perf_authoritative_baseline.md`, original kevcpp repo),
  MUL_MAT ≈ 335 ms (78%, ~339 GMAC/s), CONCAT ≈ 38 ms, GATED_DELTA_NET ≈ 31 ms.

**Project charter (objective)**: prove kevcpp is **not slower** than Python kev in steady-state latency, and keep
compressing that latency.

---

## 4. Advantages over kev

1. **Native C++/llama.cpp**: no Python runtime / no GIL; lower per-request overhead and memory usage; the vendored
   llama.cpp is a mature GGUF/Q8 inference stack (weight loading, graph, CPU kernels).
2. **State-prefix LRU cache**: the state-prefix (LRU) cache in `kev_prefix.cpp` lets **repeat states skip state prefill**;
   at steady state each request only decodes the branch, and the graph topology is reused.
3. **Output quality bit-identical to the reference (hard red-line)**: on the tested requests, `kevcpp` produces
   probabilities **bit-identical** to the Python kev reference, guaranteeing a drop-in replacement does not change results.
4. **Other ongoing advantages** (listed once real/landed):
  - **merged-LoRA single-weight forward** (`e83a9be` / `src/merge_lora_tool.cpp`): folds the LoRA into the Q8_0 base weights,
    forwarding with **a single set of weights**, avoiding runtime LoRA stacking overhead.
  - **graph warmup** (`6716f35`): `main_server` warms the graph at startup, removing ~290 ms of cold-start graph building.
  - **multi-question batched decode** (P.4, `49b6cf9`): batched decode for equal-length multi-questions (bit-identical
    precondition holds).

> **Honest claim about "faster than kev"**: this project has verified via strict byte-faithful measurement that steady-state
> latency beats Python kev (data in the original kevcpp repo's `docs/` logs). But **the absolute cross-repo numbers should be
> re-measured on your own hardware** — benchmarks differ across machines/loads; this document promises no specific absolute ratio.

---

## 5. Limitations (Honest)

### 5.1 Local Q8_0 CPU GEMM is compute-bound

- On this die (Zen5), the Q8_0 CPU GEMM is measured compute-bound at about **~339 GMAC/s (≈6% of Zen5's int8 peak)** — this
  is the dominant part of steady-state latency (MUL_MAT ≈ 78%).
- Every "faster kernel" direction was **measured to lose or end-to-end falsified** (see original kevcpp repo's `docs/F.24_…`):
  - 512-bit `vpdpbusd` Q8×Q8 (~190–256 GMAC/s, on par with the current tiled) — **falsified**;
  - Q8×F16 activation pair path (FMA, microbenchmark ~380 GMAC/s, but each GEMM needs an F32→F16 conversion eating the
    gain; zero end-to-end gain) — **falsified**;
  - F16 pre-dequantized weights (wins the microbenchmark, but is ~110 ms slower end-to-end when integrated) — **falsified**.

### 5.2 VNNI/dpbusd numeric errors (must strip from Portable distro)

- **AVX512_VNNI / AVX_VNNI Q8×Q8 kernels are numerically wrong on this model** (short conf **0.1222** vs the correct
  **0.2769**).
- Therefore the `GGML_CPU_ALL_VARIANTS` portable distribution **must strip the VNNI DLLs**:
  `ggml-cpu-{cascadelake,icelake,alderlake}.dll` (keep `x64,sse42,sandybridge,haswell,skylakex,cannonlake`).
- This is an **architectural property** of the Q8×Q8 dpbusd path on this model, not a tuning issue — strip unconditionally.
  This is consistent with the Q8×Q8 dpbusd failure in `F.24` (original kevcpp repo `docs/`).

### 5.3 Near-tie sensitivity

- The classifier's **sharp linear head + residual network** means any approximate optimization (SILU/SwiGLU activation
  sparsity, LCP/prefix reuse, low-rank) **flips the selection** at near-ties (e.g. `Capacity 0.353` vs `Invest 0.356`).
  Under the strict bit-identity red-line, **no approximate direction is shippable**.

### 5.4 Prefix/LCP reuse and throughput batching (same-state merge) ≈ zero real gain

- **LCP/prefix reuse**: technically feasible, but ROI≈0 under real traffic (different states share only ~8% of the prefix);
  deeper prefix-slot designs **break bit-identity** (batch-width numeric dependency), all reverted (F.24 §9.4/§9.5/§9.6,
  original kevcpp repo `docs/`).
- **Same-state batching (throughput)**: batch-width merging yields **no weight-read reuse gain** under compute-bound
  conditions; measured req/s barely improved (~3.07 vs 2.97), and merging >512 tokens trips the `n_batch` assertion —
  reverted (F.24 §11, original kevcpp repo `docs/`).

### 5.5 Host stability finding

- On the dev machine, **intermittent heap corruption / fast-fail under sustained request load** (suspected memory pressure;
  no debugger available to root-cause).
- This **blocks full load validation of the shared-library / portable build** (see original kevcpp repo
  `docs/portable_backend_deployment.md` §7).

### 5.6 Portable / GPU backends not final

- The pluggable `kev::Backend` + `--backend` + `GGML_CPU_ALL_VARIANTS` design **is archived on branch
  `build-portable-backends` @ `71d2c64`**, but **not yet integrated into the stable shared-library build**.
- The current CPU hard-requires **AVX-512** (Zen4+/Sapphire Rapids level), because the mainline build is `GGML_NATIVE`.
- The GPU backend is only an extension point (`--backend gpu/cuda/vulkan` currently returns a clear "not built" error).

---

## 6. Usage, Dependencies, Requirements

### 6.1 Running

```bash
# Build (CMake ≥ 3.21, MSVC 2022 Release/x64 or GCC/Clang)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Run the service
./build/bin/main_server \
    --model models/kev-merged-q8.gguf \
    --head  models/head.bin \
    --port 8008 \
    --threads 16
```

### 6.2 Endpoints

- **POST `/v1/systemone`**: the same JSON schema as kev (`state` + `questions`); request fixtures are in the original kevcpp
  repo's `tests/*.json` (`tmp_a_company.json` short request, `tmp_long_state_request.json` long request).
- **GET `/v1/models`**: model/backend info (including `backend_name`/`device`/`version`, on the portable branch).

### 6.3 Main CLI options

| Option | Description |
| --- | --- |
| `--model <gguf>` | Model path (required, `kev-merged-q8.gguf`) |
| `--head <bin>` | PointerHead `.bin` path |
| `--host ip` / `--port p` | Listen address/port (default `127.0.0.1:8008`) |
| `--threads N` / `-t N` | Compute threads (0/unset = auto-detect, capped at 32) |
| `--ctx N` | Context size (default **8192**; 2048 is fine for short states) |
| `--lru N` | State cache slots (default 2) |
| `--prewarm <file>` | Prewarm a state at startup (read raw text, prefill into LRU; optional) |
| `--backend <name>` | Inference backend: `cpu` (default); `gpu`/`cuda`/`vulkan` return a "not built" error (portable branch) |

> `--backend` is only available on the archived branch `build-portable-backends` (the current mainline build is a static
> single-llama `GGML_NATIVE`).

### 6.4 Dependencies

- **CMake ≥ 3.21**
- **C++17 toolchain** (MSVC 2022 / GCC / Clang)
- **vendored llama.cpp** (`third_party/`, no external download needed)
- cpp-httplib, nlohmann/json (vendored, used at compile time)

### 6.5 Environment requirements

- OS: **Windows** (tested) / **Linux** (acceptable).
- CPU instruction set: **AVX2+ guarantees correctness**; **AVX-512 for expected performance** (the current mainline build
  is `GGML_NATIVE`, so AVX-512 is the default hard requirement).
- Memory: with the default `n_ctx=8192` about **~1.4 GB to start** (model ~0.7 GiB + KV ~0.34 GiB + compute buffers
  ~0.24 GiB). Larger inputs can raise `n_ctx` / `n_seq_max`.

---

## 7. Architecture

### 7.1 Layers

```
┌────────────────────────────── kevcpp (orchestration) ──────────────────────────────┐
│ (a) main_server.cpp   HTTP layer (cpp-httplib), serving the systemone schema         │
│ (b) KevModel          kev_model.cpp/h —— orchestration: LLM ctx ownership, encode,    │
│      kev_prefix.cpp   state LRU prefix cache, kev_head.cpp PointerHead readout head,  │
│      kev_encode.cpp   per-question fork decode/match, compute_mutex_ serializing shared ctx │
│      kev_backend_factory.cpp  pluggable backend factory (ARCHIVED)                    │
├────────────────────────────── vendored llama.cpp / GGML ───────────────────────────┤
│ (c) hybrid GDN+attention graph, Q8_0 kernels, KV/recurrent cache, encode/decode,     │
│      qwen35.cpp builder (24 mixed layers)                                            │
│ (d) kev_backend* C-ABI abstraction (ARCHIVED, branch build-portable-backends)        │
│     —— CPU and future GPU backends loadable via --backend as DLLs                    │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Module responsibilities

- **(a) `main_server.cpp`**: HTTP layer for `/v1/systemone` (cpp-httplib), arg parsing, serial/parallel.
- **(b) `KevModel` (`kev_model.cpp/h`)**: orchestration core.
  - LLM context (ctx) ownership and lifecycle;
  - `kev_encode.cpp` assembles requests (`encode` → `KevEnc` → `KevRows`);
  - `kev_prefix.cpp` state LRU prefix cache (same-state second-request hit);
  - `kev_head.cpp` `PointerHead` readout head (`d=1024 → dp=256`, `k(h_opts)·q(h_decide)*scale`);
  - per-question fork decode / batch (`rows_from_slot` / `decode_batch`);
  - `compute_mutex_` serializes all compute sections over the shared ctx (multi-stream safe).
- **(c) vendored llama.cpp / GGML backend**: hybrid GDN+attention graph, Q8_0 kernels, KV/recurrent cache.
- **(d) pluggable `kev_backend*` C-ABI abstraction** (ARCHIVED, branch `build-portable-backends` @ `71d2c64`):
  CPU and future GPU backends loadable via `--backend` as DLLs; `KevModel` only calls llama/ggml ops through `kev::Backend*`.

### 7.3 Data flow

```
parse (/v1/systemone JSON)
   → encode (assemble state + branch as a token sequence)
   → ensure_state (LRU: hit skips state prefill / miss prefills into a slot)
   → rows_from_slot (branch decode, read hidden at decide/option positions)
   → PointerHead (kev_head.cpp)
   → probabilities (softmax z/T)
```

---

## 8. Model / HuggingFace Hub

> **Important**: the 0.8B model used here is **not an off-the-shelf Qwen-0.8B**, but a **modified / customized 0.8B variant
> owned by this project**.

### 8.1 Model spec (verified)

- **Structure**: 24 mixed layers = **18 Gated-DeltaNet (recurrent) + 6 full-attention (layers 3, 7, 11, 15, 19, 23)**.
- `n_embd = 1024`
- FFN: `1024 → 3584` (`ffn_up/gate [1024,3584]`, `ffn_down [3584,1024]`)
- `ssm_d_inner = 2048` (`ssm_out [2048,1024]`)
- `ssm_d_state = 128`, `ssm_n_group = 16`
- `vocab = 248320`, `token_embd` and `output` **tied/shared** (lm_head tied, `output_norm` independent F32)
- Weights exported as **GGUF (Q8_0, ≈ 774 MiB)**: `models/kev-merged-q8.gguf`
- Plus a separate **PointerHead**: `models/head.bin` (≈ 2 MiB)

### 8.2 HF Hub (to be uploaded)

This project's models upload to the **Hugging Face Hub** (huggingface.co).

- Repository ID / link:
  `https://huggingface.co/vanncoo/kevcpp-qwen3.5-0.8b`.
- The upload will include: **GGUF weights** (`kev-merged-q8.gguf`), **PointerHead** (`head.bin`), and the necessary model card.

---

## License / Notes

This project uses the **MIT License** (see `LICENSE`), which is the **current effective license** of kevcpp.

> kevcpp references and reuses the vendored llama.cpp (`third_party/llama.cpp`) and the behavioral semantics of the
> reference implementation kev (`https://github.com/jaredpalmer/kev`); llama.cpp's and kev's respective licenses follow
> their upstreams; the MIT license covers only this project's own source organization.