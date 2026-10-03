# Portable backend deployment — kevcpp pluggable backend + GGML_CPU_ALL_VARIANTS

> Status: **ARCHIVED DESIGN + SOURCE.** The pluggable-backend source is committed on
> branch `build-portable-backends`. This tree keeps the safe, single-module, static
> llama build (bit-identical to baseline `dd51ed7`) with the backend abstraction
> wired in-process. The full **shared-lib + portable-SIMD-variant** build and its
> **final load validation under a debugger** must be done on a **stable, adequately
> provisioned host** — see §7. This host (AMD Zen5, no debugger) intermittently
> heap-corrupts / fast-fails under sustained load even on the unmodified baseline,
> so LOAD-LEVEL VALIDATION WAS NOT PERFORMED HERE (see `logs/build-portable-backends_diagnosis.md`).

---

## 1. TL;DR — deployable artifact set

For a **portable CPU** deployment the artifacts shipped are:

```
bin/
  main_server.exe            # host executable (this tree: statically links llama — single-llama SAFE state)
  kev-backend-cpu.dll        # CPU backend plugin — the portable/dll build (NOT built in this tree; see §6/§7)
  ggml-cpu-{x64,
          sse42,
          sandybridge,
          haswell,
          skylakex,          # <-- keep (AVX512, non-VNNI, bit-identical)
          cannonlake}.dll    # <-- keep (AVX512+VBMI, non-VNNI, bit-identical)
  # --- MUST BE PRUNED (numerically WRONG) ---
  # ggml-cpu-cascadelake.dll # AVX512_VNNI vpdpbusd  -> WRONG probabilities
  # ggml-cpu-icelake.dll     # AVX512_VNNI vpdpbusd  -> WRONG probabilities
  # ggml-cpu-alderlake.dll   # AVX_VNNI    vpdpbusd  -> WRONG probabilities
```

The runtime `ggml-backend-load-best("cpu")` auto-selects the highest-scoring variant
for the host CPU. If the VNNI variants are present, a Zen5 host auto-selects
`ggml-cpu-cascadelake.dll` and **serves wrong probabilities**. **Pruning the VNNI
variants is product-critical** (§4).

---

## 2. What was changed on this branch

| Area | Files | Notes |
|------|-------|-------|
| Backend C-ABI (llama-free) | `src/kev_backend.h` | `kev_backend_meta/params/ops`, factory `kev_backend_get_meta/get_ops/create/destroy`. NO llama types. Layout-stable ops table (append, never reorder; ABI versioned). |
| Backend RAII + factory | `src/kev_backend_factory.{h,cpp}` | `kev::Backend` wrapper; `kev::BackendFactory_load(name)` → LoadLibrary/dlopen in dll mode, or the in-process CPU impl in the safe build. name→dll map: `cpu`→`kev-backend-cpu`; `gpu/cuda/vulkan`→`kev-backend-gpu` (extension point → clear "not built / use --backend cpu" error); unknown→clear error. |
| CPU backend | `src/kev_backend_cpu.cpp` | Implements the C-ABI over llama/ggml (init, decode_seq/decode_batch, read_hidden, seq_clear/seq_copy, n_ctx_seq_capacity, count_tokens, model_n_embd, threads, vocab_opaque). Built as `kev-backend-cpu.dll` in the dll build, or statically into host targets in the safe build. **Same llama underneath → bit-identical decode.** |
| KevModel refactor | `src/kev_model.{h,cpp}` | All orchestration untouched (LRU, encode assembly, rows planning, PointerHead, `compute_mutex_`, n_seq layout). llama/ggml **operator coupling** (decode_seq/batch, read_hidden, seq_clear/seq_copy, sizing, vocab handle) now goes through `kev::Backend*`; no direct `llama_*` compute calls in KevModel. |
| Server wiring | `src/main_server.cpp` | `--backend <name>` (default `cpu`); backend reported in startup log + `/v1/models` (`backend_name/device/version`). `--backend gpu/cuda/vulkan` → clear not-built error. |
| CMake | `CMakeLists.txt` | Backend sources added to all targets that compile `kev_model.cpp` (`main_server`, m3/m5/m7/m8, kab_bench, lora_ab) with `KEV_BACKEND_BUILD`; gated `KEV_BUILD_BACKEND_DLL=OFF` `kev-backend-cpu` shared target pinned. Top-of-file documents the reconciliation (§6). |
| Doc | `docs/portable_backend_deployment.md` | This file. |

Vendored `third_party/llama.cpp` is **untouched**.

---

## 3. The two-layer architecture

```
 ┌─────────────────────────────── kevcpp (orchestration) ──────────────────────────────┐
 │  KevModel: LRU · encode assembly · rows planning · PointerHead · compute_mutex_ · │
 │            n_seq layout  — calls only kev::Backend* for llama/ggml operators        │
 │  main_server: --backend <name> -> kev::BackendFactory_load(name)                   │
 └───────────────┬──────────────────────────────────────────────────────────────────────┘
                 │ C ABI (kev_backend.h: meta/params/ops, no llama types)
                 ▼
 ┌─────────────── kev backend plugin layer ────────────────────────────────┐
 │  kev-backend-cpu.dll   — links llama/ggml   (only module that does)     │
 │  kev-backend-gpu.dll   — extension point (NOT yet built; future CUDA/Vulkan) │
 └─────────────────────────────────────────────────────────────────────────────┘
```

KevModel keeps *all* orchestration. The backend exposes only the operator surface
(init/decode/read_hidden/seq-memory/sizing/vocab-handle) so a GPU backend can later
drop in without touching KevModel or the encode/rows logic.

---

## 4. GGML_CPU_ALL_VARIANTS — CPU variant matrix and the VNNI pruning rule

On MSVC, `GGML_CPU_ALL_VARIANTS` emits
`ggml-cpu-{x64,sse42,sandybridge,haswell,skylakex,cannonlake,cascadelake,icelake,alderlake}.dll`
(compiler-gated off on MSVC: `piledriver/zen4/cooperlake/sapphirerapids`). The loader
scores by host CPU features and picks the highest.

Verified per-variant numerics on this model (`models/kev-merged-q8.gguf`), short/long confidence:

| Variant dll | CPU features | short conf | == baseline? |
|-------------|--------------|-----------|--------------|
| `skylakex`  | AVX512 (non-VNNI) | 0.2769 | ✔ EXACT |
| `cannonlake`| AVX512+VBMI (non-VNNI) | 0.2769 | ✔ EXACT |
| `x64`/`sse42` | scalar/SSE | 0.2885 | ✘ diff |
| `haswell`/`alderlake` | AVX2 (Alder also AVX_VNNI) | 0.2811 | ✘ diff |
| `sandybridge` | AVX | 0.2828 | ✘ diff |
| `cascadelake`/`icelake` | **AVX512_VNNI (vpdpbusd)** | **0.1222** | ✘ **WRONG** |

**Product-critical rule:**
> Ship ONLY `ggml-cpu-{x64,sse42,sandybridge,haswell,skylakex,cannonlake}.dll`.
> **PRUNE `ggml-cpu-{cascadelake,icelake,alderlake}.dll`** (anything using the
> AVX512_VNNI/AVX_VNNI `vpdpbusd` Q8×Q8 kernel). If left present, the auto-selector
> on an AVX512-VNNI host (Zen5, many recent Intel) picks `cascadelake` and serves
> wrong probabilities (matches `docs/F.24_Q8_DOTBUS_FAIL_AND_F16_PIVOT.md`).

This is an **architectural** property of the Q8×Q8 dpbusd path on this model, not a
per-host tuning knob — prune unconditionally in the artifact set.

---

## 5. `--backend` usage (incl. GPU extension point)

```
# CPU (default, built):
main_server --model models/kev-merged-q8.gguf --head models/head.bin --backend cpu

# GPU — currently a clear "not built" error (extension point, future kev-backend-gpu.dll):
main_server ... --backend gpu      # -> "backend 'gpu' is NOT built ... Use --backend cpu"
main_server ... --backend cuda     # same
main_server ... --backend vulkan    # same

# Unknown name:
main_server ... --backend foo      # -> "unknown backend: 'foo'"
```

Startup log reports the loaded backend; `/v1/models` adds `backend_name/device/version`.

GPU wiring is confined to `BackendFactory_load()`. Adding a real GPU backend = ship a
`kev-backend-gpu.dll` implementing `kev_backend.h` and register the name→dll map.

---

## 6. The "single-llama rule" and why the plugin build needs a stable box

`llama`/`ggml` (and therefore `ggml_backend`'s CUDA/CPU scheduler state and any
statics) must live in **exactly ONE module per process**.

- **Safe (committed) state:** `main_server.exe` and the in-process CPU backend share
  the same static `llama` — one copy. Bit-identical to baseline `dd51ed7`. This is what
  this branch builds and validates.
- **Portable/dll state (NOT valid here):** `main_server.exe` must **NOT** link llama;
  it loads `kev-backend-cpu.dll` via the C ABI and the dll is the only llama link. That
  requires routing *encode/tokenize* through the backend too (so the vocab cast
  `const llama_vocab*` also lives only in the dll), which changes `kev_encode.h`'s
  `const llama_vocab*` signatures — a blast radius over every bench target. **The
  previous attempt that statically linked llama into BOTH `main_server` and a
  `kev-backend-cpu.dll` reintroduced the fast-fail crash (two llama copies) and was
  reverted** (see diagnosis).

**Bottom line:** the plugin build must be completed on a stable box with a debugger so
the two-copy risk can be demonstrably eliminated, and only then should `main_server`
stop linking llama.

---

## 7. Exact reconciliation needed on a stable host (finish the portable build)

On a stable, well-provisioned host with a debugger (windbg/cdb), from branch
`build-portable-backends`:

1. **Flip top-level CMake** (in `CMakeLists.txt`) — replace the safe-static defaults:
   ```cmake
   set(GGML_NATIVE OFF FORCE)          # REQUIRED: conflicts with GGML_BACKEND_DL
   set(BUILD_SHARED_LIBS ON)           # llama/ggml become shared; ONE module
   set(GGML_BACKEND_DL ON)             # plugin dll auto-loader
   set(GGML_CPU_ALL_VARIANTS ON)       # full portable SIMD dll set
   # keep GGML_CPU_REPACK OFF, GGML_LLAMAFILE OFF (already set)
   ```
   This is the only true "product" build; this branch intentionally keeps the static
   single-module tree so it stays cleanly compilable and doesn't destabilize the other
   bench targets (Condition 4).
2. **Build `main_server`** and verify `/v1/models` + a short `POST /v1/systemone` still
   match the baseline probs exactly (short conf **0.2769**, long **0.3691**) — it must,
   since it's the same decode over the same llama.
3. **Prune the VNNI dlls** from the shipped artifact set (§4) before serving.
4. **Route encode/tokenize through the backend** to satisfy the single-llama rule
   (change `kev_encode.h`'s `const llama_vocab*` params to the backend's opaque vocab
   handle), then stop linking llama into `main_server`. Validate bit-identity again.
5. **Final persistent-load validation under a debugger:** exercise the exact production
   load (repeated new-connection `POST /v1/systemone`) that corrupts the heap on this
   host; capture the faulting op with windbg. Root-cause the baseline fast-fail (suspected
   memory pressure) BEFORE trusting the shared/plugin build at load.

### Unbuilt / unvalidated on this host (explicitly)
- The **shared-lib + ALL_VARIANTS build** (steps 1–2) — not built here.
- The **`kev-backend-cpu.dll` plugin-load path** (`KEV_BUILD_BACKEND_DLL=ON`) — target
  is pinned in CMake but OFF; enabling it requires the shared-llama reconciliation and
  the single-llama rule, so it was intentionally not toggled.
- **GPU backend** — extension point only, not implemented.
- **Persistent-load stability** — this host fast-fails under load even on baseline; no
  load test was run here (only a short `--backend cpu` startup + one `/v1/models` and the
  `--backend gpu` error path were smoke-checked).

### Validated on this host
- Full kevcpp tree **compiles cleanly** in the safe single-module static state (all
  targets: kev, headbench, cold_profile, merge_lora, m3/m5/m7/m8/kab/lora_ab benches,
  main_server).
- `--backend cpu` server starts, reports `backend 'cpu' (device=cpu, ver=kev-backend-cpu-0.1-static-single-llama)`, serves `/v1/models` (HTTP 200).
- `--backend gpu/cuda/vulkan` → clear "not built / use --backend cpu" error.
- Bit-identical static baseline numerics (0.2769 / 0.3691) are preserved by construction
  (same llama underneath); not re-measured under load here.