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
  ggml-cpu-{x64,sse42,sandybridge,haswell,skylakex,cannonlake,
            alderlake,cascadelake,icelake}.dll   # full set is safe: runtime KAT gates the VNNI variants
```

At backend init (`op_init`, before `llama_backend_init()`) `kev_backend_cpu.cpp`
enumerates `ggml-cpu-*.dll` next to the exe, reads each variant's cpuid score,
and probes candidates **from the highest score down in a fresh subprocess**
(`<exe> --kev-probe-cpu-variant <dll>`, handled at the top of `main()` in
`main_server.cpp`). Each probe loads the variant, runs a small known-answer test
(F32 GEMM + Q8_0×F32 GEMM, K=256 to satisfy the tiled-GEMM `ne00 % 256`
constraint) and prints `KAT_BEGIN..KAT_END` metrics; the parent compares them
against the scalar `x64` reference (rel ≤1e-3 OR abs ≤1e-3 per metric). The
first candidate that matches is pre-registered via `ggml_backend_load()`, so
`llama_backend_init()`'s `ggml_backend_load_all()` (which only runs when the
registry is empty) never loads the broken variants. Environment overrides:
`KEV_CPU_VARIANT=auto` (default) | `<tag>` (force one, still KAT-gated) |
`off` (skip selection entirely, ggml default `load_best` behaviour).

Verified on Zen 5: cascadelake/icelake are auto-rejected (their Q8_0×F32 KAT
c2 fingerprint is ~2.7× off: 1834.6 vs 674.5), `cannonlake` is selected, and
the 231-question 0.8B benchmark matches the native control build exactly
(121/231, 0 per-task choice differences, P50 parity). The old
"prune the VNNI dlls from the artifact set" rule (§4) is therefore superseded
— pruning is still harmless, but no longer required.

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

**Rule (superseded by the runtime KAT gate — kept for history):**
> ~~Ship ONLY `ggml-cpu-{x64,sse42,sandybridge,haswell,skylakex,cannonlake}.dll`.
> PRUNE the VNNI variants.~~ As of branch `perf/cpu-runtime-kat` the pruning is
> enforced **at runtime** instead (see §1): the VNNI variants fail the startup
> KAT against the `x64` reference and are never registered, so shipping the
> full variant set is safe. The underlying failure is real and unchanged: on
> this model the `vpdpbusd` Q8×Q8 kernel (AVX512_VNNI/AVX_VNNI) serves wrong
> probabilities (matches `docs/F.24_Q8_DOTBUS_FAIL_AND_F16_PIVOT.md`), and an
> ungated auto-selector on an AVX512-VNNI host (Zen5, many recent Intel) picks
> `cascadelake` first.

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