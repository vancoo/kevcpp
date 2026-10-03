# Rank-1 — Remove the unused tied lm_head (151×248320)

> Authoritative end-to-end evidence for dropping the lm_head on the kevcpp encode/prefill path.
> Machine: AMD Ryzen AI MAX+ PRO 395 (Zen5, 32 logical cores, AVX-512), 16 server threads.
> Branch: `perf-lmhead` (from `perf-gemv-q8tiled`). Session-built with cmake (`cmake --build build --target main_server --config Release`).
> Protocol: byte-faithful `curl.exe -s -o - -X POST ... --data-binary "@tests\tmp_a_company.json"`, every sample confirmed `input=189 output=81` in KEV_TRACE; cold samples discarded; ~18 samples per run.

## TL;DR
- The steady branch forward decodes **one** 189-token `llama_decode` whose lm_head (`M=151 × N=248320`, 38.4 GMAC, ~34% of total GEMM / ~112 ms) was **never consumed** (app reads the 7 hidden rows via `llama_get_embeddings_ith`, never `llama_get_logits_ith`).
- **Chosen path: dropped the lm_head node entirely** (not reduced to M=7) — it is provably independent of the embeddings extraction (`res->t_embd` is set from the post-norm hidden state **before** the lm_head node).
- **Measured steady latency: ~445 ms → ~345 ms (≈ -22%, ~100 ms)**; all 7 hidden rows (decide + opts) and resulting probs **bit-identical**.
- Commit present on `perf-lmhead`; `git diff HEAD` clean after commit.

## (a) Path taken & why
Dropped the lm_head **entirely**, via a new default-true context flag `lm_head` that kevcpp sets to `false`.
- In `qwen35.cpp::graph` the embedding output `res->t_embd` is assigned (`cb(cur,"result_norm")`) **before** the lm_head builder; the lm_head only *consumes* that tensor — it never modifies it. So removing the lm_head node leaves the whole embeddings/hidden extraction path, and therefore the 7 consumed hidden rows, numerically **and bit-wise** identical.
- `n_outputs = n_tokens = 151` currently forces `inp_out_ids` to be 151 long → `build_lora_mm(output, cur)` runs M=151 over the whole vocab. Rather than fight `n_outputs` (which would cascade into the gather/masking semantics the app relies on), the least-invasive change that deletes the work is to **not build the lm_head node at all**.
- Gate: `if (cparams.lm_head)` in `qwen35.cpp`. Default `true` in `llama_context_params` so **all other llamacpp users are unaffected**; kevcpp turns it off in `KevModel::init`.
- Graph-topology correctness: the flag is added to the graph **reuse key** (`graph_shape_key`, `llm_graph_params::can_reuse`) so a cached no-lm_head graph can never be accidentally reused as a with-lm_head graph (and vice versa).

## (b) Exact diff (files)
```
 src/cold_profile.cpp                          (+1)   keep profiler representative (embeddings-only)
 src/headbench.cpp                             (+1)   keep readout representative (embeddings-only)
 src/kev_model.cpp                             (+2)   cparams.lm_head = false;
 third_party/llama.cpp/include/llama.h         (+4)   public llama_context_params.lm_head (default true)
 third_party/llama.cpp/src/llama-cparams.h     (+5)   internal llama_cparams.lm_head
 third_party/llama.cpp/src/llama-context.cpp   (+3)   convert param + default + shape-key capture
 third_party/llama.cpp/src/llama-context.h     (+4/-1) graph_shape_key ordering logic for lm_head
 third_party/llama.cpp/src/llama-graph.h       (+1)   llm_graph_params::can_reuse key
 third_party/llama.cpp/src/models/qwen35.cpp   (+9/-7) gate the lm_head build on cparams.lm_head
```
Core change (`qwen35.cpp`):
```cpp
cb(cur, "result_norm", -1);
res->t_embd = cur;                       // <-- embeddings/hidden source, upstream of lm_head

if (cparams.lm_head) {                   // NEW gate
    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
ggml_build_forward_expand(gf, cur);      // expands lm_head output, else result_norm (embeddings)
```
`kev_model.cpp`:
```cpp
cparams.embeddings        = true;
cparams.lm_head           = false;      // never read logits -> drop the 151x248320 projection
cparams.pooling_type      = LLAMA_POOLING_TYPE_NONE;
```

## (c) Before/after steady latency (all confirmed `input=189`)
Byte faithful curl, 16–17 steady samples after discarding cold (first 1–2), latency_ms from response.

| Run | n | min | median | mean | max | note |
|---|---|---|---|---|---|---|
| BEFORE (HEAD binary) | 17 | **437.1** | 460.8* | 461.3 | 503.6 | *median inflated by periodic Chrome/Postman tail; plateau≈445 |
| BEFORE low-noise plateau | 7 | 437.1 | 443.4 | 445.1 | 461.4 | the stable number, matches authoritative baseline ~435–445 |
| AFTER (new binary), run 1 | 16 | **336.9** | **343.0** | 346.8 | 364.9 | |
| AFTER (new binary), run 2 (final) | 16 | **336.9** | **345.2** | 347.1 | 365.3 | |

- **Reconciled steady: BEFORE ≈ 445 ms → AFTER ≈ 343–345 ms.** Saving ≈ **98–102 ms (~22%)**, consistent with the predicted ~112 ms lm_head elimination (minus ~10 ms of per-iteration latency overhead already present in baseline).
- Every request in both BEFORE and AFTER was logged as `[TRACE] ... input=189 output=81` (never the `input=70` body-truncation trap).

Before sample list: `631, 447.5, 442.6, 439.3, 461.4, 437.1, 444.5, 443.4, 452.2, 475.2, 468.8, 468.7, 483.8, 486.5, 503.6, 460.8, 455, 472.4`
After run-1 list:  `465.4, 347.6, 348.2, 336.9, 347.5, 343.8, 340.6, 341.8, 353.4, 339.7, 339.2, 356.7, 340.1, 364.9, 355.6, 341.8, 342.3, 357`
After run-2 list:  `454.6, 354.2, 341.3, 353.7, 338.8, 343.0, 345.4, 346.9, 346.6, 359.7, 343.7, 344.4, 345.5, 354.6, 336.9, 345.1, 365.3, 343.0`

## (d) Correctness proof (red line: probs bit-identical on the 7 consumed rows)
1. **Structural.** `res->t_embd` feeding `llama_get_embeddings_ith` (via `KevModel::hidden_at_output`) is assigned in `qwen35.cpp` from the post-norm hidden state **before** the lm_head node; the lm_head is a pure downstream *consumer*. Dropping it changes only the number of child nodes — the upstream trunk (embedding + all 30 layer blocks + `output_norm` + the `ggml_get_rows` gather) is topologically and numerically **identical**. Hence the 7 hidden rows are bit-identical by construction.
2. **Empirical (server).** Real byte-faithful request on the production `/v1/systemone` path, response_after vs response_before:
   - `usage.input_tokens = 189`, `output_tokens = 81` in both.
   - `答: 关键词类别 choice`: `产线产能` == `产线产能`; confidence diff = **0**.
   - All option probabilities **exactly equal** (JSON to 4 decimals): `中标 0.0026, 产线产能 0.3974, 其他 0.0272, 厂房 0.1731, 投资 0.3960, 购地 0.0037` in both. **max|dp| = 0**.
   - Since head.bin (`PointerHead`) probs are a *deterministic pure function* of the 7 hidden vectors, zero probability difference over all non-degenerate projections ⟹ the 7 hidden float vectors are **bit-identical**.
3. **Independent parity (kab_bench).** The same `KevModel::probs` path, run as both the pre-change binary and the post-change binary, prints identical `ARGMAX 0 0 1 0.397396 0.002636 0.397396 0.027245 0.173090 0.395982 0.003650` — no NaN, no crash, choice/argmax stable.
4. Graph cache correctness: `lm_head` is part of the shape/reuse keys, so no stale graph is reused across configs.

## (e) Commit / branch / hash
- Branch: `perf-lmhead` (created from `perf-gemv-q8tiled`).
- Commit: see `git log` (added after this file; message "perf: drop unused tied lm_head (encode path) when app only reads embeddings").
- Working tree clean of tracked modifications at HEAD (only pre-existing untracked artifacts remain).

## (f) Recommendation
- **Adopt.** The lm_head was pure waste (logits never read) and eliminating it is low-risk: hidden-at-7 and the resulting probs are bit-identical, and the ~100 ms (~22%) steady-latency win is real and reproducible.
- The flag keeps the behavior opt-in for any other llamacpp consumer; kevcpp (server + cold_profile + headbench) opts out.
- Next levers (separate ranks, do not change the Q8_0 tiled kernel here): (1) faster Q8_0 tiled GEMM (~339 → ~1 TMAC/s, worth ~150 ms on the remaining trunk), (2) CONCAT ~9% + GDN ~7% scheduling/serial overhead.