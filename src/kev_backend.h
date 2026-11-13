// kev_backend.h — kevcpp pluggable backend C-ABI (llama-free)
//
// Archival note:
//   This header defines the stable kevcpp-owned abstraction layer between
//   KevModel (orchestration) and a concrete inference backend (cpu/gpu/...).
//   It is DELIBERATELY llama-free: it must not include "llama.h" or any other
//   project header. A backend plugin (e.g. kev-backend-cpu.dll) implements this
//   C ABI and is loaded dynamically by kev::BackendFactory.
//
//   llama-free means the ABI only exchanges plain C primitives (ints, size_t,
//   char*, opaque void* handle). The only "opaque" thing we pass out is the
//   vocabulary handle (vocab_opaque) which a consumer that links llama may cast
//   to `const llama_vocab*` for encode/tokenize; the backend itself never leaks
//   llama types through this header. This is what lets the backend dll be the
//   ONLY module in the process that statically links llama (the "single-llama
//   rule" — see docs/portable_backend_deployment.md).
//
//   The ops table is a fixed function-pointer struct so the ABI is layout-stable
//   and version-able: append new ops at the END and bump `struct_size`.
#pragma once

#include <cstddef>
#include <cstdint>

#ifdef _WIN32
#  ifdef KEV_BACKEND_BUILD           // defined when compiling the plugin dll
#    define KEV_BACKEND_API __declspec(dllexport)
#  else
#    define KEV_BACKEND_API __declspec(dllimport)
#  endif
#else
#  define KEV_BACKEND_API __attribute__((visibility("default")))
#endif

// Device identifiers exposed by kev_backend_meta.device.
enum kev_backend_device {
    KEV_BACKEND_DEVICE_UNKNOWN = 0,
    KEV_BACKEND_DEVICE_CPU = 1,
    KEV_BACKEND_DEVICE_GPU = 2,
};

// Capability bit flags reported in kev_backend_meta.capabilities.
enum kev_backend_capability {
    KEV_BACKEND_CAP_NONE        = 0u,
    KEV_BACKEND_CAP_BATCH_DECODE = 1u << 0,  // decode_batch is implemented
    KEV_BACKEND_CAP_SEQ_MEMORY   = 1u << 1,  // seq_clear / seq_copy implemented
    KEV_BACKEND_CAP_HIDDEN_READ  = 1u << 2,  // read_hidden implemented
    KEV_BACKEND_CAP_LOGITS_READ  = 1u << 3,  // read_logits implemented
};

// Version of the kev_backend_ops layout.
#define KEV_BACKEND_ABI_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

// Static metadata describing a backend plugin. Filled by kev_backend_get_meta().
typedef struct kev_backend_meta {
    uint32_t abi_version;                   // == KEV_BACKEND_ABI_VERSION
    const char * name;                      // e.g. "cpu", "gpu"
    const char * device;                    // "cpu" | "gpu" | ...
    const char * version;                   // free-form build version string
    uint32_t capabilities;                  // bitwise OR of kev_backend_capability
    uint32_t struct_size;                   // sizeof(kev_backend_meta) at build time
} kev_backend_meta;

// Parameters given to the backend at create/init time.
typedef struct kev_backend_params {
    uint32_t struct_size;                   // = sizeof(kev_backend_params)
    const char * model_path;                // GGUF model
    const char * lora_path;                 // optional LoRA adapter GGUF ("" = none)
    float       lora_scale;                 // LoRA scale (ignored if lora_path empty)
    bool        lora_premerged;             // true = lora already folded into weights (no adapter).
    int         n_ctx;                      // context size
    int         n_seq_max;                  // n_lru_capacity + 1 (slots) + kMaxBatch + 1 (scratch)
    int         n_threads;                  // compute threads (0 = backend auto)
    int         n_batch;                    // llama n_batch (default 512)
    int         n_ubatch;                   // llama n_ubatch (default 256)
    int         n_outputs_max_per_seq;      // llama n_outputs_max_per_seq (default 128)
    int         embeddings;                 // 1 = read hidden states
    int         lm_head;                    // 0 = skip lm_head projection (hidden-only serving)
    // ---- MoE 专家 offload(追加在结构体末尾, 不破坏 ABI; 旧后端忽略新字段) ----
    // n_gpu_layers: 要 offload 到 GPU 的层数。负数 = 全部层到 GPU(与 llama.cpp -ngl -1 一致);
    //   0 = 全部留 CPU。供 CPU/Vulkan/GPU 后端统一透传给 llama_model_params.n_gpu_layers。
    int         n_gpu_layers = 0;
    // moe_offload: 0 = MoE 专家权重跟随 n_gpu_layers(默认全部进 VRAM); 1 = MoE 专家张量
    //   (*_exps.weight 等专家张量)强制留在系统内存(host RAM), 其余(attention/shared/KV)仍按
    //   n_gpu_layers 进 VRAM。用于"小显存显卡: 模型放内存, 专家按需转显存"以降低显存占用。
    //   仅在 GPU/Vulkan 后端、且 n_gpu_layers != 0 时有实际效果(CPU 后端本就全在内存)。
    int         moe_offload  = 0;
} kev_backend_params;

// Opaque handle for a live backend instance (one model+context).
typedef struct kev_backend kev_backend_t;

// Opaque handle to the backend's tokenizer/vocab (castable to const llama_vocab*
// only by a consumer that itself links llama — single-llama rule).
typedef struct kev_backend_vocab kev_backend_vocab_t;

// A single decode request for decode_seq (one sequence continuation).
typedef struct kev_backend_decode_seq {
    const int32_t * tokens;                 // token ids to append
    int32_t         n_tokens;
    int32_t         start_pos;              // position offset where the sequence continues
    int32_t         seq;                    // target sequence id
    const int32_t * logits_pos;             // batch indices needing hidden output (or NULL)
    int32_t         n_logits_pos;
} kev_backend_decode_seq;

// A single branch for decode_batch (all branches share a prefix and are equal length).
typedef struct kev_backend_decode_branch {
    const int32_t * ids;                    // branch token ids (length n_ids)
    int32_t         n_ids;
    int32_t         start_pos;              // = state length (Ls)
    int32_t         seq;                    // branch seq id
    int32_t         decide;                 // branch-local index producing the decide hidden
    const int32_t * opts;                   // branch-local opt indices (hidden outputs)
    int32_t         n_opts;
} kev_backend_decode_branch;

// The plugin's exported ops. Layout-stable; append, never reorder.
typedef struct kev_backend_ops {
    uint32_t struct_size;                   // = sizeof(kev_backend_ops)

    // lifecycle
    int         (*create)(kev_backend_t ** out);
    void        (*destroy)(kev_backend_t * b);

    // init the model + context. Returns 0 on success, non-zero on failure.
    int         (*init)(kev_backend_t * b, const kev_backend_params * p,
                        char * err, size_t err_cap);

    // decode a single sequence continuation into seq. logits_pos selects hidden outputs.
    int         (*decode_seq)(kev_backend_t * b, const kev_backend_decode_seq * d,
                              char * err, size_t err_cap);

    // decode several equal-length branches sharing a prefix in one call.
    int         (*decode_batch)(kev_backend_t * b, const kev_backend_decode_branch * rows,
                                int32_t n_rows, char * err, size_t err_cap);

    // copy the i-th hidden output of the most recent decode into out (length = model_n_embd).
    int         (*read_hidden)(kev_backend_t * b, int32_t out_idx, float * out,
                               size_t out_cap, char * err, size_t err_cap);

    // sequence memory: erase a seq's cached KV, or copy src's KV into dst.
    int         (*seq_clear)(kev_backend_t * b, int32_t seq);
    int         (*seq_copy)(kev_backend_t * b, int32_t src, int32_t dst);

    // Read the logit the model assigns to `token_id` at the most recent decode's
    // out_idx-th output position. `out_idx` uses the SAME index semantics as
    // read_hidden (the llama_batch row-major index / llama output row for that
    // position). Returns 0 on success and writes *out; non-zero on error (e.g.
    // the position was not marked for output, lm_head / logits disabled, or
    // token_id out of vocab range).
    // NOTE: requires the backend context to have lm_head enabled (see
    // kev_backend_params.lm_head) so the lm_head projection is computed.
    int         (*read_logits)(kev_backend_t * b, int32_t out_idx, int32_t token_id,
                               float * out);
    // Number of tokens (vocab size) of the loaded model's vocabulary.
    int32_t     (*vocab_n_tokens)(kev_backend_t * b);

    // capacity / sizing
    int32_t     (*n_ctx_seq_capacity)(kev_backend_t * b);
    int32_t     (*count_tokens)(kev_backend_t * b, const char * text);  // GGUF tokenizer
    int32_t     (*model_n_embd)(kev_backend_t * b);
    int32_t     (*threads)(kev_backend_t * b);
    int32_t     (*threads_batch)(kev_backend_t * b);

    // the backend's tokenizer/vocab handle (opaque; cast by llama-linking consumer).
    kev_backend_vocab_t * (*vocab_opaque)(kev_backend_t * b);

    // ---- delegated tokenizer (B-1) ----
    // The backend OWNS the llama_vocab (single-llama rule), so a llama-free host
    // must delegate tokenize/special-token lookup through ops instead of casting
    // vocab_opaque() to const llama_vocab*.  These are appended at the END of the
    // table (never reorder earlier fields); struct_size is sizeof() so consumers
    // detect the bigger table and can still use the older prefix if they wish.
    //
    // tokenize: tokenizes `text[0..text_len)` with add_special/parse_special
    // semantics matched to llama_tokenize.  Fills at most `cap` ids into out_ids.
    // Returns the number of ids written (>=0); on buffer-too-small returns the
    // required capacity as a NEGATIVE value (-needed) and writes nothing.
    // text_len is byte length (use < 0 to mean "strlen").
    int32_t     (*tokenize)(kev_backend_t * b, const char * text, int32_t text_len,
                            bool add_special, bool parse_special,
                            int32_t * out_ids, int32_t cap,
                            char * err, size_t err_cap);
    // special_token_id: id of the "<|name|>" special token (parse_special=true),
    // or -1 if not found.
    int32_t     (*special_token_id)(kev_backend_t * b, const char * special);

    // n_batch: the backend context's logical max tokens per llama_decode call
    // (its cparams.n_batch, default 512). KevModel uses this to size batch groups
    // and to chunk long sequences so a single decode never exceeds n_batch (which
    // would trigger llama.cpp's n_tokens_all <= n_batch assert). Appended at the
    // END (never reorder); a backend that predates this field returns 0 and the
    // host falls back to a conservative capacity.
    int32_t     (*n_batch)(kev_backend_t * b);

    // ---- B1: single-ctx multi-sequence batched decode ----
    // decode N independent sequences (each its own KV slot) in ONE llama_decode
    // call. This is the core primitive for sequence-level parallelism: concurrent
    // requests each occupy a distinct seq id and are decoded together, so the
    // shared llama_context amortizes the graph/prefill across the batch.
    //
    //   * tokens    = concatenation of all seqs' token ids, seq-major
    //                 (seq s occupies rows [base_s, base_s + n_tokens[s])).
    //   * n_tokens  = [n_seq] token count per seq.
    //   * start_pos = [n_seq] position offset where each seq continues (its own
    //                 M-RoPE position accounting; independent per seq).
    //   * seq       = [n_seq] target seq id (each must be distinct, < n_seq_max).
    //   * out_seq/out_pos = per-requested-output: which seq and which local token
    //                 position within that seq's tokens should produce logits.
    //   * out_idx   = [n_out] filled by the backend: for each requested output, the
    //                 ordinal index into llama_get_logits_ith(ctx, i) (0-based,
    //                 batch-row major among all marked logits). The host passes the
    //                 returned out_idx[] straight into read_logits(out_idx, token, &lt).
    //
    // Returns 0 on success, non-zero on error. Appended at the END (never reorder);
    // a backend that predates this feature leaves it NULL and the host falls back
    // to per-sequence decode_seq (serial).
    int         (*decode_multi_seq)(kev_backend_t * b,
                                    const int32_t * tokens,
                                    const int32_t * n_tokens,
                                    const int32_t * start_pos,
                                    const int32_t * seq,
                                    int32_t n_seq,
                                    const int32_t * out_seq,
                                    const int32_t * out_pos,
                                    int32_t n_out,
                                    int32_t * out_idx, int32_t out_idx_cap,
                                    char * err, size_t err_cap);
} kev_backend_ops;

// Factory entry points every backend dll must export.
KEV_BACKEND_API kev_backend_meta     kev_backend_get_meta(void);
KEV_BACKEND_API kev_backend_ops      kev_backend_get_ops(void);
KEV_BACKEND_API kev_backend_t *      kev_backend_create(void);
KEV_BACKEND_API void                 kev_backend_destroy(kev_backend_t * b);

// Optional extension, resolved independently of the by-value ops table. Data
// belongs to the currently loaded model. A NULL BOS/EOS token yields "".
// `needed` receives the size including the NUL; a NULL or undersized buffer
// returns 2, success returns 0, and other errors return 1 with an error message.
enum kev_backend_chat_data_kind {
    KEV_BACKEND_CHAT_TEMPLATE = 0,
    KEV_BACKEND_CHAT_BOS = 1,
    KEV_BACKEND_CHAT_EOS = 2,
};
typedef int (*kev_backend_chat_data_fn)(kev_backend_t * b, int kind,
                                        char * out, size_t cap, size_t * needed,
                                        char * err, size_t err_cap);
KEV_BACKEND_API int kev_backend_get_chat_data(kev_backend_t * b, int kind,
                                             char * out, size_t cap, size_t * needed,
                                             char * err, size_t err_cap);

#ifdef __cplusplus
} // extern "C"
#endif
