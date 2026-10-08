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
} kev_backend_ops;

// Factory entry points every backend dll must export.
KEV_BACKEND_API kev_backend_meta     kev_backend_get_meta(void);
KEV_BACKEND_API kev_backend_ops      kev_backend_get_ops(void);
KEV_BACKEND_API kev_backend_t *      kev_backend_create(void);
KEV_BACKEND_API void                 kev_backend_destroy(kev_backend_t * b);

#ifdef __cplusplus
} // extern "C"
#endif
