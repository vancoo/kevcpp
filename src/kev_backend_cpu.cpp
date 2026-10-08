// kev_backend_cpu.cpp — CPU backend plugin implementing the portable C ABI.
//
// This module implements the kev_backend C ABI from kev_backend.h over the
// vendored llama.cpp / ggml. It is the ONLY module that is allowed to link
// llama statically when running as a plugin dll ("single-llama rule").
//
// Two build flavors are supported (see CMakeLists / docs):
//   * kev-backend-cpu.dll  — shared library loaded by kev::BackendFactory via
//                            LoadLibrary/dlopen (production / portable path). Only
//                            this dll links llama; the host executable must NOT.
//   * static in-process     — linked directly into main_server/bench targets used
//                            for bit-identical validation on a stable box today.
//                            main_server and this TU then share the SAME llama
//                            copy (single module), which is the current safe state.
//
// In the plugin-dll build the ggml CPU variant set (GGML_CPU_ALL_VARIANTS) must be
// enabled with GGML_NATIVE OFF; the VNNI/dpbusd variants must be pruned from the
// deployable set (see docs/portable_backend_deployment.md) because they are
// numerically WRONG on this model.
//
// This file must compile bit-identically to the legacy single-module path:
// the ops below call the same llama_* functions kev_model.cpp used to call
// directly, so decode semantics are preserved exactly (same llama underneath).
#include "kev_backend.h"

#include "llama.h"

#include <cstring>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace {

struct KevCpuBackend {
    llama_model          * model = nullptr;
    llama_context        * ctx   = nullptr;
    llama_adapter_lora   * lora  = nullptr;
    const llama_vocab    * vocab = nullptr;
    int32_t               n_embd = 0;
    int32_t               n_seq_max = 0;
    bool                  ok = false;
};

KevCpuBackend * as_cpu(kev_backend_t * b) {
    return reinterpret_cast<KevCpuBackend *>(b);
}

void write_err(char * err, size_t cap, const std::string & msg) {
    if (err && cap) std::snprintf(err, cap, "%s", msg.c_str());
}

// llama.cpp/ggml log bridge: only ERROR passes through to stderr (filters the
// INFO/WARN metadata/backend dumps that main_server used to suppress via
// llama_log_set before — now handled inside the llama-linking backend dll, since
// the host is llama-free).
static void kev_cpu_log_filter(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (level >= GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

// ---- lifecycle ----

int op_create(kev_backend_t ** out) {
    if (!out) return 1;
    *out = reinterpret_cast<kev_backend_t *>(new KevCpuBackend());
    return 0;
}

void op_destroy(kev_backend_t * b) {
    KevCpuBackend * kb = as_cpu(b);
    if (!kb) return;
    if (kb->lora)  { llama_adapter_lora_free(kb->lora);  kb->lora  = nullptr; }
    if (kb->ctx)   { llama_free(kb->ctx);                kb->ctx   = nullptr; }
    if (kb->model) { llama_model_free(kb->model);        kb->model = nullptr; }
    if (kb->ctx || kb->model) llama_backend_free();
    kb->vocab = nullptr; kb->ok = false;
    delete kb;
}

int op_init(kev_backend_t * b, const kev_backend_params * p, char * err, size_t err_cap) {
    KevCpuBackend * kb = as_cpu(b);
    if (!kb) { write_err(err, err_cap, "null backend"); return 1; }
    if (kb->ctx || kb->model) { write_err(err, err_cap, "backend already initialized"); return 1; }
    if (!p || !p->model_path) { write_err(err, err_cap, "model_path required"); return 1; }

    llama_backend_init();

    // Host is llama-free; the log filter that used to live in main_server is now
    // installed here inside the single-llama module (must be set before any model
    // action, mirroring the old main_server llama_log_set placement).
    llama_log_set(kev_cpu_log_filter, nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    kb->model = llama_model_load_from_file(p->model_path, mparams);
    if (!kb->model) { write_err(err, err_cap, "failed to load model: " + std::string(p->model_path)); return 1; }
    kb->vocab = llama_model_get_vocab(kb->model);
    kb->n_embd = llama_model_n_embd(kb->model);

    const int n_ctx   = p->n_ctx > 0 ? p->n_ctx : 2048;
    kb->n_seq_max     = p->n_seq_max > 0 ? p->n_seq_max : 4;
    const int threads = p->n_threads > 0 ? p->n_threads : (int) std::thread::hardware_concurrency();

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = (uint32_t) n_ctx;
    cparams.n_batch           = (uint32_t) (p->n_batch > 0 ? p->n_batch : 512);
    cparams.n_ubatch          = (uint32_t) (p->n_ubatch > 0 ? p->n_ubatch : 256);
    cparams.n_seq_max         = (uint32_t) kb->n_seq_max;
    cparams.n_outputs_max_per_seq = (uint32_t) (p->n_outputs_max_per_seq > 0 ? p->n_outputs_max_per_seq : 128);
    cparams.embeddings        = p->embeddings != 0;
    cparams.lm_head           = p->lm_head != 0;
    cparams.pooling_type      = LLAMA_POOLING_TYPE_NONE;
    cparams.n_threads         = (int32_t) threads;
    cparams.n_threads_batch   = (int32_t) threads;
    kb->ctx = llama_init_from_model(kb->model, cparams);
    if (!kb->ctx) { write_err(err, err_cap, "failed to init context"); return 1; }

    // Optional LoRA adapter (only when not already folded into the weights).
    if (p->lora_path && p->lora_path[0] != '\0' && !p->lora_premerged) {
        kb->lora = llama_adapter_lora_init(kb->model, p->lora_path);
        if (!kb->lora) { write_err(err, err_cap, "failed to load lora adapter: " + std::string(p->lora_path)); return 1; }
        float scale = p->lora_scale;
        if (llama_set_adapters_lora(kb->ctx, &kb->lora, 1, &scale) != 0) {
            write_err(err, err_cap, "failed to set lora adapters"); return 1;
        }
    }

    kb->ok = true;
    return 0;
}

// ---- decode ----

int op_decode_seq(kev_backend_t * b, const kev_backend_decode_seq * d, char * err, size_t err_cap) {
    KevCpuBackend * kb = as_cpu(b);
    if (!kb || !kb->ctx) { write_err(err, err_cap, "backend not initialized"); return 1; }
    if (!d || d->n_tokens <= 0) { write_err(err, err_cap, "empty sequence to decode"); return 1; }
    const int32_t n = d->n_tokens;

    llama_batch batch = llama_batch_init(n, /*embd=*/0, kb->n_seq_max);
    for (int32_t i = 0; i < n; ++i) {
        batch.token[i]     = (llama_token) d->tokens[i];
        batch.pos[i]       = (llama_pos)(d->start_pos + i);
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = d->seq;
        batch.logits[i]    = 0;
    }
    for (int32_t k = 0; k < d->n_logits_pos; ++k) {
        const int32_t pos = d->logits_pos[k];
        if (pos >= 0 && pos < n) batch.logits[pos] = 1;
    }
    batch.n_tokens = n;

    const int r = llama_decode(kb->ctx, batch);
    llama_batch_free(batch);
    if (r != 0) { write_err(err, err_cap, "decode failed ret=" + std::to_string(r)); return 1; }
    return 0;
}

int op_decode_batch(kev_backend_t * b, const kev_backend_decode_branch * rows, int32_t n_rows,
                    char * err, size_t err_cap) {
    KevCpuBackend * kb = as_cpu(b);
    if (!kb || !kb->ctx) { write_err(err, err_cap, "backend not initialized"); return 1; }
    if (!rows || n_rows <= 0) { write_err(err, err_cap, "empty batch"); return 1; }
    const int32_t L = rows[0].n_ids;
    if (L <= 0) { write_err(err, err_cap, "empty branch in batch"); return 1; }
    for (int32_t s = 0; s < n_rows; ++s) {
        if (rows[s].n_ids != L) { write_err(err, err_cap, "batch requires equal branch lengths"); return 1; }
    }

    const int32_t n = L * n_rows;
    llama_batch batch = llama_batch_init(n, /*embd=*/0, kb->n_seq_max);
    for (int32_t s = 0; s < n_rows; ++s) {
        const kev_backend_decode_branch & br = rows[s];
        const int32_t base = s * L;
        for (int32_t t = 0; t < L; ++t) {
            const int32_t i = base + t;
            batch.token[i]     = (llama_token) br.ids[t];
            batch.pos[i]       = (llama_pos)(br.start_pos + t);
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = br.seq;
            batch.logits[i]    = 0;
        }
        for (int32_t k = 0; k < br.n_opts; ++k) {
            const int32_t o = br.opts[k];
            if (o >= 0 && o < L) batch.logits[base + o] = 1;
        }
        if (br.decide >= 0 && br.decide < L) batch.logits[base + br.decide] = 1;
    }
    batch.n_tokens = n;

    const int r = llama_decode(kb->ctx, batch);
    llama_batch_free(batch);
    if (r != 0) { write_err(err, err_cap, "batch decode failed ret=" + std::to_string(r)); return 1; }
    return 0;
}

int op_read_hidden(kev_backend_t * b, int32_t out_idx, float * out, size_t out_cap,
                   char * err, size_t err_cap) {
    KevCpuBackend * kb = as_cpu(b);
    if (!kb || !kb->ctx) { write_err(err, err_cap, "backend not initialized"); return 1; }
    float * p = llama_get_embeddings_ith(kb->ctx, out_idx);
    if (!p) { write_err(err, err_cap, "no embedding at output idx " + std::to_string(out_idx)); return 1; }
    if (out_cap < (size_t) kb->n_embd) { write_err(err, err_cap, "output buffer too small"); return 1; }
    std::memcpy(out, p, (size_t) kb->n_embd * sizeof(float));
    return 0;
}

int op_seq_clear(kev_backend_t * b, int32_t seq) {
    KevCpuBackend * kb = as_cpu(b);
    if (!kb || !kb->ctx) return 1;
    llama_memory_t mem = llama_get_memory(kb->ctx);
    llama_memory_seq_rm(mem, seq, 0, -1);
    return 0;
}

int op_seq_copy(kev_backend_t * b, int32_t src, int32_t dst) {
    KevCpuBackend * kb = as_cpu(b);
    if (!kb || !kb->ctx) return 1;
    llama_memory_t mem = llama_get_memory(kb->ctx);
    llama_memory_seq_cp(mem, src, dst, 0, -1);
    return 0;
}

// ---- sizing ----

int32_t op_n_ctx_seq_capacity(kev_backend_t * b) {
    KevCpuBackend * kb = as_cpu(b);
    return (kb && kb->ctx) ? (int32_t) llama_n_ctx_seq(kb->ctx) : 0;
}

// Plain llama tokenizer count (no kev special-token rewrite).
// NOTE: KevModel today counts billing tokens in-process via kev::tokenize(vocab_)
// (which applies the kev <|x|> -> <¦x¦> rewrite). This op is the backend-side
// utility for the fully-plugged (single-llama) state; the current in-process
// static build keeps KevModel's count_tokens() path untouched so behaviour is
// bit-identical to the baseline. See docs/portable_backend_deployment.md.
int32_t op_count_tokens(kev_backend_t * b, const char * text) {
    KevCpuBackend * kb = as_cpu(b);
    if (!kb || !kb->vocab || !text) return 0;
    std::vector<llama_token> tokens(4096);
    const int32_t n = llama_tokenize(kb->vocab, text, (int32_t) std::strlen(text),
                                     tokens.data(), (int32_t) tokens.size(),
                                     /*add_special=*/false, /*parse_special=*/false);
    return n < 0 ? 0 : n;
}

int32_t op_model_n_embd(kev_backend_t * b) {
    KevCpuBackend * kb = as_cpu(b);
    return kb ? kb->n_embd : 0;
}

int32_t op_threads(kev_backend_t * b) {
    KevCpuBackend * kb = as_cpu(b);
    return (kb && kb->ctx) ? llama_n_threads(kb->ctx) : 0;
}

int32_t op_threads_batch(kev_backend_t * b) {
    KevCpuBackend * kb = as_cpu(b);
    return (kb && kb->ctx) ? llama_n_threads_batch(kb->ctx) : 0;
}

kev_backend_vocab_t * op_vocab_opaque(kev_backend_t * b) {
    KevCpuBackend * kb = as_cpu(b);
    return kb ? reinterpret_cast<kev_backend_vocab_t *>(const_cast<llama_vocab *>(kb->vocab)) : nullptr;
}

// ---- delegated tokenizer (B-1/B-2) ----
// The host is llama-free (only this dll links llama), so tokenize / special-token
// lookup must be served here. These reproduce the exact llama_tokenize semantics
// kev_encode.cpp used to invoke in-process, so encode is bit-identical across the
// in-process (bench/m3/m5/...) and dll-delegated (main_server) paths.

int32_t op_tokenize(kev_backend_t * b, const char * text, int32_t text_len,
                    bool add_special, bool parse_special,
                    int32_t * out_ids, int32_t cap, char * err, size_t err_cap) {
    KevCpuBackend * kb = as_cpu(b);
    if (!kb || !kb->vocab || !text) { write_err(err, err_cap, "tokenize: no vocab"); return -1; }
    if (text_len < 0) text_len = (int32_t) std::strlen(text);
    if (!out_ids || cap <= 0) return -1;
    std::vector<llama_token> buf((size_t) cap);
    const int32_t n = llama_tokenize(kb->vocab, text, text_len,
                                     buf.data(), (int32_t) buf.size(),
                                     add_special, parse_special);
    if (n < 0) return n;   // llama returns -required-capacity (negative)
    for (int32_t i = 0; i < n; ++i) out_ids[i] = (int32_t) buf[i];
    return n;
}

int32_t op_special_token_id(kev_backend_t * b, const char * special) {
    KevCpuBackend * kb = as_cpu(b);
    if (!kb || !kb->vocab || !special) return -1;
    const int len = (int) std::strlen(special);
    std::vector<llama_token> buf((size_t) len * 4 + 16);
    int32_t n = llama_tokenize(kb->vocab, special, len,
                               buf.data(), (int32_t) buf.size(),
                               /*add_special=*/false, /*parse_special=*/true);
    if (n < 0) {
        buf.resize((size_t) -n);
        n = llama_tokenize(kb->vocab, special, len, buf.data(), (int32_t) buf.size(),
                           /*add_special=*/false, /*parse_special=*/true);
    }
    if (n <= 0) return -1;
    return (int32_t) buf[n - 1];   // <|name|> special token is the last id
}

// Plain-token count helper used by kev_backend_get_ops-less code paths is not
// needed; count goes through the vocab_opaque + kev::tokenize path in-process.

kev_backend_ops g_ops = {};

} // namespace

// ---- exported factory ----

extern "C" {

KEV_BACKEND_API kev_backend_meta kev_backend_get_meta(void) {
    kev_backend_meta m{};
    m.abi_version  = KEV_BACKEND_ABI_VERSION;
    m.name         = "cpu";
    m.device       = "cpu";
    m.version      = "kev-backend-cpu-0.1-static-single-llama";
    m.capabilities = KEV_BACKEND_CAP_BATCH_DECODE | KEV_BACKEND_CAP_SEQ_MEMORY | KEV_BACKEND_CAP_HIDDEN_READ;
    m.struct_size  = sizeof(kev_backend_meta);
    return m;
}

KEV_BACKEND_API kev_backend_ops kev_backend_get_ops(void) {
    if (g_ops.struct_size == 0) {
        g_ops.struct_size     = sizeof(kev_backend_ops);
        g_ops.create          = op_create;
        g_ops.destroy         = op_destroy;
        g_ops.init            = op_init;
        g_ops.decode_seq      = op_decode_seq;
        g_ops.decode_batch    = op_decode_batch;
        g_ops.read_hidden     = op_read_hidden;
        g_ops.seq_clear       = op_seq_clear;
        g_ops.seq_copy        = op_seq_copy;
        g_ops.n_ctx_seq_capacity = op_n_ctx_seq_capacity;
        g_ops.count_tokens    = op_count_tokens;
        g_ops.model_n_embd    = op_model_n_embd;
        g_ops.threads         = op_threads;
        g_ops.threads_batch   = op_threads_batch;
        g_ops.vocab_opaque    = op_vocab_opaque;
        g_ops.tokenize        = op_tokenize;
        g_ops.special_token_id = op_special_token_id;
    }
    return g_ops;
}

KEV_BACKEND_API kev_backend_t * kev_backend_create(void) {
    kev_backend_t * out = nullptr;
    (void) op_create(&out);
    return out;
}

KEV_BACKEND_API void kev_backend_destroy(kev_backend_t * b) {
    op_destroy(b);
}

} // extern "C"
