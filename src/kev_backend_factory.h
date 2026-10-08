// kev_backend_factory.h — RAII backend wrapper + factory that loads a backend
// plugin (kev-backend-cpu.dll / kev-backend-gpu.dll / ...) at runtime.
//
// KevModel owns a kev::Backend* and calls it for the llama/ggml operator
// coupling (decode/read_hidden/seq memory/sizing), keeping all orchestration
// (LRU, encode assembly, rows planning, PointerHead, compute_mutex_, n_seq
// layout) inside KevModel.
//
// Factory name -> plugin mapping:
//   "cpu"            -> kev-backend-cpu    (built)
//   "gpu"/"cuda"/"vulkan" -> kev-backend-gpu (extension point; NOT built yet —
//                          returns the clear "not built / use --backend cpu" error)
//   anything else     -> clear "unknown backend" error
#pragma once

#include "kev_backend.h"
#include "kev_encode.h"   // kev::TokenFeeder (llama-free tokenizer accessor)

#include <memory>
#include <string>
#include <vector>

namespace kev {

// Higher-level parameters passed to Backend::init(). Keeps the C-ABI params
// construction inside the factory so KevModel stays decoupled from it.
struct BackendOptions {
    std::string model_path;
    std::string lora_path;      // "" = none
    float       lora_scale   = 1.0f;
    bool        lora_premerged = false;  // true = lora already folded into model weights
    int         n_ctx        = 2048;
    int         n_seq_max    = 4;
    int         n_threads    = 0;       // 0 = auto
    int         n_batch      = 512;
    int         n_ubatch     = 256;
    int         n_outputs_max_per_seq = 128;
    int         embeddings   = 1;
    int         lm_head      = 0;
};

// A single branch for batch decode (backend-ABI flavor; llama_seq_id is int).
struct BackendBranch {
    std::vector<int32_t> ids;
    int32_t              start_pos = 0;
    int32_t              seq       = 0;
    int32_t              decide    = 0;
    std::vector<int32_t> opts;
};

// Backing state for Backend::token_feeder(): holds a copy of the ops + handle so
// the llama-free kev::TokenFeeder closures (non-capturing) can reach them via `ud`.
struct BackendFeederState {
    kev_backend_ops ops{};
    kev_backend_t * handle = nullptr;
};

// RAII wrapper around a live backend instance. Non-copyable.
class Backend {
public:
    Backend() = default;
    ~Backend() { close(); }
    Backend(const Backend &) = delete;
    Backend & operator=(const Backend &) = delete;

    // Move support.
    Backend(Backend && other) noexcept { *this = std::move(other); }
    Backend & operator=(Backend && other) noexcept;

    bool init(const BackendOptions & opt, std::string * err = nullptr);

    // llama/ggml operator coupling.
    bool decode_seq(const std::vector<int32_t> & tokens, int32_t start_pos, int32_t seq,
                    std::string * err,
                    const std::vector<int32_t> & logits_pos = {});
    bool decode_batch(const std::vector<BackendBranch> & branches, std::string * err);
    std::vector<float> read_hidden(int32_t out_idx, std::string * err);
    bool seq_clear(int32_t seq);
    bool seq_copy(int32_t src, int32_t dst);

    int32_t n_ctx_seq_capacity() const;
    int32_t model_n_embd() const;
    int32_t threads() const;
    int32_t threads_batch() const;

    // Opaque tokenizer/vocab handle (castable to const llama_vocab* only by a
    // consumer that itself links llama — single-llama rule).
    void * vocab_opaque() const;

    // llama-free tokenizer accessor (B-3): returns a TokenFeeder bound to this
    // Backend that forwards tokenize / special_token_id to the backend's ops.
    // This is what a llama-free host (main_server / kev_model) uses for encode,
    // instead of casting vocab_opaque() to const llama_vocab*.
    TokenFeeder token_feeder() const;

    const char * name() const { return meta_.name; }
    const char * device() const { return meta_.device; }
    const char * version() const { return meta_.version; }
    bool ok() const { return handle_ != nullptr && ops_.init != nullptr; }
    void close();

private:
    friend std::unique_ptr<Backend> Backend_wrap(kev_backend_ops, kev_backend_meta, kev_backend_t *);

    // factory-internal: adopt a freshly created backend (used by the factory).
    void adopt(kev_backend_ops ops, kev_backend_meta meta, kev_backend_t * handle) {
        ops_ = ops; meta_ = meta; handle_ = handle;
    }

    // Backing state for token_feeder() (refreshed by the const method).
    mutable BackendFeederState feeder_state_;

    kev_backend_ops ops_{};
    kev_backend_t  * handle_ = nullptr;
    kev_backend_meta meta_{};
};

// Loads a backend by name from a plugin dll (LoadLibrary/dlopen) or, in the
// safe in-process build, the statically-linked CPU implementation. Clears the
// error string on success; on failure sets err to a human-readable reason.
std::unique_ptr<Backend> BackendFactory_load(const std::string & name,
                                             std::string * err = nullptr);

// Internal: wraps a freshly-created backend + ops table + meta into a Backend.
// Only BackendFactory_load uses this; declared here so Backend can grant friendship.
std::unique_ptr<Backend> Backend_wrap(kev_backend_ops ops, kev_backend_meta meta,
                                      kev_backend_t * handle);

} // namespace kev
