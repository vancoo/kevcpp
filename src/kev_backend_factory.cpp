// kev_backend_factory.cpp — Backend RAII wrapper + plugin factory.
#include "kev_backend_factory.h"
#include "kev_chat_template.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#  define NOMINMAX
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace kev {

// ---------------------------------------------------------------------------
// Backend (RAII wrapper around the C-ABI ops + handle)
// ---------------------------------------------------------------------------

Backend & Backend::operator=(Backend && other) noexcept {
    if (this != &other) {
        close();
        std::memcpy(&ops_, &other.ops_, sizeof(ops_));
        handle_ = other.handle_;
        meta_   = other.meta_;
        chat_data_ = other.chat_data_;
        other.chat_data_ = nullptr;
        std::memset(&other.ops_, 0, sizeof(other.ops_));
        other.handle_ = nullptr;
        std::memset(&other.meta_, 0, sizeof(other.meta_));
    }
    return *this;
}

void Backend::close() {
    if (handle_ && ops_.destroy) ops_.destroy(handle_);
    handle_ = nullptr;
    std::memset(&ops_, 0, sizeof(ops_));
    std::memset(&meta_, 0, sizeof(meta_));
    chat_data_ = nullptr;
}

namespace {
void set_err(std::string * err, const std::string & msg) { if (err) *err = msg; }
} // namespace

bool Backend::init(const BackendOptions & opt, std::string * err) {
    if (!handle_ || !ops_.init) { set_err(err, "backend not loaded"); return false; }
    kev_backend_params p{};
    p.struct_size           = sizeof(kev_backend_params);
    p.model_path            = opt.model_path.c_str();
    p.lora_path             = opt.lora_path.empty() ? "" : opt.lora_path.c_str();
    p.lora_scale            = opt.lora_scale;
    p.lora_premerged        = opt.lora_premerged;
    p.n_ctx                 = opt.n_ctx;
    p.n_seq_max             = opt.n_seq_max;
    p.n_threads             = opt.n_threads;
    p.n_batch               = opt.n_batch;
    p.n_ubatch              = opt.n_ubatch;
    p.n_outputs_max_per_seq = opt.n_outputs_max_per_seq;
    p.embeddings            = opt.embeddings;
    p.lm_head               = opt.lm_head;
    p.n_gpu_layers          = opt.n_gpu_layers;
    p.moe_offload           = opt.moe_offload;

    char ebuf[1024] = {0};
    if (ops_.init(handle_, &p, ebuf, sizeof(ebuf)) != 0) {
        set_err(err, std::string("backend init failed: ") + (ebuf[0] ? ebuf : "unknown"));
        return false;
    }
    return true;
}

bool Backend::decode_seq(const std::vector<int32_t> & tokens, int32_t start_pos, int32_t seq,
                         std::string * err, const std::vector<int32_t> & logits_pos) {
    if (!handle_ || !ops_.decode_seq) { set_err(err, "backend not loaded"); return false; }
    kev_backend_decode_seq d{};
    d.tokens       = tokens.empty() ? nullptr : tokens.data();
    d.n_tokens     = (int32_t) tokens.size();
    d.start_pos    = start_pos;
    d.seq          = seq;
    d.logits_pos   = logits_pos.empty() ? nullptr : logits_pos.data();
    d.n_logits_pos = (int32_t) logits_pos.size();
    char ebuf[1024] = {0};
    if (ops_.decode_seq(handle_, &d, ebuf, sizeof(ebuf)) != 0) {
        set_err(err, ebuf[0] ? ebuf : "decode_seq failed");
        return false;
    }
    return true;
}

bool Backend::decode_batch(const std::vector<BackendBranch> & branches, std::string * err) {
    if (!handle_ || !ops_.decode_batch) { set_err(err, "backend not loaded"); return false; }
    if (branches.empty()) { set_err(err, "empty batch"); return false; }
    std::vector<kev_backend_decode_branch> rows(branches.size());
    for (size_t i = 0; i < branches.size(); ++i) {
        const BackendBranch & b = branches[i];
        rows[i].ids       = b.ids.empty() ? nullptr : b.ids.data();
        rows[i].n_ids     = (int32_t) b.ids.size();
        rows[i].start_pos = b.start_pos;
        rows[i].seq       = b.seq;
        rows[i].decide    = b.decide;
        rows[i].opts      = b.opts.empty() ? nullptr : b.opts.data();
        rows[i].n_opts    = (int32_t) b.opts.size();
    }
    char ebuf[1024] = {0};
    if (ops_.decode_batch(handle_, rows.data(), (int32_t) rows.size(), ebuf, sizeof(ebuf)) != 0) {
        set_err(err, ebuf[0] ? ebuf : "decode_batch failed");
        return false;
    }
    return true;
}

bool Backend::decode_multi_seq(const std::vector<MultiSeqRequest> & reqs,
                               std::string * err,
                               const std::vector<std::pair<int32_t,int32_t>> & out,
                               std::vector<int32_t> * out_idx) {
    if (!handle_ || !ops_.decode_multi_seq) { set_err(err, "backend does not support decode_multi_seq (plugin ABI too old)"); return false; }
    if (reqs.empty()) { set_err(err, "empty multi-seq batch"); return false; }
    // Flatten tokens seq-major and gather the per-seq side arrays.
    std::vector<int32_t> tokens;              tokens.reserve(0);
    std::vector<int32_t> n_tokens(reqs.size());
    std::vector<int32_t> start_pos(reqs.size());
    std::vector<int32_t> seqs(reqs.size());
    for (size_t s = 0; s < reqs.size(); ++s) {
        const MultiSeqRequest & r = reqs[s];
        n_tokens[s]  = (int32_t) r.tokens.size();
        start_pos[s] = r.start_pos;
        seqs[s]      = r.seq;
        if (r.tokens.empty()) { set_err(err, "empty sequence in multi-seq batch"); return false; }
        tokens.insert(tokens.end(), r.tokens.begin(), r.tokens.end());
    }
    std::vector<int32_t> out_seq(out.size());
    std::vector<int32_t> out_pos(out.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out_seq[i] = out[i].first;
        out_pos[i] = out[i].second;
    }
    std::vector<int32_t> idx(out.size());
    char ebuf[1024] = {0};
    const int rc = ops_.decode_multi_seq(
        handle_,
        tokens.empty() ? nullptr : tokens.data(),
        n_tokens.data(), start_pos.data(), seqs.data(), (int32_t) reqs.size(),
        out.empty() ? nullptr : out_seq.data(),
        out.empty() ? nullptr : out_pos.data(), (int32_t) out.size(),
        idx.data(), (int32_t) idx.size(),
        ebuf, sizeof(ebuf));
    if (rc != 0) {
        set_err(err, ebuf[0] ? ebuf : "decode_multi_seq failed");
        return false;
    }
    if (out_idx) *out_idx = std::move(idx);
    return true;
}

std::vector<float> Backend::read_hidden(int32_t out_idx, std::string * err) {
    if (!handle_ || !ops_.read_hidden) { set_err(err, "backend not loaded"); return {}; }
    const int32_t n = model_n_embd();
    if (n <= 0) { set_err(err, "model_n_embd unavailable"); return {}; }
    std::vector<float> out((size_t) n);
    char ebuf[1024] = {0};
    if (ops_.read_hidden(handle_, out_idx, out.data(), out.size() * sizeof(float), ebuf, sizeof(ebuf)) != 0) {
        set_err(err, ebuf[0] ? ebuf : "read_hidden failed");
        return {};
    }
    return out;
}

bool Backend::read_logits(int32_t out_idx, int32_t token_id, float * out, std::string * err) {
    if (!handle_ || !ops_.read_logits) { set_err(err, "backend does not support read_logits (plugin ABI too old)"); return false; }
    if (!out) { set_err(err, "null out"); return false; }
    if (ops_.read_logits(handle_, out_idx, token_id, out) != 0) {
        set_err(err, "read_logits failed (position not marked for output or lm_head disabled)");
        return false;
    }
    return true;
}

int32_t Backend::vocab_n_tokens() const {
    return (handle_ && ops_.vocab_n_tokens) ? ops_.vocab_n_tokens(handle_) : 0;
}

bool Backend::seq_clear(int32_t seq) {
    return handle_ && ops_.seq_clear && ops_.seq_clear(handle_, seq) == 0;
}
bool Backend::seq_copy(int32_t src, int32_t dst) {
    return handle_ && ops_.seq_copy && ops_.seq_copy(handle_, src, dst) == 0;
}

int32_t Backend::n_ctx_seq_capacity() const {
    return (handle_ && ops_.n_ctx_seq_capacity) ? ops_.n_ctx_seq_capacity(handle_) : 0;
}
int32_t Backend::model_n_embd() const {
    return (handle_ && ops_.model_n_embd) ? ops_.model_n_embd(handle_) : 0;
}
int32_t Backend::threads() const {
    return (handle_ && ops_.threads) ? ops_.threads(handle_) : 0;
}
int32_t Backend::threads_batch() const {
    return (handle_ && ops_.threads_batch) ? ops_.threads_batch(handle_) : 0;
}
int32_t Backend::n_batch() const {
    return (handle_ && ops_.n_batch) ? ops_.n_batch(handle_) : 0;
}
void * Backend::vocab_opaque() const {
    return (handle_ && ops_.vocab_opaque) ? (void *) ops_.vocab_opaque(handle_) : nullptr;
}

namespace {
// Non-capturing adapters that forward the llama-free kev::TokenFeeder contract to
// the backend ops. `ud` points to a kev::BackendFeederState (a copy of ops + handle
// owned by the enclosing kev::Backend), valid for that Backend's lifetime.
int32_t feeder_tokenize(void * ud, const char * text, int32_t text_len,
                        bool add_special, bool parse_special,
                        int32_t * out_ids, int32_t cap) {
    const auto * st = static_cast<const kev::BackendFeederState *>(ud);
    if (!st || !st->handle || !st->ops.tokenize) return -1;
    return st->ops.tokenize(st->handle, text, text_len, add_special, parse_special,
                            out_ids, cap, nullptr, 0);
}
int32_t feeder_special_token_id(void * ud, const char * special) {
    const auto * st = static_cast<const kev::BackendFeederState *>(ud);
    if (!st || !st->handle || !st->ops.special_token_id) return -1;
    return st->ops.special_token_id(st->handle, special);
}
} // namespace

TokenFeeder Backend::token_feeder() const {
    feeder_state_.ops = ops_;
    feeder_state_.handle = handle_;
    TokenFeeder tf;
    tf.ud = &feeder_state_;
    tf.tokenize         = &feeder_tokenize;
    tf.special_token_id = &feeder_special_token_id;
    return tf;
}

bool Backend::chat_template_data(std::string * source, std::string * bos,
                                 std::string * eos, std::string * err) const {
    if (err) err->clear();
    if (!source || !bos || !eos) { set_err(err, "null chat template output"); return false; }
    if (!handle_ || !chat_data_) {
        set_err(err, "backend does not support GGUF chat template metadata");
        return false;
    }
    std::string values[3];
    for (int kind = 0; kind < 3; ++kind) {
        size_t needed = 0;
        char ebuf[1024] = {};
        int rc = chat_data_(handle_, kind, nullptr, 0, &needed, ebuf, sizeof(ebuf));
        if (rc != 2 || needed == 0 || needed > 16 * 1024 * 1024) {
            set_err(err, ebuf[0] ? ebuf : "invalid chat metadata size or plugin response");
            return false;
        }
        std::vector<char> buf(needed);
        size_t actual = 0;
        rc = chat_data_(handle_, kind, buf.data(), buf.size(), &actual, ebuf, sizeof(ebuf));
        if (rc != 0 || actual != needed || buf.back() != '\0' ||
            std::memchr(buf.data(), '\0', buf.size() - 1) != nullptr) {
            set_err(err, ebuf[0] ? ebuf : "invalid chat metadata plugin response");
            return false;
        }
        values[kind].assign(buf.data(), buf.size() - 1);
    }
    *source = std::move(values[0]);
    *bos = std::move(values[1]);
    *eos = std::move(values[2]);
    return true;
}

bool Backend::render_chat(const std::string & system, const std::string & user,
                          std::string * output, std::string * err) const {
    if (output == err && output) return false;
    if (!output) { set_err(err, "invalid chat output"); return false; }
    try {
        if (err) err->clear();
        std::string source, bos, eos;
        if (!chat_template_data(&source, &bos, &eos, err)) return false;
        std::string rendered = render_gguf_chat_template(source, bos, eos, system, user);
        *output = std::move(rendered);
        return true;
    } catch (const std::exception & e) {
        set_err(err, std::string("GGUF chat template render failed: ") + e.what());
    } catch (...) {
        set_err(err, "GGUF chat template render failed: unknown error");
    }
    return false;
}

// Definition of the friend free function declared in kev_backend_factory.h.
std::unique_ptr<Backend> Backend_wrap(kev_backend_ops ops, kev_backend_meta meta,
                                      kev_backend_t * handle, kev_backend_chat_data_fn chat_data) {
    auto b = std::unique_ptr<Backend>(new Backend());
    b->adopt(ops, meta, handle);
    b->chat_data_ = chat_data;
    return b;
}

// ---------------------------------------------------------------------------
// Plugin loading
// ---------------------------------------------------------------------------

#if defined(_WIN32)
struct PluginHandle { HMODULE mod; };
static bool load_symbol(PluginHandle & h, const char * name, void ** fn) {
    *fn = (void *) GetProcAddress(h.mod, name);
    return *fn != nullptr;
}
static void unload_plugin(PluginHandle & h) { if (h.mod) { FreeLibrary(h.mod); h.mod = nullptr; } }
static const char * dl_error() { return "GetProcAddress/LoadLibrary failed"; }
#else
struct PluginHandle { void * mod; };
static bool load_symbol(PluginHandle & h, const char * name, void ** fn) {
    *fn = dlsym(h.mod, name);
    return *fn != nullptr;
}
static void unload_plugin(PluginHandle & h) { if (h.mod) { dlclose(h.mod); h.mod = nullptr; } }
static const char * dl_error() { return dlerror() ? dlerror() : "dlopen/dlsym failed"; }
#endif

// Wrap a loaded set of factory symbols into a kev::Backend (plugin or in-process).
static std::unique_ptr<Backend> wrap_loaded(
        kev_backend_meta (*meta_fn)(void),
        kev_backend_ops  (*ops_fn)(void),
        kev_backend_t  * (*create_fn)(void),
        kev_backend_chat_data_fn chat_data,
        std::string * err) {
    if (!meta_fn || !ops_fn || !create_fn) {
        set_err(err, "backend plugin missing required symbols (kev_backend_get_meta/get_ops/create/destroy)");
        return nullptr;
    }
    kev_backend_meta meta = meta_fn();
    kev_backend_ops  ops  = ops_fn();
    if (meta.abi_version != KEV_BACKEND_ABI_VERSION || ops.struct_size == 0) {
        set_err(err, "backend ABI version mismatch");
        return nullptr;
    }
    // Defensive ABI guard (B-1): the host expects at least sizeof(kev_backend_ops)
    // (including the trailing delegated-tokenizer ops). If the loaded plugin was
    // built from an OLDER kev_backend.h (smaller struct_size), reading the trailing
    // function pointers would be out of bounds — reject up front.
    if (ops.struct_size < sizeof(kev_backend_ops)) {
        set_err(err, "backend ops table too small (plugin ABI older than host); "
                     "rebuild the backend dll from the current kev_backend.h");
        return nullptr;
    }
    kev_backend_t * handle = create_fn();
    if (!handle) { set_err(err, "backend create failed"); return nullptr; }
    return Backend_wrap(ops, meta, handle, chat_data);
}

// In-process CPU backend (single-module static build). Links the exported C
// functions from kev_backend_cpu.cpp directly; main_server and this TU share
// the SAME llama copy — the safe state for bit-identical validation here.
// Compiled only in the static in-process build (KEV_BACKEND_USE_DLL off); when
// the host uses plugin dlls, kev_backend_cpu.cpp is NOT linked into main_server,
// so these symbols do not exist there.
#ifndef KEV_BACKEND_USE_DLL
std::unique_ptr<Backend> load_cpu_inprocess(std::string * err) {
    return wrap_loaded(&kev_backend_get_meta, &kev_backend_get_ops, &kev_backend_create, &kev_backend_get_chat_data, err);
}
#endif

#ifdef KEV_BACKEND_GPU_SINGLE_LLAMA
// In-process GPU (HIP) backend. Only compiled when the enclosing build enables
// GGML_HIP and co-links kev_backend_gpu.cpp into the SAME static llama module as
// main_server (single-llama rule). The GPU backend export symbols are prefixed
// kev_backend_gpu_* so they do not collide with the CPU backend's
// kev_backend_* (both live in one binary in the in-process GPU build).
extern "C" {
    kev_backend_meta kev_backend_gpu_get_meta(void);
    kev_backend_ops  kev_backend_gpu_get_ops(void);
    kev_backend_t  * kev_backend_gpu_create(void);
    int kev_backend_gpu_get_chat_data(kev_backend_t *, int, char *, size_t, size_t *, char *, size_t);
}
std::unique_ptr<Backend> load_gpu_inprocess(std::string * err) {
    return wrap_loaded(&kev_backend_gpu_get_meta, &kev_backend_gpu_get_ops, &kev_backend_gpu_create, &kev_backend_gpu_get_chat_data, err);
}
#endif

#ifdef KEV_BACKEND_USE_DLL
// Load a backend plugin dll by file base name (e.g. "kev-backend-cpu").
// Linked only when KEV_BACKEND_USE_DLL is defined (production portable path;
// requires a stable host + debugger for final validation).
static std::unique_ptr<Backend> load_plugin_dll(const std::string & dll_base, std::string * err) {
    PluginHandle h{};
#if defined(_WIN32)
    std::string name = dll_base + ".dll";
    h.mod = LoadLibraryA(name.c_str());
#else
    std::string name = "lib" + dll_base + ".so";
    h.mod = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    if (!h.mod) { set_err(err, "failed to load backend plugin '" + dll_base + "': " + dl_error()); return nullptr; }

    void * meta_f = nullptr, * ops_f = nullptr, * create_f = nullptr, * destroy_f = nullptr;
    if (!load_symbol(h, "kev_backend_get_meta", &meta_f) ||
        !load_symbol(h, "kev_backend_get_ops",  &ops_f)  ||
        !load_symbol(h, "kev_backend_create",  &create_f) ||
        !load_symbol(h, "kev_backend_destroy", &destroy_f)) {
        set_err(err, "backend plugin '" + dll_base + "' missing same-module factory symbols");
        unload_plugin(h);
        return nullptr;
    }
    void * chat_data_f = nullptr;
    load_symbol(h, "kev_backend_get_chat_data", &chat_data_f);
    auto b = wrap_loaded((kev_backend_meta(*)(void))(meta_f),
                         (kev_backend_ops (*)(void))(ops_f),
                         (kev_backend_t *(*)(void))(create_f),
                         (kev_backend_chat_data_fn)(chat_data_f),
                         err);
    if (!b) unload_plugin(h);
    return b;
}
#else
// Never used in the safe build; kept only to document intent.
static std::unique_ptr<Backend> load_plugin_dll(const std::string & dll_base, std::string * err) {
    (void) dll_base;
    set_err(err, "backend dll loading not compiled (KEV_BACKEND_USE_DLL off)");
    return nullptr;
}
#endif

std::unique_ptr<Backend> BackendFactory_load(const std::string & name, std::string * err) {
    if (err) err->clear();

    // CPU backend: the only one implemented today.
    if (name == "cpu") {
#ifdef KEV_BACKEND_USE_DLL
        return load_plugin_dll("kev-backend-cpu", err);
#else
        return load_cpu_inprocess(err);
#endif
    }

    // GPU / CUDA / Vulkan: loaded as a plugin dll (when KEV_BACKEND_USE_DLL) or
    // an in-process HIP backend when the build co-links GGML_HIP (single-llama
    // in-process, mirrors the CPU in-process path).
    if (name == "vulkan") {
#ifdef KEV_BACKEND_USE_DLL
        // Vulkan uses its OWN dll (kev-backend-vulkan.dll, GGML_VULKAN build),
        // distinct from the HIP gpu dll, so a Vulkan runtime does not need ROCm.
        return load_plugin_dll("kev-backend-vulkan", err);
#else
        set_err(err,
            "backend 'vulkan' requires a plugin-dll host build (KEV_BACKEND_USE_DLL); "
            "this build only supports 'cpu' (and in-process HIP when GGML_HIP is co-linked).");
        return nullptr;
#endif
    }
    if (name == "gpu" || name == "cuda" || name == "cuda0" || name == "llamacpp-gpu") {
#ifdef KEV_BACKEND_USE_DLL
        return load_plugin_dll("kev-backend-gpu", err);
#elif defined(KEV_BACKEND_GPU_SINGLE_LLAMA)
        return load_gpu_inprocess(err);
#else
        set_err(err,
            "backend '" + name + "' is NOT built in this kevcpp build; "
            "the GPU backend requires a GGML_HIP build (KEV_BUILD_BACKEND_GPU=ON "
            "compiled with hipcc/clang + NMake). "
            "Use --backend cpu (the only backend compiled here).");
        return nullptr;
#endif
    }

    set_err(err, "unknown backend: '" + name + "' (expected 'cpu', 'vulkan', or 'gpu'/'cuda' as extension point)");
    return nullptr;
}

} // namespace kev
