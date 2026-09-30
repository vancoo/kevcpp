// kev_model.cpp
#include "kev_model.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <set>
#include <string>
#include <thread>

namespace kev {

namespace {
double now_ms() {
    return (double) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() / 1e3;
}
} // namespace

KevModel::~KevModel() { close(); }

void KevModel::close() {
    if (lora_)  llama_adapter_lora_free(lora_);
    if (ctx_)   llama_free(ctx_);
    if (model_) llama_model_free(model_);
    if (ctx_ || model_) llama_backend_free();
    ctx_ = nullptr; model_ = nullptr; vocab_ = nullptr; lora_ = nullptr;
    ok_ = false;
}

bool KevModel::init(const KevModelOptions & opt, std::string * err) {
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    model_ = llama_model_load_from_file(opt.model_path.c_str(), mparams);
    if (!model_) { if (err) *err = "failed to load model: " + opt.model_path; return false; }
    vocab_ = llama_model_get_vocab(model_);

    const int cap = opt.lru_capacity > 0 ? opt.lru_capacity : 2;
    const int base = 1;                 // seq 0 = scratch, seq 1..cap = state slots
    const int scratch = base + cap;
    n_seq_max_ = scratch + 1;
    scratch_seq_ = scratch;
    lru_ = PrefixLRU(cap, base);
    // A3(M7): n_threads==0 时按逻辑/物理核数自动检测, 默认开满核(上限 32 防超大核过度);
    // 否则 llama.cpp 默认 0/1 线程导致 ~1.1s。--threads N 可显式覆盖。
    n_threads_ = opt.n_threads > 0 ? opt.n_threads : (int) std::thread::hardware_concurrency();
    if (n_threads_ <= 0) n_threads_ = 1;
    if (n_threads_ > 32) n_threads_ = 32;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = (uint32_t) (opt.n_ctx > 0 ? opt.n_ctx : 4096);
    cparams.n_batch           = 512;
    cparams.n_ubatch          = 512;
    cparams.n_seq_max         = (uint32_t) n_seq_max_;
    cparams.n_outputs_max_per_seq = 512;
    cparams.embeddings        = true;
    cparams.pooling_type      = LLAMA_POOLING_TYPE_NONE;
    cparams.n_threads         = (int32_t) n_threads_;
    cparams.n_threads_batch   = (int32_t) n_threads_;
    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) { if (err) *err = "failed to init context"; close(); return false; }

    // M5: 可选 LoRA 适配器 (llama.cpp 原生)
    if (!opt.lora_path.empty()) {
        lora_ = llama_adapter_lora_init(model_, opt.lora_path.c_str());
        if (!lora_) { if (err) *err = "failed to load lora adapter: " + opt.lora_path; close(); return false; }
        float scale = opt.lora_scale;
        if (llama_set_adapters_lora(ctx_, &lora_, 1, &scale) != 0) {
            if (err) *err = "failed to set lora adapters";
            close(); return false;
        }
        std::fprintf(stderr, "kev_model: lora adapter loaded (%s, scale=%.2f)\n",
                     opt.lora_path.c_str(), scale);
    }

    if (!opt.head_path.empty()) {
        if (!head_.load_from_file(opt.head_path)) { if (err) *err = "failed to load head: " + opt.head_path; return false; }
    }
    if (head_.dim_in() != 0 && head_.dim_in() != llama_model_n_embd(model_)) {
        if (err) *err = "head dim_in mismatch";
        return false;
    }
    ok_ = true;
    return true;
}

KevEnc KevModel::encode(const KevRequest & req, std::string * err) const {
    return kev::encode(vocab_, req, 384, 1024, err);
}

KevRows KevModel::rows_of(const KevEnc & enc, std::string * err) const {
    return kev::rows_of(enc, err);
}

bool KevModel::decode_seq(const std::vector<int32_t> & tokens, int start_pos,
                          llama_seq_id seq, std::string * err) {
    const int32_t n = (int32_t) tokens.size();
    if (n == 0) { if (err) *err = "empty sequence to decode"; return false; }
    llama_batch batch = llama_batch_init(n, /*embd=*/0, n_seq_max_);
    for (int32_t i = 0; i < n; ++i) {
        batch.token[i] = (llama_token) tokens[i];
        batch.pos[i]   = (llama_pos)(start_pos + i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq;
        batch.logits[i] = 1;   // output embeddings for all tokens (pooling NONE)
    }
    batch.n_tokens = n;
    const int r = llama_decode(ctx_, batch);
    llama_batch_free(batch);
    if (r != 0) { if (err) *err = "decode failed ret=" + std::to_string(r); return false; }
    return true;
}

std::vector<float> KevModel::hidden_at_output(int out_idx, std::string * err) {
    float * p = llama_get_embeddings_ith(ctx_, out_idx);
    const int n_embd = llama_model_n_embd(model_);
    if (!p) { if (err) *err = "no embedding at output idx " + std::to_string(out_idx); return {}; }
    return std::vector<float>(p, p + n_embd);
}

int KevModel::ensure_state(const std::vector<int32_t> & state_ids, double * ps_state, bool * hit) {
    int evict = -1;
    auto [cached, seq] = lru_.lookup(state_ids, &evict);
    double t0 = now_ms();
    if (cached) {
        *hit = true;
        *ps_state = now_ms() - t0;
        return seq;
    }
    *hit = false;
    // prefill into `evict` (clear any previous content there, then decode state)
    llama_memory_t mem = llama_get_memory(ctx_);
    llama_memory_seq_rm(mem, evict, 0, -1);
    std::string err;
    if (!decode_seq(state_ids, /*start_pos=*/0, evict, &err)) {
        std::fprintf(stderr, "kev_model: state prefill failed: %s\n", err.c_str());
        return -1;
    }
    lru_.register_(state_ids, evict);
    *ps_state = now_ms() - t0;
    return evict;
}

std::vector<std::vector<float>> KevModel::rows_from_slot(const KevRows & rows, int slot, double * t_rows,
                                                         std::vector<std::vector<float>> * out_hidden,
                                                         std::vector<double> * rows_ms) {
    *t_rows = 0.0;
    const int Ls = (int) rows.state_ids.size();
    llama_memory_t mem = llama_get_memory(ctx_);
    std::vector<std::vector<float>> probs;
    if (out_hidden) out_hidden->clear();
    if (rows_ms) rows_ms->clear();
    std::string err;
    for (const auto & r : rows.rows) {
        double t0 = now_ms();
        llama_memory_seq_cp(mem, slot, scratch_seq_, 0, -1);   // replicate state prefix (KV + recurrent) to scratch
        if (!decode_seq(r.ids, Ls, scratch_seq_, &err)) {
            std::fprintf(stderr, "kev_model: branch decode failed: %s\n", err.c_str());
            probs.clear(); return probs;
        }
        auto hd = hidden_at_output(r.decide, &err);
        std::vector<const float *> optp;
        std::vector<std::vector<float>> hs;
        for (int32_t o : r.opts) {
            auto v = hidden_at_output(o, &err);
            hs.push_back(v); optp.push_back(hs.back().data());
        }
        if (hd.empty()) { probs.clear(); return probs; }
        probs.push_back(head_.probs(hd.data(), optp));
        if (out_hidden) out_hidden->push_back(std::move(hd));
        double tq = now_ms() - t0;
        if (rows_ms) rows_ms->push_back(tq);
        *t_rows += tq;
    }
    return probs;
}

KevModel::ProbsResult KevModel::probs(const KevRequest & req, std::string * err) {
    ProbsResult res;
    double t0 = now_ms();
    auto enc = encode(req, err);
    if (enc.decide_idx.empty()) { if (err && err->empty()) *err = "no questions"; return res; }
    auto rows = rows_of(enc, err);

    int slot = ensure_state(rows.state_ids, &res.t_state_ms, &res.state_hit);
    if (slot < 0) { if (err) *err = "ensure_state failed"; return res; }
    res.probs = rows_from_slot(rows, slot, &res.t_rows_ms, &res.hidden_decide, &res.rows_ms_per_q);
    res.t_total_ms = now_ms() - t0;
    return res;
}

int KevModel::count_tokens(const std::string & text) const {
    if (!vocab_) return 0;
    return (int) kev::tokenize(vocab_, text, /*add_special=*/false, /*parse_special=*/false).size();
}

} // namespace kev