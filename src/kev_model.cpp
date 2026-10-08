// kev_model.cpp
#include "kev_model.h"
#include "kev_merge.h"
// NOTE: llama-free host. No #include "llama.h" here — all llama coupling
// (decode/encode/tokenize) goes through the kev::Backend abstraction
// (single-llama rule, see docs/portable_backend_deployment.md).

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <set>
#include <string>
#include <thread>

namespace kev {

namespace {
double now_ms() {
    return (double) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() / 1e3;
}
// P.4: KEV_NO_BATCH=1 强制逐问解码(与批量形态对拍的对照开关, init 时读取一次)。
bool env_force_per_question() {
    static const bool v = std::getenv("KEV_NO_BATCH") != nullptr;
    return v;
}
// merged-LoRA: 默认启用(把 LoRA 折叠进基础权重, 前向走单份权重)。
bool env_force_merge_lora() {
    const char * v = std::getenv("KEV_MERGE_LORA");
    if (v) return std::string(v) != "0";
    return true;
}
std::string default_merged_path(const std::string & model_path) {
    namespace fs = std::filesystem;
    try {
        fs::path p(model_path);
        std::string base = p.stem().string();
        fs::path dir = p.parent_path();
        if (dir.empty()) dir = fs::current_path();
        return (dir / (base + "-merged.gguf")).string();
    } catch (...) {
        return model_path + ".merged.gguf";
    }
}
} // namespace

KevModel::~KevModel() { close(); }

void KevModel::close() {
    backend_.reset();
    ok_ = false;
}

bool KevModel::init(const KevModelOptions & opt, std::string * err) {
    if (err && err->size()) err->clear();

    // merged-LoRA: 默认启用。merged 输出无法落盘时自动回退到分离 LoRA(不硬失败)。
    const bool use_merged = opt.merge_lora || env_force_merge_lora();
    std::string model_path = opt.model_path;
    bool merged_active = false;
    std::string lora_path = opt.lora_path;
    if (use_merged && !opt.lora_path.empty()) {
        std::string merged = !opt.merged_model_path.empty() ? opt.merged_model_path
                                                            : default_merged_path(opt.model_path);
        std::string merge_err;
        if (merge_lora_cached(opt.model_path, opt.lora_path, merged, opt.lora_scale, &merge_err)) {
            model_path = merged;
            merged_active = true;
            lora_path.clear();
            std::fprintf(stderr, "kev_model: merged-LoRA enabled, loading folded model %s\n",
                         merged.c_str());
        } else {
            std::fprintf(stderr,
                "kev_model: merged-LoRA unavailable (%s); falling back to separate LoRA\n",
                merge_err.c_str());
        }
    }

    const int cap = opt.lru_capacity > 0 ? opt.lru_capacity : 2;
    const int base = 1;
    const int branch_base = base + cap;
    const int scratch     = branch_base + kMaxBatch;
    n_seq_max_ = scratch + 1;
    scratch_seq_ = scratch;
    branch_seq_base_ = branch_base;
    lru_ = PrefixLRU(cap, base);
    n_threads_ = opt.n_threads > 0 ? opt.n_threads : (int) std::thread::hardware_concurrency();
    if (n_threads_ <= 0) n_threads_ = 1;
    if (n_threads_ > 32) n_threads_ = 32;

    // Pluggable backend: load by name (default "cpu"), then init it with the
    // same llama parameters the legacy single-module KevModel used.
    backend_ = BackendFactory_load(opt.backend.empty() ? std::string("cpu") : opt.backend, err);
    if (!backend_) {
        if (err && err->empty()) *err = "failed to load backend: " + opt.backend;
        return false;
    }
    BackendOptions bopt;
    bopt.model_path        = model_path;
    bopt.lora_path         = lora_path;
    bopt.lora_scale        = opt.lora_scale;
    bopt.lora_premerged    = merged_active;   // lora already folded -> no adapter
    bopt.n_ctx             = opt.n_ctx > 0 ? opt.n_ctx : 2048;
    bopt.n_seq_max         = n_seq_max_;
    bopt.n_threads         = n_threads_;
    bopt.n_batch           = 512;             // legacy value
    bopt.n_ubatch          = 256;             // legacy value
    bopt.n_outputs_max_per_seq = 128;         // legacy value
    bopt.embeddings        = 1;
    bopt.lm_head           = 0;               // Rank-1: hidden-only serving
    if (!backend_->init(bopt, err)) return false;

    if (!opt.head_path.empty()) {
        if (!head_.load_from_file(opt.head_path)) { if (err) *err = "failed to load head: " + opt.head_path; return false; }
    }
    if (head_.dim_in() != 0 && head_.dim_in() != backend_->model_n_embd()) {
        if (err) *err = "head dim_in mismatch";
        return false;
    }
    std::fprintf(stderr, "kev_model: backend '%s' loaded (device=%s, ver=%s)\n",
                 backend_->name(), backend_->device(), backend_->version());
    batch_enabled_ = !env_force_per_question();
    ok_ = true;
    return true;
}

KevEnc KevModel::encode(const KevRequest & req, std::string * err) const {
    if (!backend_) { if (err && err->empty()) *err = "backend not initialized"; return {}; }
    return kev::encode(backend_->token_feeder(), req, 384, 1024, err);
}

KevRows KevModel::rows_of(const KevEnc & enc, std::string * err) const {
    return kev::rows_of(enc, err);
}

bool KevModel::decode_seq(const std::vector<int32_t> & tokens, int start_pos,
                          int seq, std::string * err,
                          const std::vector<int32_t> & logits_pos) {
    if (!backend_ || !backend_->ok()) { if (err && err->empty()) *err = "backend not initialized"; return false; }
    return backend_->decode_seq(tokens, start_pos, seq, err, logits_pos);
}

bool KevModel::decode_batch(const std::vector<BackendBranch> & branches, std::string * err) {
    if (!backend_ || !backend_->ok()) { if (err && err->empty()) *err = "backend not initialized"; return false; }
    return backend_->decode_batch(branches, err);
}

std::vector<float> KevModel::hidden_at_output(int out_idx, std::string * err) {
    if (!backend_ || !backend_->ok()) { if (err && err->empty()) *err = "backend not initialized"; return {}; }
    return backend_->read_hidden(out_idx, err);
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
    if (backend_) backend_->seq_clear(evict);
    std::string err;
    if (!decode_seq(state_ids, /*start_pos=*/0, evict, &err, {})) {
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
    const int Q  = (int) rows.rows.size();
    std::vector<std::vector<float>> probs(Q);
    std::vector<std::vector<float>> hidden(Q);
    std::vector<double> qms(Q, 0.0);
    if (out_hidden) { out_hidden->clear(); out_hidden->resize(Q); }
    if (rows_ms) { rows_ms->clear(); rows_ms->resize(Q); }
    std::string err;

    // A2(M7): 每题只输出 <decide> 与各 <opt> 位置的 hidden。
    auto build_out_pos = [&](int qi, std::vector<int32_t> & out_pos) {
        const auto & r = rows.rows[qi];
        out_pos.push_back(r.decide);
        for (int32_t o : r.opts)
            if (std::find(out_pos.begin(), out_pos.end(), o) == out_pos.end()) out_pos.push_back(o);
    };

    // ---- 逐问解码(单问 / 无法批: 与旧路径逐位一致) ----
    auto decode_one = [&](int qi) -> bool {
        const auto & r = rows.rows[qi];
        if (backend_) { backend_->seq_clear(scratch_seq_); backend_->seq_copy(slot, scratch_seq_); }
        std::vector<int32_t> out_pos;
        build_out_pos(qi, out_pos);
        if (!decode_seq(r.ids, Ls, scratch_seq_, &err, out_pos)) {
            std::fprintf(stderr, "kev_model: branch decode failed: %s\n", err.c_str());
            return false;
        }
        return true;
    };

    auto finish_to = [&](int qi, int base_row) -> bool {
        const auto & r = rows.rows[qi];
        auto hd = hidden_at_output(base_row + r.decide, &err);
        if (hd.empty()) return false;
        std::vector<const float *> optp;
        std::vector<std::vector<float>> hs;
        for (int32_t o : r.opts) {
            auto v = hidden_at_output(base_row + o, &err);
            if (v.empty()) return false;
            hs.push_back(std::move(v)); optp.push_back(hs.back().data());
        }
        probs[qi] = head_.probs(hd.data(), optp);
        hidden[qi] = std::move(hd);
        return true;
    };

    // ---- P.4 分组: 等长 branch 才可一次 batch decode ----
    const bool batching = batch_enabled_;
    std::vector<char> used(Q, 0);
    struct Group { std::vector<int> members; int len; };
    std::vector<Group> groups;
    for (int i = 0; i < Q; ++i) {
        if (used[i]) continue;
        Group g; g.members.push_back(i); g.len = (int) rows.rows[i].ids.size(); used[i] = 1;
        if (batching) {
            for (int j = i + 1; j < Q; ++j) {
                if (!used[j] && (int) rows.rows[j].ids.size() == g.len && (int) g.members.size() + 1 <= kMaxBatch) {
                    g.members.push_back(j); used[j] = 1;
                }
            }
        }
        groups.push_back(std::move(g));
    }

    for (auto & g : groups) {
        if (g.members.size() >= 2) {
            double t0 = now_ms();
            std::vector<BackendBranch> branches;
            branches.reserve(g.members.size());
            for (size_t b = 0; b < g.members.size(); ++b) {
                int qi = g.members[b];
                BackendBranch bd;
                bd.ids = rows.rows[qi].ids;
                bd.start_pos = Ls;
                bd.seq = branch_seq((int) b);
                bd.decide = rows.rows[qi].decide;
                bd.opts  = rows.rows[qi].opts;
                branches.push_back(std::move(bd));
            }
            { int b = 0; for (int qi : g.members) {    // 每个 branch seq 从 slot 复制共享前缀
                int s = branch_seq((int) b++);
                if (backend_) { backend_->seq_clear(s); backend_->seq_copy(slot, s); }
            } }
            if (!decode_batch(branches, &err)) {
                std::fprintf(stderr, "kev_model: batch decode failed: %s\n", err.c_str());
                probs.clear(); return probs;
            }
            const int32_t L = g.len;
            const double alloc = (now_ms() - t0) / (double) g.members.size();
            bool ok = true;
            for (size_t b = 0; b < g.members.size() && ok; ++b) {
                int qi = g.members[b];
                ok = finish_to(qi, (int) b * L);
                if (ok) { qms[qi] = alloc; *t_rows += alloc; }
            }
            if (!ok) { probs.clear(); return probs; }
        } else {
            int qi = g.members[0];
            double t0 = now_ms();
            if (!decode_one(qi)) { probs.clear(); return probs; }
            if (!finish_to(qi, 0)) { probs.clear(); return probs; }
            double tq = now_ms() - t0;
            qms[qi] = tq; *t_rows += tq;
        }
    }

    if (out_hidden) *out_hidden = std::move(hidden);
    if (rows_ms) *rows_ms = std::move(qms);
    return probs;
}

KevModel::ProbsResult KevModel::probs(const KevRequest & req, std::string * err) {
    auto enc = encode(req, err);
    return probs(enc, err);
}

KevModel::ProbsResult KevModel::probs(const KevEnc & enc, std::string * err) {
    ProbsResult res;
    double t0 = now_ms();
    if (enc.decide_idx.empty()) { if (err && err->empty()) *err = "no questions"; return res; }
    auto rows = rows_of(enc, err);
    {
        std::lock_guard<std::mutex> lock(compute_mutex_);
        int slot = ensure_state(rows.state_ids, &res.t_state_ms, &res.state_hit);
        if (slot < 0) { if (err) *err = "ensure_state failed"; return res; }
        res.probs = rows_from_slot(rows, slot, &res.t_rows_ms, &res.hidden_decide, &res.rows_ms_per_q);
    }
    res.t_total_ms = now_ms() - t0;
    return res;
}

bool KevModel::prewarm_state(const std::vector<int32_t> & state_ids) {
    if (state_ids.empty()) return false;
    std::lock_guard<std::mutex> lock(compute_mutex_);
    double t_state = 0.0;
    bool hit = false;
    int slot = ensure_state(state_ids, &t_state, &hit);
    return slot >= 0;
}

bool KevModel::warm_graph(bool warm_branch, int state_tokens, int branch_tokens, std::string * err) {
    if (!backend_ || !backend_->ok()) { if (err && err->empty()) *err = "context not initialized"; return false; }
    std::lock_guard<std::mutex> lock(compute_mutex_);
    const int S = state_tokens > 0 ? state_tokens : 38;
    std::string warn_err;
    if (backend_) backend_->seq_clear(scratch_seq_);
    {
        std::vector<int32_t> stok((size_t) S, 1);
        if (!decode_seq(stok, /*start_pos=*/0, scratch_seq_, err ? err : &warn_err, {})) return false;
    }
    if (backend_) backend_->seq_clear(scratch_seq_);
    if (warm_branch) {
        const int B = branch_tokens > 0 ? branch_tokens : 151;
        std::vector<int32_t> btok((size_t) B, 1);
        std::vector<int32_t> out_pos;
        if (B >= 2) { out_pos.push_back(B - 1); out_pos.push_back(B / 2); }
        else out_pos.push_back(0);
        if (!decode_seq(btok, /*start_pos=*/S, scratch_seq_, err ? err : &warn_err, out_pos)) return false;
        if (backend_) backend_->seq_clear(scratch_seq_);
    }
    return true;
}

int KevModel::count_tokens(const std::string & text) const {
    if (!backend_) return 0;
    return (int) kev::tokenize(backend_->token_feeder(), text,
                               /*add_special=*/false, /*parse_special=*/false).size();
}

} // namespace kev
