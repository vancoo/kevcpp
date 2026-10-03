// kev_model.cpp
#include "kev_model.h"
#include "kev_merge.h"
#include "llama.h"

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
// merged-LoRA: 默认启用 merged(把 LoRA 折叠进基础权重, 前向走单份权重)。
// KEV_MERGE_LORA=0 显式回退到分离 LoRA(保留位一致 / 对拍用)。
bool env_force_merge_lora() {
    const char * v = std::getenv("KEV_MERGE_LORA");
    if (v) return std::string(v) != "0";   // 显式值: "0"=off, 其余=on
    return true;                            // 默认 merged
}
// 默认 merged GGUF 落盘路径: <model 同目录>/<basename>-merged.gguf
std::string default_merged_path(const std::string & model_path) {
    namespace fs = std::filesystem;
    try {
        fs::path p(model_path);
        std::string base = p.stem().string();           // 去掉 .gguf
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
    if (lora_)  llama_adapter_lora_free(lora_);
    if (ctx_)   llama_free(ctx_);
    if (model_) llama_model_free(model_);
    if (ctx_ || model_) llama_backend_free();
    ctx_ = nullptr; model_ = nullptr; vocab_ = nullptr; lora_ = nullptr;
    ok_ = false;
}

bool KevModel::init(const KevModelOptions & opt, std::string * err) {
    llama_backend_init();
    // merged-LoRA: 默认启用(把 LoRA fold 进基础权重, 前向走单份权重)。
    // KEV_MERGE_LORA=0 或 opt.merge_lora=false 走分离 LoRA。merged 输出无法落盘时
    // 自动回退到分离 LoRA(不硬失败), 保证生产路径可回退。
    const bool use_merged = opt.merge_lora || env_force_merge_lora();
    std::string model_path = opt.model_path;
    bool merged_active = false;
    if (use_merged && !opt.lora_path.empty()) {
        std::string merged = !opt.merged_model_path.empty() ? opt.merged_model_path
                                                            : default_merged_path(opt.model_path);
        std::string merge_err;
        if (merge_lora_cached(opt.model_path, opt.lora_path, merged, opt.lora_scale, &merge_err)) {
            model_path = merged;
            merged_active = true;
            std::fprintf(stderr, "kev_model: merged-LoRA enabled, loading folded model %s\n",
                         merged.c_str());
        } else {
            std::fprintf(stderr,
                "kev_model: merged-LoRA unavailable (%s); falling back to separate LoRA\n",
                merge_err.c_str());
        }
    }
    if (err && err->size()) err->clear();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    model_ = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model_) { if (err) *err = "failed to load model: " + model_path; return false; }
    vocab_ = llama_model_get_vocab(model_);

    const int cap = opt.lru_capacity > 0 ? opt.lru_capacity : 2;
    const int base = 1;                 // seq 1..cap = state slots
    // P.4: 在 scratch 前预留 kMaxBatch 个可批 branch seq(每次共享同前缀、等长的多问占用)。
    // 顺序: [0...unused] [slot1..slotCap] [branch0..branch(kMaxBatch-1)] [scratch]。
    const int branch_base = base + cap;
    const int scratch     = branch_base + kMaxBatch;
    n_seq_max_ = scratch + 1;
    scratch_seq_ = scratch;
    branch_seq_base_ = branch_base;
    lru_ = PrefixLRU(cap, base);
    // A3(M7): n_threads==0 时按逻辑/物理核数自动检测, 默认开满核(上限 32 防超大核过度);
    // 否则 llama.cpp 默认 0/1 线程导致 ~1.1s。--threads N 可显式覆盖。
    n_threads_ = opt.n_threads > 0 ? opt.n_threads : (int) std::thread::hardware_concurrency();
    if (n_threads_ <= 0) n_threads_ = 1;
    if (n_threads_ > 32) n_threads_ = 32;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = (uint32_t) (opt.n_ctx > 0 ? opt.n_ctx : 2048);
    // P.3(PERF): n_ubatch 512 -> 256。n_ubatch 决定 graph_reserve 与 CPU compute buffer 宽度
    // (491MB -> 245MB, 省一半内存); n_batch 保持 512 = 单次 llama_decode 的容量上限不变
    // (state<=384 / 单 branch 的通用场景不缩水)。实时行宽(state 38 / branch 89)远小于 256,
    // 故每次 decode 仍是单 ubatch, 无切批开销。实测同负载冷启动 latency 中性, 概率逐位一致。
    cparams.n_batch           = 512;
    cparams.n_ubatch          = 256;
    cparams.n_seq_max         = (uint32_t) n_seq_max_;
    cparams.n_outputs_max_per_seq = 128;   // A2(M7): 每问仅 1 decide + n_opts 个输出,512->128 省输出缓冲
    cparams.embeddings        = true;
    cparams.lm_head           = false;   // Rank-1: app only reads hidden via llama_get_embeddings_ith,
                                         // never logits -> skip the 151x248320 lm_head projection entirely.
    cparams.pooling_type      = LLAMA_POOLING_TYPE_NONE;
    cparams.n_threads         = (int32_t) n_threads_;
    cparams.n_threads_batch   = (int32_t) n_threads_;
    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) { if (err) *err = "failed to init context"; close(); return false; }

    // M5: 可选 LoRA 适配器 (llama.cpp 原生)。merged 模式已把 LoRA 折进权重, 不再挂 adapter。
    if (!merged_active && !opt.lora_path.empty()) {
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
    batch_enabled_ = !env_force_per_question();   // P.4 默认启用批量; KEV_NO_BATCH=1 关闭
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
                          llama_seq_id seq, std::string * err,
                          const std::vector<int32_t> & logits_pos) {
    const int32_t n = (int32_t) tokens.size();
    if (n == 0) { if (err) *err = "empty sequence to decode"; return false; }
    // A2(M7): 只在需要 hidden 的位置(<decide>/<opt>)设 batch.logits[i]=1,去掉全 token 输出。
    // 空集合 => 纯 prefill,不产生任何输出。(llama.cpp 侧 output_ids[i]=i,故 output row 索引
    // 仍等于 token 下标,hidden_at_output(idx) 的调用方式不变,概率保持逐位一致。)
    llama_batch batch = llama_batch_init(n, /*embd=*/0, n_seq_max_);
    for (int32_t i = 0; i < n; ++i) {
        batch.token[i] = (llama_token) tokens[i];
        batch.pos[i]   = (llama_pos)(start_pos + i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq;
        batch.logits[i] = 0;
    }
    for (int32_t p : logits_pos) {
        if (p >= 0 && p < n) batch.logits[p] = 1;
    }
    batch.n_tokens = n;
    const int r = llama_decode(ctx_, batch);
    llama_batch_free(batch);
    if (r != 0) { if (err) *err = "decode failed ret=" + std::to_string(r); return false; }
    return true;
}

bool KevModel::decode_batch(const std::vector<BranchDec> & branches, std::string * err) {
    const int Q = (int) branches.size();
    if (Q == 0) { if (err) *err = "empty batch"; return false; }
    const int32_t L = (int32_t) branches[0].ids.size();
    if (L == 0) { if (err) *err = "empty branch in batch"; return false; }
    // 全部 branch 等长(equal_seqs 前提, 否则 llama.cpp find_slot GGML_ASSERT)。
    for (const auto & br : branches) {
        if ((int32_t) br.ids.size() != L) { if (err) *err = "batch requires equal branch lengths"; return false; }
    }

    const int32_t n = L * Q;
    llama_batch batch = llama_batch_init(n, /*embd=*/0, n_seq_max_);
    for (int32_t s = 0; s < Q; ++s) {
        const BranchDec & br = branches[s];
        const int32_t base = s * L;
        for (int32_t t = 0; t < L; ++t) {
            const int32_t i = base + t;
            batch.token[i] = (llama_token) br.ids[t];
            batch.pos[i]   = (llama_pos)(br.start_pos + t);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = br.seq;
            batch.logits[i] = 0;
        }
        // 需要 hidden 的位置(decide/opt)
        for (int32_t o : br.opts) {
            if (o >= 0 && o < L) batch.logits[base + o] = 1;
        }
        if (br.decide >= 0 && br.decide < L) batch.logits[base + br.decide] = 1;
    }
    batch.n_tokens = n;
    const int r = llama_decode(ctx_, batch);
    llama_batch_free(batch);
    if (r != 0) { if (err) *err = "batch decode failed ret=" + std::to_string(r); return false; }
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
    llama_memory_t mem = llama_get_memory(ctx_);
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
        llama_memory_seq_rm(mem, scratch_seq_, 0, -1);
        llama_memory_seq_cp(mem, slot, scratch_seq_, 0, -1);
        std::vector<int32_t> out_pos;
        build_out_pos(qi, out_pos);
        if (!decode_seq(r.ids, Ls, scratch_seq_, &err, out_pos)) {
            std::fprintf(stderr, "kev_model: branch decode failed: %s\n", err.c_str());
            return false;
        }
        return true;
    };

    // 从最近一次 decode 的 output 读取决定/选项 hidden 并算概率, 写回 qi。
    // base_row: 该 branch 在该次 decode 的 batch 内起始行(单问=0, 批= b*L)。
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
            std::vector<BranchDec> branches;
            branches.reserve(g.members.size());
            for (size_t b = 0; b < g.members.size(); ++b) {
                int qi = g.members[b];
                BranchDec bd;
                bd.ids = rows.rows[qi].ids;
                bd.start_pos = Ls;
                bd.seq = branch_seq((int) b);         // 组内第 b 个 group 用 branch seq 池第 b 个(组间可复用)
                bd.decide = rows.rows[qi].decide;
                bd.opts  = rows.rows[qi].opts;
                branches.push_back(std::move(bd));
            }
            { int b = 0; for (int qi : g.members) {    // 每个 branch seq 从 slot 复制共享前缀
                int s = branch_seq((int) b++);
                llama_memory_seq_rm(mem, s, 0, -1);
                llama_memory_seq_cp(mem, slot, s, 0, -1);
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
                ok = finish_to(qi, (int) b * L);   // base_row = b*L (决定/选项都用 base_row + 相对位置)
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
        // P0b: 串行化共享 llama ctx / lru_ / scratch_seq_ 的计算段。
        // rows_of 是 const 只读, encode 在锁外完成; 锁不改变单请求计算顺序, 数值逐位一致。
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
    if (!ctx_) { if (err && err->empty()) *err = "context not initialized"; return false; }
    std::lock_guard<std::mutex> lock(compute_mutex_);
    // 用确定性的通配 token(与 vocab 有界的合法 id) 构造 dummy 序列; token 值不影响图形状/耗时。
    // -- state 形状: 纯 prefill(无 logits 输出), 对标 ensure_state 的 decode_seq(..., {})。
    // -- branch 形状: 带输出位置(logits=1), 对标 rows_from_slot 的 readout decode。
    // 两者都写在 scratch_seq_ 上: 真实 branch decode 前总会先 clear scratch, 故不污染任何
    // 缓存 state / 后续数值; 也不触碰 1..cap 的 LRU slot => state_hit / 缓存行为完全不变。
    llama_memory_t mem = llama_get_memory(ctx_);
    const int S = state_tokens > 0 ? state_tokens : 38;
    std::string warn_err;
    llama_memory_seq_rm(mem, scratch_seq_, 0, -1);
    {
        std::vector<int32_t> stok((size_t) S, 1);
        if (!decode_seq(stok, /*start_pos=*/0, scratch_seq_, err ? err : &warn_err, {})) return false;
    }
    llama_memory_seq_rm(mem, scratch_seq_, 0, -1);
    if (warm_branch) {
        const int B = branch_tokens > 0 ? branch_tokens : 151;
        std::vector<int32_t> btok((size_t) B, 1);
        // 带输出的 readout 形状: 在分支内 take decide/opt 位置(与真实 decode 相同的输出路径)。
        std::vector<int32_t> out_pos;
        if (B >= 2) { out_pos.push_back(B - 1); out_pos.push_back(B / 2); }  // 代表性 decide/opt
        else out_pos.push_back(0);
        if (!decode_seq(btok, /*start_pos=*/S, scratch_seq_, err ? err : &warn_err, out_pos)) return false;
        llama_memory_seq_rm(mem, scratch_seq_, 0, -1);
    }
    return true;
}

int KevModel::count_tokens(const std::string & text) const {
    if (!vocab_) return 0;
    return (int) kev::tokenize(vocab_, text, /*add_special=*/false, /*parse_special=*/false).size();
}

} // namespace kev