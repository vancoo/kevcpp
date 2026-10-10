// kev_model.cpp
#include "kev_model.h"
#include "kev_merge.h"
#include "kev_config.h"
// NOTE: llama-free host. No #include "llama.h" here — all llama coupling
// (decode/encode/tokenize) goes through the kev::Backend abstraction
// (single-llama rule, see docs/portable_backend_deployment.md).

#include <algorithm>
#include <chrono>
#include <cmath>
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
    const int n_prefix    = opt.prefix_capacity > 0 ? opt.prefix_capacity : 4;
    const int prefix_base = scratch + 1;            // P2: 前缀 slot 池(在 scratch/branch 之后)
    n_seq_max_ = prefix_base + n_prefix;
    scratch_seq_ = scratch;
    branch_seq_base_ = branch_base;
    prefix_seq_base_ = prefix_base;                 // P2
    prefix_capacity_ = n_prefix;                    // P2
    lru_ = PrefixLRU(cap, base);
    prefix_lru_ = PrefixLRU(n_prefix, prefix_base); // P2: 稳定公共前缀的独立 LRU(按前缀 token ids 索引)
    n_threads_ = opt.n_threads > 0 ? opt.n_threads : (int) std::thread::hardware_concurrency();
    if (n_threads_ <= 0) n_threads_ = 1;
    if (n_threads_ > 32) n_threads_ = 32;

    // ==== 决策配置解析(模型无关 marker + 温度 + logits 通道), 在初始化后端之前解析 ====
    // 因为 enable_logits 决定后端是否需要 lm_head 投影(read_logits 依赖它)。
    decision_ = opt.decision;
    std::string cfg_err;
    if (!opt.decision_config.empty() &&
        !load_decision_config(opt.decision_config, decision_, &cfg_err)) {
        std::string m = "failed to load decision config '" + opt.decision_config + "': " + cfg_err;
        if (err) *err = m;
        return false;
    }
    if (opt.temperature_override > 0.0f) decision_.temperature = opt.temperature_override;
    if (!opt.option_end_override.empty()) decision_.option_end = opt.option_end_override;
    if (!opt.decide_marker_override.empty()) decision_.decide_marker = opt.decide_marker_override;
    // 方案3: 按决策配置构建 logits 决策映射校准器(默认恒等/关闭)。
    rebuild_calibrator(err);

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
    // embeddings: 生成式 logits 判别主通道(read_logits 经 llama_get_logits_ith 读 token logits)
    // 只需要 lm_head 投影的 logits 缓冲, **不需要** hidden 输出(embeddings=true 会多付一次
    // hidden 计算, 实测 ~84ms/+28%)。故 enable_logits=true(主通道) 时关掉 embeddings;
    // 仅当 enable_logits=false(KEV 回退, 经 read_hidden 读 hidden 打分) 时才保留 embeddings。
    // 安全性: 生产服务(/v1/systemone)在 logits 通道只走 read_logits, 无任何 read_hidden /
    // embeddings 消费者; 独立的 main_cli/headbench/m*_bench 各自建自己的 context, 不受影响。
    bopt.embeddings        = decision_.enable_logits ? 0 : 1;
    // lm_head: 生成式 logits 判别主通道需要 lm_head 投影(read_logits 依赖 logits 缓冲);
    // enable_logits=false 时保持原 hidden-only 配置(KEV 回退逐位一致)。get_embeddings 不受影响。
    bopt.lm_head           = decision_.enable_logits ? 1 : 0;
    if (!backend_->init(bopt, err)) return false;

    // ==== 路线 A: 无独立 head.bin / 无 PointerHead 权重 ====
    // (决策配置已在后端初始化前解析完毕, 见上文; head_path 保留兼容但不再使用。)

    // 无权重 KEV 打分器: dim = model_n_embd。
    const int n_embd = backend_->model_n_embd();
    kevhead_.set_dim(n_embd);
    kevhead_.set_temperature(decision_.temperature);
    // (旧 PointerHead head_ 不再加载/使用; kev_head.cpp 仍编译以支持遗留 bench 目标。)

    std::fprintf(stderr, "kev_model: backend '%s' loaded (device=%s, ver=%s)\n",
                 backend_->name(), backend_->device(), backend_->version());
    if (!opt.head_path.empty()) {
        std::fprintf(stderr, "kev_model: warning: --head '%s' is ignored (Route A: no weight head; "
                     "using weightless KEV scoring, n_embd=%d)\n", opt.head_path.c_str(), n_embd);
    }
    batch_enabled_ = !env_force_per_question();
    ok_ = true;
    return true;
}

KevEnc KevModel::encode(const KevRequest & req, std::string * err) const {
    if (!backend_) { if (err && err->empty()) *err = "backend not initialized"; return {}; }
    // 模型无关: 向编码器注入解析后的决策配置(marker + 温度)。
    return kev::encode(backend_->token_feeder(), req, 384, 1024, err, &decision_);
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

// 把 ids 切成 <= chunk 的若干段, 逐段 decode_seq 续写同一 seq, 从而让任何单次
// llama_decode 的 token 数不超过后端 n_batch(后端 decode_seq 内部不切分, 会把整段
// 交给一次 llama_decode; 超 n_batch 即触发 llama.cpp 断言)。所有要读的输出位置
// (out_global, 0-base 于 ids 内)必须落在最后一段, 成功后把它们的 read_logits/
// read_hidden 索引写进 out_local。
bool KevModel::decode_seq_chunked(const std::vector<int32_t> & ids, int start_pos, int seq,
                                  std::string * err, const std::vector<int32_t> & out_global,
                                  std::vector<int32_t> & out_local, int chunk) {
    out_local.clear();
    if (!backend_ || !backend_->ok()) { if (err && err->empty()) *err = "backend not initialized"; return false; }
    const int L = (int) ids.size();
    if (L <= 0) { if (err && err->empty()) *err = "empty sequence to decode"; return false; }
    if (chunk <= 0) chunk = 512;

    // 预检: 若输出位置不在 ids 范围内, 属于调用方错误。
    for (int g : out_global) {
        if (g < 0 || g >= L) { if (err && err->empty()) *err = "decode_seq_chunked: out_global out of range"; return false; }
    }

    // ---- 单段快路径(P0): 整段长度 <= 后端 n_batch 时一次 llama_decode 完成 ----
    // 旧实现无条件把序列切成"头部[无输出] + 尾部[带输出]"两段分别 llama_decode: 长序列
    // (如 6-候选中文 prompt, Lp=189 n_batch=512)会被切成 188+1 两段, 破坏 llama.cpp 的
    // 图/批复用(实测 ~58ms)。当 L <= chunk(后端 n_batch) 时单次 decode 即可容纳整段,
    // 且所有输出位置直接落在该段内(末位置输出 logits), 不触发 llama n_tokens_all<=n_batch
    // 断言。语义: 与"末尾+输出"分块逐位一致(causal 注意力下同一批位置一次性解码只改变
    // 并行/图复用, 不改变 logits 数值, 容许极小浮点差)。
    if (L <= chunk) {
        out_local.clear();
        std::vector<int32_t> local_pos(out_global.size());
        for (size_t i = 0; i < out_global.size(); ++i)
            local_pos[i] = out_global[i];   // 输出位置已在 ids 内的局部位次
        if (!backend_->decode_seq(ids, start_pos, seq, err, local_pos)) return false;
        out_local = std::move(local_pos);
        return true;
    }

    // 输出必须全部落在最后一段。取首个输出位置作为"尾部起始": 头部(其前)可任意切块,
    // 尾部 = [first_out .. L-1], 需保证其长度 <= chunk(本调用方场景输出都在序列末尾,
    // 尾部很短, 恒成立)。
    const int first_out = out_global.empty() ? (L > chunk ? chunk : 0) : *std::min_element(out_global.begin(), out_global.end());
    const int tail_start = first_out;   // 最后一段从这里开始
    if (L - tail_start > chunk) {
        // 输出横跨超过一个 chunk(本设计不支持): 保守直接硬解码, 由调用方/llama 兜底。
        if (err && err->empty()) *err = "decode_seq_chunked: outputs span beyond one chunk";
        return false;
    }

    // 头部: [0 .. tail_start-1], 每次至多 chunk 个 token, 无需输出。
    int pos = 0;
    while (pos < tail_start) {
        const int end = std::min(tail_start, pos + chunk);
        const int seg_len = end - pos;
        std::vector<int32_t> seg(ids.begin() + pos, ids.begin() + end);
        if (!backend_->decode_seq(seg, start_pos + pos, seq, err, {})) return false;
        pos = end;
    }

    // 尾部: [tail_start .. L-1], 标记所有输出(logits)。
    std::vector<int32_t> tail(ids.begin() + tail_start, ids.end());
    std::vector<int32_t> local_pos;
    for (int g : out_global) local_pos.push_back(g - tail_start);
    if (!backend_->decode_seq(tail, start_pos + tail_start, seq, err, local_pos)) return false;
    out_local = local_pos;
    return true;
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
        // 路线 A: 无权重 KEV 打分 (q 半边 · k 半边 / sqrt(m)) + 温度 softmax。
        probs[qi].resize(optp.size());
        KevKevHead::probs(hd.data(), optp.data(), (int) optp.size(),
                          kevhead_.dim(), decision_.temperature, probs[qi].data());
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
    // 生成式 logits 判别主通道生效时走 probs_logits(填空式 prompt + 候选 token logits);
    // 否则回退到旧 KEV 打分(encode -> rows_from_slot)，保持向后兼容。
    if (logits_channel_active()) return probs_logits(req, err);
    auto enc = encode(req, err);
    return probs(enc, err);
}

// 方案3: 只取候选 key 的原始 logits(不应用校准映射的 softmax 概率)。校准/评测 harness 用。
bool KevModel::raw_logits(const KevRequest & req, std::vector<std::vector<float>> & out,
                          std::string * err) {
    out.clear();
    if (!logits_channel_active()) { if (err && err->empty()) *err = "logits channel inactive"; return false; }
    std::string e2;
    const bool prev_enabled = decision_.calib.enabled;
    // 只收集原始 logits, 不应用校准映射(临时关闭校准, 后续由 harness 离线应用)。
    decision_.calib.enabled = false;
    calibrator_.enabled = false;
    probs_logits(req, &e2, &out);
    decision_.calib.enabled = prev_enabled;
    calibrator_.enabled = prev_enabled;
    if (out.empty() && !e2.empty()) {
        if (err && err->empty()) *err = e2;
        return false;
    }
    return true;
}

// 方案3: 按 decision_.calib 重建校准器(含可选的 params_file 加载)。模型无关。
void KevModel::rebuild_calibrator(std::string * err) {
    const LogitsCalibration & c = decision_.calib;
    calibrator_.enabled = c.enabled;
    calibrator_.temperature_mode = c.temperature_mode;
    calibrator_.single_T = c.temperature > 0.0f ? c.temperature : decision_.temperature;
    if (calibrator_.single_T <= 0.0f) calibrator_.single_T = 1.0f;
    calibrator_.position_bias = c.position_bias;
    calibrator_.drop_low_conf = c.drop_low_conf;
    calibrator_.by_n_T.clear();
    if (!c.params_file.empty()) {
        std::string perr;
        if (!calibrator_.load(c.params_file, &perr) && err && err->empty()) *err = perr;
    }
    if (c.enabled && calibrator_.temperature_mode == "by_n" && calibrator_.by_n_T.empty()) {
        // by_n 但无分桶文件: 退化回单档(single_T), 仍生效(诚实)。
        calibrator_.temperature_mode = "single";
    }
}

bool KevModel::logits_channel_active() const {
    return decision_.enable_logits && backend_ &&
           backend_->ok() && backend_->vocab_n_tokens() > 0;
}

namespace {

// 数值稳定 softmax(dirichlet temperature)。z 已按 1/T 缩放; 返回写入 out(长度=z.size)。
void softmax_scaled(const std::vector<float> & z, float temperature, std::vector<float> & out) {
    out.resize(z.size());
    if (z.empty()) return;
    float T = temperature > 0.0f ? temperature : 1.0f;
    float mx = z[0]; for (float x : z) mx = std::max(mx, x);
    std::vector<double> e(z.size()); double s = 0;
    for (size_t i = 0; i < z.size(); ++i) {
        e[i] = std::exp((double)((z[i] - mx) / T));
        s += e[i];
    }
    if (s <= 0) s = 1.0;
    for (size_t i = 0; i < z.size(); ++i) out[i] = (float)(e[i] / s);
}

} // namespace

// P2: 稳定公共前缀 slot 查找/分配(logits 通道)。
// 命中 -> 返回已缓存该前缀 token ids 的 seq(>= prefix_seq_base_); 未命中 -> 返回 -1 并经
// *evict_seq 给出"需先 seq_clear 再 prefill"的旧槽, 由调用方 decode P[0..Dp) 后调 prefix_lru_.register_。
// 无后端 / 前缀空 / 无空闲槽 -> 返回 -1(调用方回退"从 pos 0 全文解码")。
int KevModel::prefix_slot(const std::vector<int32_t> & prefix_ids, int * evict_seq) {
    if (evict_seq) *evict_seq = -1;
    if (!backend_ || !backend_->ok() || prefix_ids.empty() || prefix_capacity_ <= 0) return -1;
    int ev = -1;
    auto [hit, seq] = prefix_lru_.lookup(prefix_ids, &ev);
    if (hit) return seq;                 // 命中: 前缀 KV 已在 seq 中
    if (evict_seq) *evict_seq = ev;      // 未命中: 待 prefill 的旧槽(可能为全新未用槽)
    return -1;
}

// 生成式 logits 判别主通道(模型无关, 实测验证路线)。
// 对每问: 用决策配置的 prompt_template 组装填空式 prompt, 把该问每个候选 key 作为"续写
// 答案 token"(aggregation=first 取首个 token), 逐个作为支路 decode; 读该位置模型给该 token
// 的 logits; 对全体候选做相对 softmax(可选 /temperature), 得各 key 概率。候选之间相对比较,
// 不做全局 argmax。
KevModel::ProbsResult KevModel::probs_logits(const KevRequest & req, std::string * err,
                                             std::vector<std::vector<float>> * raw_out) {
    ProbsResult res;
    double t0 = now_ms();
    const int Q = (int) req.questions.size();
    if (!backend_ || !backend_->ok()) { if (err && err->empty()) *err = "backend not initialized"; return res; }
    if (Q == 0) { if (err && err->empty()) *err = "no questions"; return res; }

    const TokenFeeder tf = backend_->token_feeder();
    const std::string & tpl = decision_.prompt_template;
    const std::string & agg = decision_.aggregation;
    const int nv = backend_->vocab_n_tokens();
    if (nv <= 0) { if (err && err->empty()) *err = "vocab unavailable (read_logits unsupported or lm_head disabled)"; return res; }

    // 组装 prompt 文本: {state} / {instructions} / {options} 占位替换。
    auto render_prompt = [&](const KevQuestion & q, const std::vector<std::string> & opts) {
        std::string p = tpl;
        auto rep = [&](const std::string & ph, const std::string & val) {
            size_t at = 0;
            while ((at = p.find(ph, at)) != std::string::npos) { p.replace(at, ph.size(), val); at += val.size(); }
        };
        std::string opts_str;
        for (size_t i = 0; i < opts.size(); ++i) {
            if (i) opts_str += ", ";
            opts_str += opts[i];
        }
        rep("{state}", req.state);
        rep("{instructions}", q.instr);
        rep("{options}", opts_str);
        return p;
    };

    std::vector<std::vector<float>> probs(Q);
    std::vector<double> qms(Q, 0.0);
    std::lock_guard<std::mutex> lock(compute_mutex_);

    for (int k = 0; k < Q; ++k) {
        const KevQuestion & q = req.questions[k];
        const int K = (int) q.keys.size();
        if (K == 0) { if (err && err->empty()) *err = "question " + std::to_string(k) + ": no candidate keys"; return res; }
        std::string prompt = render_prompt(q, q.options);
        std::vector<int32_t> P = kev::tokenize(tf, prompt, /*add_special=*/false, /*parse_special=*/false);
        while (!P.empty() && P.back() < 0) P.pop_back();   // 剔除退化/占位
        const int32_t Lp = (int32_t) P.size();
        if (Lp == 0) { if (err && err->empty()) *err = "question " + std::to_string(k) + ": empty prompt"; return res; }

        double tq0 = now_ms();
        bool keep = agg == "sum" || agg == "mean";
        std::vector<std::vector<int32_t>> cand_tok(K);
        std::vector<int32_t> cand_first(K, -1);
        for (int j = 0; j < K; ++j) {
            cand_tok[j] = kev::tokenize(tf, q.keys[j], /*add_special=*/false, /*parse_special=*/false);
            cand_first[j] = cand_tok[j].empty() ? -1 : cand_tok[j][0];
        }
        std::vector<float> z(K, 0.0f);
        std::string err2;

        // ---- n_batch 安全解码 ----
        const int32_t chunk = (backend_->n_batch() > 0) ? backend_->n_batch() : 512;

        // ===================== P2: 稳定公共前缀缓存 =====================
        // 结构: {instructions}\n[ {options} ]\n{state}\nAnswer:
        //   [D0..Dp) = tpl 中 {state} 之前的部分(渲染 instructions/options) = 同任务不变的稳定前缀
        //   [Dp..)   = state + "\nAnswer:"(每次请求变化)
        // 仅当 P[0..Dp) 与单独 tokenize(前缀文本) 逐位相等(BPE merge 未跨界)时才可复用缓存 KV —
        // 保证"从公共前缀续接 state"与"从 pos 0 全量解码"得到同一条 token 序列(后者回退路径逐位一致)。
        int Dp = -1;                     // <0 = 本问不使用前缀缓存
        std::vector<int32_t> Pg;         // 稳定前缀 token ids = 前缀缓存指纹
        const bool want_prefix = decision_.cache_prefix && backend_ && backend_->ok();
        if (want_prefix) {
            const size_t sp = tpl.find("{state}");
            if (sp != std::string::npos && tpl.find("{state}", sp + 7) == std::string::npos) {
                std::string pfx_tpl = tpl.substr(0, sp);   // {state} 之前(不含 {state})
                std::string pg_str = pfx_tpl;
                auto rep_pfx = [&](const std::string & ph, const std::string & val) {
                    size_t at = 0;
                    while ((at = pg_str.find(ph, at)) != std::string::npos) { pg_str.replace(at, ph.size(), val); at += val.size(); }
                };
                std::string opts_str_pfx;
                for (size_t oi = 0; oi < q.options.size(); ++oi) { if (oi) opts_str_pfx += ", "; opts_str_pfx += q.options[oi]; }
                rep_pfx("{instructions}", q.instr);
                rep_pfx("{options}", opts_str_pfx);
                Pg = kev::tokenize(tf, pg_str, /*add_special=*/false, /*parse_special=*/false);
                if (!Pg.empty() && (int)Pg.size() < Lp &&
                    std::equal(Pg.begin(), Pg.end(), P.begin(), P.begin() + (int64_t)Pg.size())) {
                    Dp = (int) Pg.size();
                }
            }
        }
        // 本问的前缀槽(base 之上); <0 表示无可用前缀槽(补齐到 pseq 需 prefesh 稳定前缀)。
        int pseq = -1;
        if (Dp > 0) {
            int evict = -1;
            pseq = prefix_slot(Pg, &evict);
            if (pseq < 0 && evict >= 0) {
                // 未命中: 在该槽 prefill 稳定前缀 [0..Dp)
                if (backend_) backend_->seq_clear(evict);
                std::vector<int32_t> out_pre;
                std::vector<int32_t> head(P.begin(), P.begin() + Dp);
                if (decode_seq_chunked(head, 0, evict, &err2, {}, out_pre, chunk)) {
                    prefix_lru_.register_(Pg, evict);
                    pseq = evict;
                }
            }
        }
        const bool use_prefix = (Dp > 0) && (pseq >= 0);
        // 从公共前缀槽续接 tail(state+Answer [+候选]) 到 scratch_seq_: 清空->复制前缀->decode。
        auto decode_tail_from_prefix = [&](const std::vector<int32_t> & tail, int32_t start_pos,
                                           const std::vector<int32_t> & out_global, int pseq,
                                           std::vector<int32_t> & out_local) -> bool {
            if (backend_) backend_->seq_clear(scratch_seq_);
            if (!backend_ || !backend_->seq_copy(pseq, scratch_seq_)) return false;
            return decode_seq_chunked(tail, start_pos, scratch_seq_, &err2, out_global, out_local, chunk);
        };

        if (!keep) {
            // aggregation == "first"(默认, 优化路径): 单次 decode, 末位置 Lp-1 输出全量 logits,
            // 按候选首 token 直接查 logit(K 候选 = 1 次 decode)。
            std::vector<int32_t> out_local;
            bool done = false;
            if (use_prefix) {
                // P2 续接: 只解码 state+Answer(跳过重算稳定前缀)。
                std::vector<int32_t> tail(P.begin() + Dp, P.end());
                std::vector<int32_t> out_tail;
                if (decode_tail_from_prefix(tail, Dp, {Lp - 1 - Dp}, pseq, out_tail) && !out_tail.empty()) {
                    out_local = out_tail; done = true;
                }
            }
            if (!done) {
                // 回退: 从 pos 0 全新解码整条 P(既有路径, 保证功能/正确性不坏)。
                if (backend_) backend_->seq_clear(scratch_seq_);
                if (!decode_seq_chunked(P, 0, scratch_seq_, &err2, {Lp - 1}, out_local, chunk)) {
                    // 单次 decode 失败(如仍超 n_batch 或其它): 降级逐候选(结果等价, 不崩溃)。
                    std::fprintf(stderr,
                        "kev_model: logits single-decode failed at q=%d (%s); degrading to per-key decode\n",
                        k, err2.c_str());
                    for (int j = 0; j < K; ++j) {
                        if (cand_first[j] < 0) { z[j] = 0.0f; continue; }
                        std::vector<int32_t> ids = P; ids.push_back(cand_first[j]);
                        if (backend_) backend_->seq_clear(scratch_seq_);
                        std::vector<int32_t> out1;
                        if (!decode_seq_chunked(ids, 0, scratch_seq_, &err2, {Lp - 1}, out1, chunk)) {
                            std::fprintf(stderr,
                                "kev_model: degraded logits seq decode failed at q=%d,key=%d (%s)\n",
                                k, j, err2.c_str());
                            if (err && err->empty()) *err = err2;
                            return res;
                        }
                        float lt = 0;
                        if (out1.empty() || !backend_->read_logits(out1[0], cand_first[j], &lt, &err2)) lt = 0.0f;
                        z[j] = lt;
                    }
                } else {
                    const int32_t out_idx = out_local.empty() ? -1 : out_local[0];
                    for (int j = 0; j < K; ++j) {
                        if (cand_first[j] < 0) { z[j] = 0.0f; continue; }
                        float lt = 0;
                        if (out_idx < 0 || !backend_->read_logits(out_idx, cand_first[j], &lt, &err2)) {
                            std::fprintf(stderr, "kev_model: read_logits failed at q=%d,key=%d (%s); using 0\n",
                                         k, j, err2.c_str());
                            lt = 0.0f;
                        }
                        z[j] = lt;
                    }
                }
            } else {
                const int32_t out_idx = out_local.empty() ? -1 : out_local[0];
                if (out_idx < 0) {
                    if (err && err->empty()) *err = "logits decode produced no output position";
                    return res;
                }
                for (int j = 0; j < K; ++j) {
                    if (cand_first[j] < 0) { z[j] = 0.0f; continue; }
                    float lt = 0;
                    if (!backend_->read_logits(out_idx, cand_first[j], &lt, &err2)) {
                        std::fprintf(stderr, "kev_model: read_logits failed at q=%d,key=%d (%s); using 0\n",
                                     k, j, err2.c_str());
                        lt = 0.0f;
                    }
                    z[j] = lt;
                }
            }
        } else {
            // "sum"/"mean": 逐候选 decode(P + 全 key token), 读每 token 位置的 logits 并聚合。
            for (int j = 0; j < K; ++j) {
                const int32_t M = (int32_t) cand_tok[j].size();
                if (M == 0) { z[j] = 0.0f; continue; }
                std::vector<int32_t> ids;
                std::vector<int32_t> out_global;
                bool okd;
                if (use_prefix) {
                    ids.insert(ids.end(), P.begin() + Dp, P.end());
                    for (int m = 0; m < M; ++m) out_global.push_back((Lp - 1 - Dp) + m);
                    ids.insert(ids.end(), cand_tok[j].begin(), cand_tok[j].end());
                } else {
                    ids = P;
                    for (int m = 0; m < M; ++m) out_global.push_back(Lp - 1 + m);
                    ids.insert(ids.end(), cand_tok[j].begin(), cand_tok[j].end());
                }
                std::vector<int32_t> out_local;
                if (use_prefix) okd = decode_tail_from_prefix(ids, Dp, out_global, pseq, out_local);
                else {
                    if (backend_) backend_->seq_clear(scratch_seq_);
                    okd = decode_seq_chunked(ids, 0, scratch_seq_, &err2, out_global, out_local, chunk);
                }
                if (!okd) {
                    std::fprintf(stderr, "kev_model: logits seq decode failed (%s)\n", err2.c_str());
                    if (err && err->empty()) *err = err2;
                    return res;
                }
                float s = 0;
                for (int m = 0; m < M; ++m) {
                    float lt = 0;
                    if (!backend_->read_logits(out_local[m], cand_tok[j][m], &lt, &err2)) lt = 0.0f;
                    s += lt;
                }
                z[j] = (agg == "mean") ? (s / (float) M) : s;
            }
        }

        if (raw_out) {
            // 收集候选 key 的原始 logits(校准/缩放前), 供校准 harness 离线统计。
            if (raw_out->size() <= (size_t) k) raw_out->resize((size_t) k + 1);
            (*raw_out)[k] = z;
        }
        // 方案3: 决策映射层校准(默认关闭 = 原行为 softmax(z / decision.temperature))。
        if (calibrator_.enabled) {
            calibrator_.apply(z, probs[k]);
        } else {
            softmax_scaled(z, decision_.temperature, probs[k]);
        }
        qms[k] = now_ms() - tq0;
    }
    res.t_rows_ms = 0.0; for (double x : qms) res.t_rows_ms += x;
    res.probs = std::move(probs);
    res.rows_ms_per_q = std::move(qms);
    res.t_state_ms = 0.0; res.state_hit = false;
    res.t_total_ms = now_ms() - t0;
    return res;
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
