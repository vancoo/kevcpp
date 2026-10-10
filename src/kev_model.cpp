// kev_model.cpp
#include "kev_model.h"
#include "kev_config.h"
// NOTE: llama-free host. No #include "llama.h" here — all llama coupling
// (decode/encode/tokenize) goes through the kev::Backend abstraction
// (single-llama rule, see docs/portable_backend_deployment.md).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
    backend_.reset();
    ok_ = false;
}

bool KevModel::init(const KevModelOptions & opt, std::string * err) {
    if (err && err->size()) err->clear();

    // P2: 前缀缓存槽位数; seq 布局 = scratch(1) + prefix 池。
    const int n_prefix = opt.prefix_capacity > 0 ? opt.prefix_capacity : 4;
    const int scratch   = 1;
    const int prefix_base = scratch + 1;          // P2: 前缀 slot 池(在 scratch 之后)
    n_seq_max_ = prefix_base + n_prefix;
    scratch_seq_ = scratch;
    prefix_seq_base_ = prefix_base;               // P2
    prefix_capacity_ = n_prefix;                  // P2
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

    // Pluggable backend: load by name (default "cpu"), then init it with the
    // same llama parameters the legacy single-module KevModel used.
    backend_ = BackendFactory_load(opt.backend.empty() ? std::string("cpu") : opt.backend, err);
    if (!backend_) {
        if (err && err->empty()) *err = "failed to load backend: " + opt.backend;
        return false;
    }
    BackendOptions bopt;
    bopt.model_path        = opt.model_path;
    bopt.lora_path         = opt.lora_path;
    bopt.lora_scale        = opt.lora_scale;
    bopt.n_ctx             = opt.n_ctx > 0 ? opt.n_ctx : 2048;
    bopt.n_seq_max         = n_seq_max_;
    bopt.n_threads         = n_threads_;
    bopt.n_batch           = 512;             // legacy value
    bopt.n_ubatch          = 256;             // legacy value
    bopt.n_outputs_max_per_seq = 128;         // legacy value
    // 生成式 logits 判别主通道(read_logits 经 llama_get_logits_ith 读 token logits)
    // 只需要 lm_head 投影的 logits 缓冲, 不需要 hidden 输出(embeddings=true 会多付一次
    // hidden 计算)。故 logits 主线恒 关 embeddings、开 lm_head。
    bopt.embeddings        = 0;
    bopt.lm_head           = 1;
    if (!backend_->init(bopt, err)) return false;

    std::fprintf(stderr, "kev_model: backend '%s' loaded (device=%s, ver=%s)\n",
                 backend_->name(), backend_->device(), backend_->version());
    ok_ = true;
    return true;
}

bool KevModel::decode_seq(const std::vector<int32_t> & tokens, int start_pos,
                          int seq, std::string * err,
                          const std::vector<int32_t> & logits_pos) {
    if (!backend_ || !backend_->ok()) { if (err && err->empty()) *err = "backend not initialized"; return false; }
    return backend_->decode_seq(tokens, start_pos, seq, err, logits_pos);
}

// 把 ids 切成 <= chunk 的若干段, 逐段 decode_seq 续写同一 seq, 从而让任何单次
// llama_decode 的 token 数不超过后端 n_batch(后端 decode_seq 内部不切分, 会把整段
// 交给一次 llama_decode; 超 n_batch 即触发 llama.cpp 断言)。所有要读的输出位置
// (out_global, 0-base 于 ids 内)必须落在最后一段, 成功后把它们的 read_logits
// 索引写进 out_local。
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

KevModel::ProbsResult KevModel::probs(const KevRequest & req, std::string * err) {
    // 生成式 logits 判别主通道(logits 主线; 不再有 KEV 回退)。
    return probs_logits(req, err);
}

bool KevModel::logits_channel_active() const {
    return backend_ && backend_->ok() && backend_->vocab_n_tokens() > 0;
}

// 填空式 prompt 渲染({state}/{instructions}/{options} 占位替换)。
std::string KevModel::render_prompt(const std::string & tpl, const KevQuestion & q,
                                    const KevRequest & req,
                                    const std::vector<std::string> & opts) {
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

// logits 通道该请求的输入 token 数 = 各问中最长 prompt 的 token 数。
int KevModel::logits_input_tokens(const KevRequest & req) const {
    if (!backend_ || !backend_->ok()) return 0;
    const TokenFeeder tf = backend_->token_feeder();
    const std::string & tpl = decision_.prompt_template;
    int worst = 0;
    for (const auto & q : req.questions) {
        std::string prompt = render_prompt(tpl, q, req, q.options);
        std::vector<int32_t> P = kev::tokenize(tf, prompt, /*add_special=*/false, /*parse_special=*/false);
        while (!P.empty() && P.back() < 0) P.pop_back();
        worst = std::max(worst, (int) P.size());
    }
    return worst;
}

// 生成式 logits 判别主通道(模型无关, 实测验证路线)。
// 对每问: 用决策配置的 prompt_template 组装填空式 prompt, 把该问每个候选 key 作为"续写
// 答案 token"(aggregation=first 取首个 token), 逐个作为支路 decode; 读该位置模型给该 token
// 的 logits; 对全体候选做相对 softmax(可选 /temperature), 得各 key 概率。候选之间相对比较,
// 不做全局 argmax。
KevModel::ProbsResult KevModel::probs_logits(const KevRequest & req, std::string * err) {
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

    std::vector<std::vector<float>> probs(Q);
    std::vector<double> qms(Q, 0.0);
    std::lock_guard<std::mutex> lock(compute_mutex_);

    for (int k = 0; k < Q; ++k) {
        const KevQuestion & q = req.questions[k];
        const int K = (int) q.keys.size();
        if (K == 0) { if (err && err->empty()) *err = "question " + std::to_string(k) + ": no candidate keys"; return res; }
        std::string prompt = render_prompt(tpl, q, req, q.options);
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
        // 本问的前缀槽(base 之上); <0 表示无可用前缀槽。
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

        softmax_scaled(z, decision_.temperature, probs[k]);
        qms[k] = now_ms() - tq0;
    }
    res.t_rows_ms = 0.0; for (double x : qms) res.t_rows_ms += x;
    res.probs = std::move(probs);
    res.rows_ms_per_q = std::move(qms);
    res.t_total_ms = now_ms() - t0;
    return res;
}

bool KevModel::warm_graph(int state_tokens, int branch_tokens, std::string * err) {
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
    if (branch_tokens > 0) {
        const int B = branch_tokens;
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
