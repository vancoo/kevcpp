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

    // P2: 前缀缓存槽位数; seq 布局 = scratch(1) + 前缀池 + B2 并发 scratch 池。
    const int n_prefix = opt.prefix_capacity > 0 ? opt.prefix_capacity : 4;
    const int scratch   = 1;
    const int prefix_base = scratch + 1;          // P2: 前缀 slot 池(在 scratch 之后)
    prefix_seq_base_ = prefix_base;               // P2
    prefix_capacity_ = n_prefix;                  // P2
    const int conc_base = prefix_base + n_prefix; // B2: 并发 scratch 槽(在前缀池之后)
    concurrent_seq_base_ = conc_base;
    max_concurrency_     = opt.max_concurrency > 0 ? opt.max_concurrency : 8;
    if (max_concurrency_ > 64) max_concurrency_ = 64;
    n_seq_max_ = conc_base + max_concurrency_;
    scratch_seq_ = scratch;
    prefix_lru_ = PrefixLRU(n_prefix, prefix_base); // P2: LCP 前缀 LRU(按稳定公共前缀 token ids 索引)
    slot_used_.assign(max_concurrency_, false);
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
    bopt.n_batch           = opt.n_batch > 0 ? opt.n_batch : 512;    // legacy default 512
    bopt.n_ubatch          = opt.n_ubatch > 0 ? opt.n_ubatch : 256;  // legacy default 256
    bopt.n_outputs_max_per_seq = 128;         // legacy value
    // 生成式 logits 判别主通道(read_logits 经 llama_get_logits_ith 读 token logits)
    // 只需要 lm_head 投影的 logits 缓冲, 不需要 hidden 输出(embeddings=true 会多付一次
    // hidden 计算)。故 logits 主线恒 关 embeddings、开 lm_head。
    // · 调研实验: hidden 聚合判别通道(KEV_DECIDE=hidden|hidden_state)需要读隐层向量,
    //   故此处恒开 embeddings=1(read_hidden 可用)。llama.cpp 在 embeddings 模式下会对全部
    //   token 输出 hidden, 但 logits/embeddings 共享同套 output_ids 索引, 现有 read_logits
    //   通道的 out_idx 语义不变, 不受影响(仅多一次 hidden 计算, 实验通道才实际读取)。
    //   仅当非实验(不使用 hidden 通道)时… 实验期统一开, 保持简单。
    bopt.embeddings        = 1;
    bopt.lm_head           = 1;
    // MoE 专家 offload: 透传 n_gpu_layers 与 moe_offload(见 kev_backend.h)。
    bopt.n_gpu_layers      = opt.n_gpu_layers;
    bopt.moe_offload       = opt.moe_offload;
    if (!backend_->init(bopt, err)) return false;

    // B2: 批量调度器(仅在启用且后端支持单 ctx 多 seq decode 时创建)。单 worker 集中
    // llama_decode + read_logits, 并发请求各自用独立 sequence 槽, 合并进同一次 decode。
    batching_enabled_ = opt.enable_batching && backend_->supports_multi_seq();
    if (batching_enabled_) {
        scheduler_ = std::make_unique<BatchScheduler>(backend_.get(), backend_->n_batch(), max_concurrency_);
        scheduler_->set_aggregation_us(opt.batch_aggregation_us);
        std::fprintf(stderr, "kev_model: batch scheduler ON (max_concurrency=%d, max_jobs/decode=%d, "
                             "n_seq_max=%d, conc_base=%d, agg_us=%ld)\n",
                     max_concurrency_, max_concurrency_, n_seq_max_, concurrent_seq_base_,
                     opt.batch_aggregation_us);
    } else {
        std::fprintf(stderr, "kev_model: batch scheduler OFF (enable_batching=%d, backend_multi_seq=%d) "
                             "; serial compute_mutex_ path\n",
                     (int) opt.enable_batching, backend_->supports_multi_seq() ? 1 : 0);
    }

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

// B2: 解码 tokens(可选从缓存 prefix `copy_src` 续接)写入 seq, 在 out_pos 读取候选 logits。
// 批处理路径: 提交给 BatchScheduler 单 worker(集中 llama_decode + read_logits, 与并发请求
// 合并进同一次 decode_multi_seq)。串行路径(compute_mutex_ 已持, 未启用 batch): 直调
// decode_seq_chunked + read_logits。两种模式产出 logits[out][cand]。
bool KevModel::decode_one_to_logits(const std::vector<int32_t> & tokens, int32_t start_pos,
                                    int32_t seq, int32_t copy_src,
                                    const std::vector<int32_t> & out_pos,
                                    const std::vector<std::vector<int32_t>> & out_cands,
                                    std::vector<std::vector<float>> & logits,
                                    std::string * err) {
    logits.assign(out_pos.size(), {});
    if (batching_enabled_ && scheduler_) {
        auto job = std::make_shared<BatchJob>();
        Backend::MultiSeqRequest r;
        r.tokens = tokens; r.start_pos = start_pos; r.seq = seq;
        job->reqs.push_back(std::move(r));
        job->copy_src.push_back(copy_src);
        for (int32_t p : out_pos) job->outs.push_back({0, p});
        job->out_cands = out_cands;
        if (!scheduler_->submit(job)) {
            if (err && err->empty()) *err = job->err.empty() ? std::string("batched decode failed") : job->err;
            return false;
        }
        logits = std::move(job->logits);
        return true;
    }
    // serial fallback (caller holds compute_mutex_; ctx exclusively ours)
    if (!backend_) { if (err && err->empty()) *err = "backend not initialized"; return false; }
    std::string e2;
    (void) backend_->seq_clear(seq);
    if (copy_src >= 0 && !backend_->seq_copy(copy_src, seq)) { if (err && err->empty()) *err = "seq_copy failed"; return false; }
    std::vector<int32_t> out_local;
    const int chunk = backend_->n_batch() > 0 ? backend_->n_batch() : 512;
    if (!decode_seq_chunked(tokens, start_pos, seq, &e2, out_pos, out_local, chunk)) {
        if (err && err->empty()) *err = e2;
        return false;
    }
    for (size_t o = 0; o < out_local.size(); ++o) {
        logits[o].resize(out_cands[o].size());
        for (size_t c = 0; c < out_cands[o].size(); ++c) {
            float v = 0.0f;
            backend_->read_logits(out_local[o], out_cands[o][c], &v, &e2);
            logits[o][c] = v;
        }
    }
    return true;
}

int KevModel::acquire_scratch_slot() {
    std::lock_guard<std::mutex> lk(slot_mutex_);
    for (size_t i = 0; i < slot_used_.size(); ++i) {
        if (!slot_used_[i]) { slot_used_[i] = true; return concurrent_seq_base_ + (int) i; }
    }
    return -1;
}

void KevModel::release_scratch_slot(int seq) {
    std::lock_guard<std::mutex> lk(slot_mutex_);
    const int idx = seq - concurrent_seq_base_;
    if (idx >= 0 && idx < (int) slot_used_.size()) slot_used_[idx] = false;
}

KevModel::ProbsResult KevModel::probs(const KevRequest & req, std::string * err) {
    // 生成式 logits 判别主通道(logits 主线; 不再有 KEV 回退)。
    // 调研实验: 用 KEV_DECIDE 启用 hidden 聚合判别通道(零训练), default 保持 logits。
    //   "hidden"         : score[c] = cos(h_decide, h_cand[c])  决定向量 vs 各候选表示
    //   "hidden_state"   : 先对 {state} 的 hidden 做注意力加权聚合 -> h_agg, 再
    //                       score[c] = cos(h_agg, h_cand[c])(即'用注意力找相关内容再聚合')
    const char * dec = std::getenv("KEV_DECIDE");
    std::string m = dec ? dec : "";
    if (m == "hidden" || m == "hidden_state") {
        return probs_hidden(req, m == "hidden_state", err);
    }
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

// (注: 前缀槽查找/命中/未命中 prefill 的逻辑已内联进 probs_logits 的 P2 块(见上), 使用
// PrefixLRU::lookup 的 LCP 匹配; 这里不再有独立的 prefix_slot 辅助。)

// logits 通道该请求的输入 token 数 = 各问中最长 prompt 的 token 数。
// P0 (concurrency): 该函数在锁外被 API 层(kev_api.cpp logits_input_tokens 预检)并发调用,
// 其内部 kev::tokenize → llama_tokenize 访问共享 llama_vocab(session 态、非线程安全)。
// 故在此取同一把 compute_mutex_, 使“任意时刻至多一个线程在分词”, 与 probs_logits 内受锁
// 保护的 tokenize 串行, 消除真并发下的资料竞争崩溃。锁内只做渲染+分词, 无重入, 无死锁。
int KevModel::logits_input_tokens(const KevRequest & req) const {
    std::lock_guard<std::mutex> lock(compute_mutex_);
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

    // B2: 并发批处理模式不持 compute_mutex_(解码经 BatchScheduler 单 worker 串行 + 批量);
    // 非批处理回退路径保持原「整函数持 compute_mutex_」串行语义。批处理时每请求取一个独立
    // sequence 槽(并发池), 避免请求间 KV 冲突; 分词 / LRU 前缀管理分别用短锁串行。
    const bool batched = batching_enabled_ && scheduler_;
    std::unique_ptr<std::lock_guard<std::mutex>> compute_lock;
    if (!batched) compute_lock.reset(new std::lock_guard<std::mutex>(compute_mutex_));
    const int my_slot = batched ? acquire_scratch_slot() : scratch_seq_;
    if (my_slot < 0) { if (err && err->empty()) *err = "no free concurrent sequence slot"; return res; }
    struct SlotGuard {
        bool active; int seq; KevModel * m;
        ~SlotGuard() { if (active) m->release_scratch_slot(seq); }
    } slot_guard{batched, my_slot, this};
    // 分词(共享 llama_vocab 非线程安全, 短锁串行; 非批处理也已持 compute_mutex_ 无需此锁但无害)。
    auto tok = [&](const std::string & t) {
        std::lock_guard<std::mutex> lk(tokenizer_mutex_);
        return kev::tokenize(tf, t, /*add_special=*/false, /*parse_special=*/false);
    };

    for (int k = 0; k < Q; ++k) {
        const KevQuestion & q = req.questions[k];
        const int K = (int) q.keys.size();
        if (K == 0) { if (err && err->empty()) *err = "question " + std::to_string(k) + ": no candidate keys"; return res; }
        std::string prompt = render_prompt(tpl, q, req, q.options);
        std::vector<int32_t> P = tok(prompt);
        while (!P.empty() && P.back() < 0) P.pop_back();   // 剔除退化/占位
        const int32_t Lp = (int32_t) P.size();
        if (Lp == 0) { if (err && err->empty()) *err = "question " + std::to_string(k) + ": empty prompt"; return res; }

        double tq0 = now_ms();
        bool keep = agg == "sum" || agg == "mean";
        std::vector<std::vector<int32_t>> cand_tok(K);
        std::vector<int32_t> cand_first(K, -1);
        for (int j = 0; j < K; ++j) {
            cand_tok[j] = tok(q.keys[j]);
            cand_first[j] = cand_tok[j].empty() ? -1 : cand_tok[j][0];
        }
        std::vector<float> z(K, 0.0f);
        std::string err2;

        // ===================== P2: 最长公共前缀(LCP)缓存 =====================
        // 默认模板 {instructions}\n[ {options} ]\n{state}\nAnswer: 中 {state} 在**末尾**。故
        // 「{state} 之前的模板段(渲染 instructions+options)」是同一任务跨请求的**稳定公共前缀**,
        // 是跨 state(仅 state 变化)复用 KV 的自然候选。
        // LCP 匹配 + M-RoPE 约束: KV 缓存是连续的一段, 新解码的起始位置必须严格大于该 seq 已存
        // KV 的末尾位置(X < Y), 否则报 M-RoPE 校验失败。因此**只能整体复用一个槽存储的完整前缀**
        // (复用长度 = 该槽存储前缀长度), 不能复用"更长缓存序列的中间段"。据此, 缓存 key = 稳定
        // 公共前缀 token ids; 命中 = 某缓存槽存的整段前缀是请求 P 的前缀(公共前缀语义), 取最长者;
        // Dp = 该槽的完整存储前缀长度, 从 Dp 续算 tail(state+Answer)。这取代了旧的「候选 A/B 双路径」,
        // 且同任务连发(含相同报文)与跨 state 都能命中。
        const bool want_prefix = decision_.cache_prefix && backend_ && backend_->ok() && Lp >= 2;
        // [TRACE] 诊断: 前缀缓存各环节(KEV_TRACE=1 时打印)。
        const bool PFX_TRACE = std::getenv("KEV_TRACE") != nullptr && std::string(std::getenv("KEV_TRACE")) == "1";
        int Dp = -1;                       // 本问复用前缀长度(<0 = 不使用前缀缓存)
        int pseq = -1;                     // 前缀槽(base 之上); <0 = 无可用/未命中
        std::string slot_hit = "miss";
        int trace_lcp = 0;                 // [TRACE] 命中时复用长度(仅供诊断)
        if (want_prefix) {
            // B2(批处理): LRU prepare + 前缀 prefill 需跨请求串行(避免两请求并发 prefill 同一
            // 淘汰槽). 非批处理已在 compute_mutex_ 内, 无需额外锁(prefix_mutex_ 未呕由此保护).
            std::unique_lock<std::mutex> pfx_lock;
            if (batched) pfx_lock = std::unique_lock<std::mutex>(prefix_mutex_);
            const int min_lcp = decision_.min_lcp_tokens > 0 ? decision_.min_lcp_tokens : 1;
            // 待缓存/匹配的目标前缀 = 模板 {state} 之前的稳定段(渲染 instructions+options)。
            std::vector<int32_t> Pg;
            const size_t sp = tpl.find("{state}");
            if (sp != std::string::npos && tpl.find("{state}", sp + 7) == std::string::npos) {
                std::string pg_str = tpl.substr(0, sp);
                auto rep_pfx = [&](const std::string & ph, const std::string & val) {
                    size_t at = 0;
                    while ((at = pg_str.find(ph, at)) != std::string::npos) { pg_str.replace(at, ph.size(), val); at += val.size(); }
                };
                std::string opts_str_pfx;
                for (size_t oi = 0; oi < q.options.size(); ++oi) { if (oi) opts_str_pfx += ", "; opts_str_pfx += q.options[oi]; }
                rep_pfx("{instructions}", q.instr);
                rep_pfx("{options}", opts_str_pfx);
                Pg = tok(pg_str);
                // BPE 边界校验: Pg 必须是整条 P 的前缀, 否则不能安全复用(跨界)。
                if (!Pg.empty() && (int) Pg.size() < Lp &&
                    std::equal(Pg.begin(), Pg.end(), P.begin(), P.begin() + (int64_t) Pg.size())) {
                    PrefixMatch pm = prefix_lru_.lookup(P, min_lcp);
                    if (pm.hit) {
                        // 命中: 某缓存槽存的整段前缀是 P 的前缀, 整体复用(长度 = 该槽存储长度)。
                        Dp = pm.lcp; pseq = pm.seq; slot_hit = "HIT"; trace_lcp = pm.lcp;
                    } else if ((int) Pg.size() >= min_lcp && pm.evict_seq >= 0) {
                        // 未命中: prefill 稳定公共前缀 Pg 到淘汰槽并登记为新 key; 当前请求也
                        // 从该前缀续算 tail(state+Answer), 避免多一次全量解码。
                        Dp = (int) Pg.size(); slot_hit = "miss";
                        std::vector<std::vector<float>> pre_lg;
                        if (decode_one_to_logits(Pg, 0, pm.evict_seq, /*copy_src=*/-1,
                                                 /*out_pos=*/{}, /*out_cands=*/{}, pre_lg, &err2)) {
                            prefix_lru_.register_(Pg, pm.evict_seq);
                            pseq = pm.evict_seq;
                        }
                    }
                }
            }
        }
        const bool use_prefix = (Dp > 0) && (pseq >= 0) && (Dp < Lp);
        if (PFX_TRACE) {
            std::fprintf(stderr,
                "[PFX][q=%d] Lp=%d want_prefix=%d min_lcp=%d lcp=%d Dp=%d pseq=%d slot=%s use_prefix=%d cachedPrefixes=%d\n",
                k, (int)Lp, (int)want_prefix,
                (want_prefix ? (decision_.min_lcp_tokens > 0 ? decision_.min_lcp_tokens : 1) : 0),
                trace_lcp, Dp, pseq, slot_hit.c_str(), (int)use_prefix,
                (int)prefix_lru_.contents().size());
        }
        // 解码并读候选 logits(批处理: 经 BatchScheduler 单 worker + per-request my_slot,
        // 与并发请求合并进同一次 decode_multi_seq; 串行: compute_mutex_ + my_slot==scratch_seq_)。
        if (!keep) {
            // aggregation == "first"(默认): 单次 decode 末位置 Lp-1 输出全量候选 logits。
            std::vector<std::vector<int32_t>> one_cand(1, cand_first);
            std::vector<std::vector<float>> lg;
            bool okd = false;
            if (use_prefix) {
                // P2 续接: 只解码 state+Answer(跳过重算稳定前缀), copy pseq -> my_slot。
                std::vector<int32_t> tail(P.begin() + Dp, P.end());
                okd = decode_one_to_logits(tail, Dp, my_slot, pseq, {Lp - 1 - Dp}, one_cand, lg, &err2);
            }
            if (!okd) {
                // 回退: 从 pos 0 全新解码整条 P。
                std::vector<std::vector<float>> lg2;
                okd = decode_one_to_logits(P, 0, my_slot, /*copy_src=*/-1, {Lp - 1}, one_cand, lg2, &err2);
                if (okd) lg = std::move(lg2);
                else {
                    // 仍失败: 降级逐候选(结果等价, 不崩溃)。
                    std::fprintf(stderr,
                        "kev_model: logits single-decode failed at q=%d (%s); degrading to per-key decode\n",
                        k, err2.c_str());
                    for (int j = 0; j < K; ++j) {
                        if (cand_first[j] < 0) { z[j] = 0.0f; continue; }
                        std::vector<int32_t> ids = P; ids.push_back(cand_first[j]);
                        std::vector<std::vector<int32_t>> kc(1, std::vector<int32_t>{cand_first[j]});
                        std::vector<std::vector<float>> klg;
                        if (!decode_one_to_logits(ids, 0, my_slot, -1, {Lp - 1}, kc, klg, &err2)) {
                            std::fprintf(stderr,
                                "kev_model: degraded logits seq decode failed at q=%d,key=%d (%s)\n",
                                k, j, err2.c_str());
                            if (err && err->empty()) *err = err2;
                            return res;
                        }
                        z[j] = klg.empty() || klg[0].empty() ? 0.0f : klg[0][0];
                    }
                }
            }
            if (okd && !lg.empty() && !lg[0].empty()) {
                for (int j = 0; j < K; ++j) {
                    if (cand_first[j] < 0) { z[j] = 0.0f; continue; }
                    z[j] = lg[0][j];
                }
            }
        } else {
            // "sum"/"mean": 逐候选 decode(P + 全 key token), 读每 token 位置 logits 并聚合。
            for (int j = 0; j < K; ++j) {
                const int32_t M = (int32_t) cand_tok[j].size();
                if (M == 0) { z[j] = 0.0f; continue; }
                std::vector<int32_t> ids;
                std::vector<int32_t> out_global;
                int32_t start_pos; int copy_src;
                if (use_prefix) {
                    ids.insert(ids.end(), P.begin() + Dp, P.end());
                    for (int m = 0; m < M; ++m) out_global.push_back((Lp - 1 - Dp) + m);
                    ids.insert(ids.end(), cand_tok[j].begin(), cand_tok[j].end());
                    start_pos = Dp; copy_src = pseq;
                } else {
                    ids = P;
                    for (int m = 0; m < M; ++m) out_global.push_back(Lp - 1 + m);
                    ids.insert(ids.end(), cand_tok[j].begin(), cand_tok[j].end());
                    start_pos = 0; copy_src = -1;
                }
                std::vector<std::vector<int32_t>> m_cands(M);
                for (int m = 0; m < M; ++m) m_cands[m] = { cand_tok[j][m] };
                std::vector<std::vector<float>> mlg;
                if (!decode_one_to_logits(ids, start_pos, my_slot, copy_src, out_global, m_cands, mlg, &err2)) {
                    std::fprintf(stderr, "kev_model: logits seq decode failed (%s)\n", err2.c_str());
                    if (err && err->empty()) *err = err2;
                    return res;
                }
                float s = 0;
                for (int m = 0; m < M; ++m) s += (mlg[m].empty() ? 0.0f : mlg[m][0]);
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

// ---- 调研实验: hidden 聚合判别通道(零训练) ----
// decode + 在多个位置读 hidden 表示向量。
bool KevModel::decode_one_to_hidden(const std::vector<int32_t> & tokens, int32_t start_pos,
                                    int32_t seq, int32_t copy_src,
                                    const std::vector<int32_t> & out_pos,
                                    std::vector<std::vector<float>> & hidden,
                                    std::string * err) {
    hidden.assign(out_pos.size(), {});
    if (!backend_ || !backend_->ok()) { if (err && err->empty()) *err = "backend not initialized"; return false; }
    const int32_t n_embd = backend_->model_n_embd();
    if (n_embd <= 0) { if (err && err->empty()) *err = "hidden channel needs embeddings (read_hidden) enabled"; return false; }

    std::string e2;
    (void) backend_->seq_clear(seq);
    if (copy_src >= 0 && !backend_->seq_copy(copy_src, seq)) { if (err && err->empty()) *err = "seq_copy failed"; return false; }
    std::vector<int32_t> out_local;
    const int chunk = backend_->n_batch() > 0 ? backend_->n_batch() : 512;
    if (!decode_seq_chunked(tokens, start_pos, seq, &e2, out_pos, out_local, chunk)) {
        if (err && err->empty()) *err = e2;
        return false;
    }
    for (size_t o = 0; o < out_local.size(); ++o) {
        auto h = backend_->read_hidden(out_local[o], &e2);
        if (h.empty()) { if (err && err->empty()) *err = e2.empty() ? std::string("read_hidden failed") : e2; return false; }
        hidden[o] = std::move(h);
    }
    return true;
}

// 计算填充式 P 里 {state} 占位替换所得文本的 token 区间 [start,end)。
// P 由 render_prompt 得到(已把 {state} 换成 req.state)。此处复刻前缀/后缀替换以确定边界。
namespace {
void hidden_state_span(const std::string & tpl, const KevQuestion & q, const KevRequest & req,
                       const std::vector<std::string> & opts, const TokenFeeder & tf,
                       std::vector<int32_t> & P,
                       int * state_start, int * state_end) {
    auto fill = [&](std::string s) {
        auto rep = [&](const char * ph, const std::string & val) {
            size_t at = 0;
            while ((at = s.find(ph, at)) != std::string::npos) { s.replace(at, std::strlen(ph), val); at += val.size(); }
        };
        std::string opts_str;
        for (size_t i = 0; i < opts.size(); ++i) { if (i) opts_str += ", "; opts_str += opts[i]; }
        rep("{instructions}", q.instr);
        rep("{options}", opts_str);
        return s;
    };
    const size_t sp = tpl.find("{state}");
    std::string prefix_s = fill(tpl.substr(0, sp));      // {state} 之前(含 instructions/options)
    std::string prefix_state = prefix_s + req.state;     // prefix + state(即 {state} 替换后截止)
    auto strip = [&](std::vector<int32_t> & v) { while (!v.empty() && v.back() < 0) v.pop_back(); };
    std::vector<int32_t> Pfx = kev::tokenize(tf, prefix_s, /*add_special=*/false, /*parse_special=*/false); strip(Pfx);
    std::vector<int32_t> PfxSt = kev::tokenize(tf, prefix_state, /*add_special=*/false, /*parse_special=*/false); strip(PfxSt);
    *state_start = (int) Pfx.size();
    *state_end   = (int) PfxSt.size();
    // 与 render_prompt 产生的整条 P 对齐; 若 P 长度小于约定(异常), 夹紧。
    if (*state_end > (int) P.size()) { *state_end = (int) P.size(); if (*state_start > *state_end) *state_start = *state_end; }
}
} // namespace

KevModel::ProbsResult KevModel::probs_hidden(const KevRequest & req, bool use_state_agg, std::string * err) {
    ProbsResult res;
    double t0 = now_ms();
    const int Q = (int) req.questions.size();
    if (!backend_ || !backend_->ok()) { if (err && err->empty()) *err = "backend not initialized"; return res; }
    if (Q == 0) { if (err && err->empty()) *err = "no questions"; return res; }

    const TokenFeeder tf = backend_->token_feeder();
    const std::string & tpl = decision_.prompt_template;
    const int n_embd = backend_->model_n_embd();
    if (n_embd <= 0) { if (err && err->empty()) *err = "hidden channel unavailable (embeddings disabled)"; return res; }

    // 串行路径: 整个函数持有 compute_mutex_(实验通道不接入 batch 调度器, 简单可靠)。
    std::lock_guard<std::mutex> lock(compute_mutex_);
    auto tok = [&](const std::string & t) {
        std::lock_guard<std::mutex> lk(tokenizer_mutex_);
        return kev::tokenize(tf, t, /*add_special=*/false, /*parse_special=*/false);
    };

    std::vector<std::vector<float>> probs(Q);
    std::vector<double> qms(Q, 0.0);

    // 余弦相似度 query(决定向量或聚合向量) vs 各候选续写末尾 hidden。
    auto cos_ = [&](const std::vector<float> & a, const std::vector<float> & b) {
        if (a.size() != b.size() || a.empty()) return 0.0f;
        double na = 0, nb = 0, d = 0;
        for (size_t i = 0; i < a.size(); ++i) { d += (double) a[i] * b[i]; na += (double) a[i] * a[i]; nb += (double) b[i] * b[i]; }
        double den = std::sqrt(na) * std::sqrt(nb);
        return (float)(den > 0 ? d / den : 0.0);
    };

    for (int k = 0; k < Q; ++k) {
        const KevQuestion & q = req.questions[k];
        const int K = (int) q.keys.size();
        if (K == 0) { if (err && err->empty()) *err = "question " + std::to_string(k) + ": no candidate keys"; return res; }
        std::string prompt = render_prompt(tpl, q, req, q.options);
        std::vector<int32_t> P = tok(prompt);
        while (!P.empty() && P.back() < 0) P.pop_back();
        const int32_t Lp = (int32_t) P.size();
        if (Lp == 0) { if (err && err->empty()) *err = "question " + std::to_string(k) + ": empty prompt"; return res; }

        double tq0 = now_ms();
        // 决定向量 = 最后一个位置(prompt 末尾, 'Answer:' 之后)的 hidden, 已对全文注意力聚合。
        std::vector<std::vector<float>> hd;
        std::string e2;
        if (!decode_one_to_hidden(P, 0, scratch_seq_, -1, {Lp - 1}, hd, &e2)) {
            if (err && err->empty()) *err = e2; return res;
        }
        std::vector<float> h_decide = hd.empty() ? std::vector<float>() : hd[0];

        // 可选: 用 h_decide 对 {state} 的 token hidden 做软注意力加权聚合 -> h_agg。
        std::vector<float> query = h_decide;
        if (use_state_agg) {
            int st0 = 0, st1 = 0;
            hidden_state_span(tpl, q, req, q.options, tf, P, &st0, &st1);
            if (st1 > st0 && st0 >= 0 && st1 <= (int) Lp) {
                // 取 state 末尾至多 kStateMaxTok 个 token(控制读取成本), 读取各自 hidden。
                const int kStateMaxTok = 256;
                int take_start = std::max(st0, st1 - kStateMaxTok);
                std::vector<int32_t> state_pos;
                for (int p = take_start; p < st1; ++p) state_pos.push_back(p);
                std::vector<std::vector<float>> hs;
                if (!decode_one_to_hidden(P, 0, scratch_seq_, -1, state_pos, hs, &e2)) { if (err && err->empty()) *err = e2; return res; }
                // 软注意力: w_t = softmax( (h_decide·h_t)/sqrt(d) ) over state tokens;
                // 数值稳定缩放到余弦相似度区间。
                double mx = -1e30; std::vector<double> att(hs.size(), 0.0);
                const double scale = 1.0 / std::sqrt((double) n_embd);
                for (size_t t = 0; t < hs.size(); ++t) {
                    double dp = 0;
                    for (int i = 0; i < n_embd && i < (int) h_decide.size() && i < (int) hs[t].size(); ++i) dp += (double) h_decide[i] * hs[t][i];
                    att[t] = dp * scale; if (att[t] > mx) mx = att[t];
                }
                for (size_t t = 0; t < hs.size(); ++t) att[t] = std::exp(att[t] - mx);
                double zn = 0; for (double a : att) zn += a; if (zn > 0) for (double & a : att) a /= zn;
                query.assign(n_embd, 0.0f);
                for (size_t t = 0; t < hs.size(); ++t)
                    for (int i = 0; i < n_embd && i < (int) hs[t].size(); ++i) query[i] += (float)(att[t] * hs[t][i]);
            }
        }

        // 每候选: 解码 P + 完整候选 token, 读末尾 hidden(该候选续写后的表示)。
        // 评分 = 中心化后的余弦(query, h_cand)。原始高维残差流里各候选共享一大片"通用 LM
        // 激活"方向, 导致 raw cos 全部近同(实测近 0.5/0.5 均布)。故先对各候选 hidden 求质心,
        // 把 query 与各候选都减去质心, 只保留"候选相对彼此差异 + 该差异与 query 的对齐"。
        std::vector<std::vector<float>> HC(K);
        std::vector<float> centroid(n_embd, 0.0f);
        bool anyCand = false;
        for (int j = 0; j < K; ++j) {
            std::vector<int32_t> cand = tok(q.keys[j]);
            while (!cand.empty() && cand.back() < 0) cand.pop_back();
            std::vector<int32_t> ids = P; ids.insert(ids.end(), cand.begin(), cand.end());
            int out_pos = ids.empty() ? (int) Lp - 1 : (int) ids.size() - 1;
            std::vector<std::vector<float>> hc;
            if (!decode_one_to_hidden(ids, 0, scratch_seq_, -1, {out_pos}, hc, &e2)) { HC[j].clear(); continue; }
            HC[j] = hc.empty() ? query : hc[0];
            if ((int) HC[j].size() == n_embd) {
                anyCand = true;
                for (int i = 0; i < n_embd; ++i) centroid[i] += HC[j][i];
            } else { HC[j].clear(); }
        }
        if (anyCand) for (int i = 0; i < n_embd; ++i) centroid[i] /= (float) K;

        std::vector<float> z(K, 0.0f);
        for (int j = 0; j < K; ++j) {
            if ((int) HC[j].size() != n_embd) { z[j] = 0.0f; continue; }
            // query 与候选都减去候选质心(只取相对差异方向)。
            std::vector<float> qc(n_embd), kc(n_embd);
            for (int i = 0; i < n_embd; ++i) { qc[i] = query[i] - centroid[i]; kc[i] = HC[j][i] - centroid[i]; }
            z[j] = cos_(qc, kc);
            if (std::isnan(z[j])) z[j] = 0.0f;
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
    std::lock_guard<std::mutex> lock(compute_mutex_);
    if (!backend_) return 0;
    return (int) kev::tokenize(backend_->token_feeder(), text,
                               /*add_special=*/false, /*parse_special=*/false).size();
}

} // namespace kev
