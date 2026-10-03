// cold_profile.cpp — 受控性能计时探针: 隔离 "cold 请求" 的时间在 图构建/调度固定开销 vs 前向带宽。
//
// 目标形态: state prefill(一次 llama_decode, 38 tok) + branch decode(一次 llama_decode, 89 tok),
// cold 总 ~650ms, 但只估了 "~400ms 带宽 + ~250ms 图构建/调度", 没实测隔离。本探针把它拆开。
//
// 独立 bench: 直接用 llama C API(llama_model_load_from_file / llama_init_from_model / llama_batch_init /
// llama_decode), 不经过 KevModel 私有方法/互斥锁/LRU/head —— 只测 llama 本身。
//
// 用法:
//   cold_profile --model <gguf> [--lora <gguf>] [--ctx 8192] [--threads T]
//                [--n 38] [--n2 89] [--reps 5] [--warmup 8] [--list]
//   --model    必需
//   --lora     可选 LoRA 适配器
//   --ctx      context 大小(默认 8192; 图大小随 n_ctx 变, 想贴近生产就用生产值)
//   --threads  线程数(默认自动=逻辑核, 上限 32, 同 kev_model)
//   --n        单点/分段模式第一段 token 数(默认 38, 对标 state)
//   --n2       第二段 token 数(默认 89, 对标 branch; 设为 0 则只测单点)
//   --reps     每档同形状重复次数(默认 5, 取 best)
//   --warmup   内循环前的预热 decode token 数(默认 8, 拉稳 TLB/frequency)
//   --list     打印内置 n 曲线(1..256, 含 38/89)的 n vs best/first/mean
//
// 计时对象(steady_clock, ms; 同 kev_model.cpp 的 now_ms()):
//   A. first_ms — 新起 llama_context 后第一次 llama_decode(n tokens) = build_graph+sched_alloc+compute
//                  (cold 的"图构建"部分)。
//   B. best/mean_ms — 同一 context 里对完全相同的形状(n tokens, 每次都清空 KV 后从 pos0 重解)重复 decode,
//                      取 best/mean。can_reuse 命中 => repeat≈纯 compute(快); 否则每次重建 => 也含建图(慢),
//                      用于分辨"图重建是否每次发生"。
//   ctx_ms — context 初始化(内存分配)单独报, 不混入逐 decode 数据。
//
// 输出为易解析行, 便于最小二乘(n vs best_ms):
//   单点:   point n=38 ctx_ms=188.0 first_ms=205.3 best_ms=37.2 mean_ms=42.1
//   分段:   seg_state n=38 first_ms=205.3 best_ms=37.2 mean_ms=42.1
//           seg_branch n=89 after_state_ms=96.7 best_ms=55.4 mean_ms=60.0
//           cold_est state+first  + after_state = 302.0   (真实 cold 两段在同 ctx 连解)
//           cold_est graph_overhead = first-best (n=38) + (after_state-best) (n=89) 之类的差值, 见下
//   曲线:   list n=1 first_ms=... best_ms=... mean_ms=...
//
// 读法: best(n) 斜率 ≈ 每 token 带宽; (first-best) 在中小 n ≈ 图构建+sched_alloc 固定开销
//       (它不随 n 变); repeat best≈mean 说明 can_reuse 命中图复用, best≈first 说明每次都重建图。
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace {

double now_ms() {
    return (double) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() / 1e3;
}

struct Options {
    std::string model_path;
    std::string lora_path;
    int   ctx = 8192;
    int   threads = 0;    // 0 => 自动
    int   n = 38;         // 第一段(对标 state)
    int   n2 = 89;        // 第二段(对标 branch); 0 => 只测单点
    int   reps = 5;
    int   warmup = 8;
    bool  list = false;
};

// 内建 n 曲线(含实际 cold 用到的 38 / 89)。
static const int k_list_n[] = {1, 2, 4, 8, 16, 32, 38, 64, 89, 128, 192, 256};
static const int k_list_count = (int) (sizeof(k_list_n) / sizeof(k_list_n[0]));

// 在一次 decode 内把所有 batch 成员填好(token/pos/seq_id/logits), 照抄 kev_model.cpp decode_seq 的
// llama_batch_init 用法, 但这里是独立 C API(纯 prefill, logits 全 0)。
static bool decode_n(llama_context * ctx, const std::vector<int32_t> & toks, int64_t pos0) {
    const int32_t n = (int32_t) toks.size();
    if (n == 0) return false;
    llama_batch batch = llama_batch_init(n, /*embd=*/0, /*n_seq_max=*/1);
    for (int32_t i = 0; i < n; ++i) {
        batch.token[i]     = (llama_token) toks[(size_t) i];
        batch.pos[i]       = (llama_pos)(pos0 + i);
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 0;   // 纯 prefill, 无输出(对标 decode_seq 默认 logits 集合为空)
    }
    batch.n_tokens = n;
    const int r = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return r == 0;
}

// 通用重复计时: 在 ctx 上对完全相同形状(n tokens)做 reps 次 decode, 每次先清空 seq0 KV 再从 pos0 重解,
// 使每次形状/位置完全一致, 最干净地暴露 can_reuse 图复用与否。返回 best/mean。
static bool run_repeat(llama_context * ctx, llama_memory_t mem, int n, int reps,
                       double * best_ms, double * mean_ms) {
    // 构造 n 个合法通配 token(重复 id=1)。token 值不影响图形状/耗时, 只需要数量=n 且位置连续。
    std::vector<int32_t> toks((size_t) n, 1);
    *best_ms = 1e18;
    double sum = 0;
    int done = 0;
    for (int r = 0; r < reps; ++r) {
        llama_memory_seq_rm(mem, 0, 0, -1);   // 清空 seq0 => 每次从空 KV、pos0 开始, 形状固定
        double t0 = now_ms();
        bool ok = decode_n(ctx, toks, /*pos0=*/0);
        double dt = now_ms() - t0;
        if (!ok) return false;
        if (dt < *best_ms) *best_ms = dt;
        sum += dt;
        done++;
    }
    if (done == 0) return false;
    *mean_ms = sum / (double) done;
    return true;
}

struct CtxBundle {
    llama_context * ctx = nullptr;
    llama_memory_t mem = nullptr;
    double ctx_ms = 0.0;   // 仅 context 初始化(含内存分配 + lora set, 不含任何 decode)
};

// 新起一个 context(可选挂 lora), 返回打开时的耗时与句柄。
static bool make_ctx(llama_model * model, llama_adapter_lora * lora, const Options & opt,
                     int need_tokens, CtxBundle & out) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                  = (uint32_t)(opt.ctx > 0 ? opt.ctx : 4096);
    // 保证一次 decode 至少容纳 need_tokens; n_batch/n_ubatch 放宽到 >n 使单 ubatch 一次处理完, 无切批开销。
    cparams.n_batch                = (uint32_t) std::max(need_tokens, 256);
    cparams.n_ubatch               = (uint32_t) std::max(need_tokens, 128);
    cparams.n_seq_max              = 4;
    cparams.n_outputs_max_per_seq  = 128;    // A2(M7) 同款, 省输出缓冲
    cparams.embeddings             = true;
    cparams.lm_head                = false;  // Rank-1: this profiler only reads embeddings, never logits
    cparams.pooling_type           = LLAMA_POOLING_TYPE_NONE;
    cparams.n_threads              = (int32_t) opt.threads;
    cparams.n_threads_batch        = (int32_t) opt.threads;

    double t0 = now_ms();
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) return false;
    if (lora) {
        float scale = 1.0f;
        if (llama_set_adapters_lora(ctx, &lora, 1, &scale) != 0) {
            llama_free(ctx);
            return false;
        }
    }
    out.ctx = ctx;
    out.mem = llama_get_memory(ctx);
    out.ctx_ms = now_ms() - t0;
    return true;
}

// 单个点: 新 context 的 first decode(cold 建图) + 同形状 repeat(best/mean)。
struct Point {
    int n = 0;
    double ctx_ms = 0.0, first_ms = 0.0, best_ms = 0.0, mean_ms = 0.0;
    bool ok = false;
};

static Point sample_point(llama_model * model, llama_adapter_lora * lora, const Options & opt, int n) {
    Point p;
    p.n = n;
    CtxBundle b;
    if (!make_ctx(model, lora, opt, n, b)) return p;
    p.ctx_ms = b.ctx_ms;
    std::vector<int32_t> toks((size_t) n, 1);
    // A. cold: 新 context 第一次 decode(建图+sched_alloc+compute)。
    double t0 = now_ms();
    bool ok = decode_n(b.ctx, toks, /*pos0=*/0);
    p.first_ms = now_ms() - t0;
    if (!ok) { llama_free(b.ctx); return p; }
    // B. 同形状重复。
    if (!run_repeat(b.ctx, b.mem, n, opt.reps, &p.best_ms, &p.mean_ms)) { llama_free(b.ctx); return p; }
    llama_free(b.ctx);
    p.ok = true;
    return p;
}

// 分段模式(对标真实 cold: 同一 context 里 state(38) 后再 branch(89), 位置连续)。
struct Segment {
    int n_state = 0, n_branch = 0;
    double ctx_ms = 0.0;
    double state_first_ms = 0.0;   // 38 的 cold first
    double state_best_ms = 0.0;    // 38 同形状 best
    double branch_after_ms = 0.0;  // 89 紧跟 state 之后(位置 38..126)的第二次 decode
    double branch_best_ms = 0.0;   // 89 同形状 best
    double branch_mean_ms = 0.0;
    bool ok = false;
};

static Segment sample_segment(llama_model * model, llama_adapter_lora * lora, const Options & opt) {
    Segment s;
    s.n_state = opt.n;
    s.n_branch = opt.n2;
    CtxBundle b;
    if (!make_ctx(model, lora, opt, std::max(opt.n, opt.n2), b)) return s;
    s.ctx_ms = b.ctx_ms;

    // state cold / 从空 KV 建图 + compute。
    std::vector<int32_t> stok((size_t) opt.n, 1);
    double t0 = now_ms();
    bool ok = decode_n(b.ctx, stok, /*pos0=*/0);
    s.state_first_ms = now_ms() - t0;
    if (!ok) { llama_free(b.ctx); return s; }

    // state 同形状 best(清空 KV 重解, 测 38 的稳定 forward)。
    double _mean = 0.0;
    if (!run_repeat(b.ctx, b.mem, opt.n, opt.reps, &s.state_best_ms, &_mean)) { llama_free(b.ctx); return s; }

    // branch: 不清理, 直接续在 state(38) 之后(位置 38..126), 与真实 cold 一致。
    std::vector<int32_t> btok((size_t) opt.n2, 1);
    t0 = now_ms();
    ok = decode_n(b.ctx, btok, /*pos0=*/opt.n);
    s.branch_after_ms = now_ms() - t0;
    if (!ok) { llama_free(b.ctx); return s; }

    // branch 同形状 best(清空后测 89 的稳定 forward)。
    if (!run_repeat(b.ctx, b.mem, opt.n2, opt.reps, &s.branch_best_ms, &s.branch_mean_ms)) { llama_free(b.ctx); return s; }

    llama_free(b.ctx);
    s.ok = true;
    return s;
}

} // namespace

int main(int argc, char ** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
             if (a == "--model")   opt.model_path = next();
        else if (a == "--lora")    opt.lora_path = next();
        else if (a == "--ctx")     opt.ctx = std::atoi(next().c_str());
        else if (a == "--threads") opt.threads = std::atoi(next().c_str());
        else if (a == "--n")       opt.n = std::atoi(next().c_str());
        else if (a == "--n2")      opt.n2 = std::atoi(next().c_str());
        else if (a == "--reps")    opt.reps = std::atoi(next().c_str());
        else if (a == "--warmup")  opt.warmup = std::atoi(next().c_str());
        else if (a == "--list")    opt.list = true;
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (opt.model_path.empty()) {
        std::fprintf(stderr, "usage: cold_profile --model <gguf> [--lora gguf] [--ctx N] [--threads T]"
                             " [--n 38] [--n2 89] [--reps 5] [--warmup 8] [--list]\n");
        return 2;
    }
    if (opt.reps < 1) opt.reps = 1;
    if (opt.n < 1) opt.n = 1;
    if (opt.n2 < 0) opt.n2 = 0;
    if (opt.ctx > 0 && opt.n > opt.ctx) { std::fprintf(stderr, "--n > --ctx; clamp n\n"); opt.n = opt.ctx; }

    // 线程默认: 同 kev_model, 自动用逻辑核, 上限 32。
    if (opt.threads <= 0) {
        opt.threads = (int) std::thread::hardware_concurrency();
        if (opt.threads <= 0) opt.threads = 1;
        if (opt.threads > 32) opt.threads = 32;
    }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(opt.model_path.c_str(), mparams);
    if (!model) { std::fprintf(stderr, "failed to load model: %s\n", opt.model_path.c_str()); return 1; }

    llama_adapter_lora * lora = nullptr;
    if (!opt.lora_path.empty()) {
        lora = llama_adapter_lora_init(model, opt.lora_path.c_str());
        if (!lora) { std::fprintf(stderr, "failed to load lora: %s\n", opt.lora_path.c_str()); return 1; }
    }

    std::printf("# cold_profile: threads=%d ctx=%d n=%d n2=%d reps=%d%s\n",
                opt.threads, opt.ctx, opt.n, opt.n2, opt.reps,
                opt.lora_path.empty() ? "" : " lora=1");

    // 预热: 一次足够大的 decode, 把 FMA/TLB/frequency 拉稳(不计时)。
    {
        CtxBundle w;
        if (make_ctx(model, lora, opt, opt.warmup, w)) {
            std::vector<int32_t> wto((size_t) opt.warmup, 1);
            decode_n(w.ctx, wto, 0);
            llama_free(w.ctx);
        }
    }

    if (opt.list) {
        std::printf("list  n first_ms best_ms mean_ms\n");
        for (int k = 0; k < k_list_count; ++k) {
            int n = k_list_n[k];
            if (n > opt.ctx) break;
            Point p = sample_point(model, lora, opt, n);
            if (!p.ok) { std::fprintf(stderr, "sample failed n=%d\n", n); continue; }
            std::printf("list n=%d first_ms=%.3f best_ms=%.3f mean_ms=%.3f ctx_ms=%.3f\n",
                        p.n, p.first_ms, p.best_ms, p.mean_ms, p.ctx_ms);
        }
    } else if (opt.n2 > 0) {
        // 分段模式: 真实 cold 形态(state 后 branch 续帧)。
        Segment s = sample_segment(model, lora, opt);
        if (!s.ok) { std::fprintf(stderr, "segment failed\n"); return 1; }
        std::printf("seg_state  n=%d first_ms=%.3f best_ms=%.3f ctx_ms=%.3f\n",
                    s.n_state, s.state_first_ms, s.state_best_ms, s.ctx_ms);
        std::printf("seg_branch n=%d after_state_ms=%.3f best_ms=%.3f mean_ms=%.3f\n",
                    s.n_branch, s.branch_after_ms, s.branch_best_ms, s.branch_mean_ms);
        // 两个独立冷请求上界(first 两段都从空建图) vs 真实同 ctx 续解。
        double sum_first = s.state_first_ms + s.branch_after_ms;      // 同 ctx 续解, 最贴近 cold
        double sum_best  = s.state_best_ms + s.branch_best_ms;        // 纯 forward(图已复用)
        // 固定图开销估计(state/state_best 在 cold 段 ctx 上近似; branch 用 after_state - best)。
        double graph_overhead = (s.state_first_ms - s.state_best_ms) +
                                (s.branch_after_ms - s.branch_best_ms);
        std::printf("cold_est   state_1st=%+.3f branch_after=%+.3f sum=%.3f\n",
                    s.state_first_ms, s.branch_after_ms, sum_first);
        std::printf("cold_est   sum_first=%.3f  sum_best=%.3f  graph_overhead_first_vs_best=%.3f\n",
                    sum_first, sum_best, graph_overhead);
        std::printf("cold_est   can_reuse: branch best(%.3f) vs after(%.3f) -> %s\n",
                    s.branch_best_ms, s.branch_after_ms,
                    (s.branch_best_ms < s.branch_after_ms * 0.9) ? "partial" : "no-critical");
    } else {
        // 单点模式。
        Point p = sample_point(model, lora, opt, opt.n);
        if (!p.ok) { std::fprintf(stderr, "sample failed n=%d\n", opt.n); return 1; }
        std::printf("point n=%d ctx_ms=%.3f first_ms=%.3f best_ms=%.3f mean_ms=%.3f\n",
                    p.n, p.ctx_ms, p.first_ms, p.best_ms, p.mean_ms);
        std::printf("point  graph_overhead(first-best)=%.3f ms\n", p.first_ms - p.best_ms);
    }

    if (lora) llama_adapter_lora_free(lora);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}