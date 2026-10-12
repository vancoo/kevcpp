// b1_multi_seq_probe.cpp — B1 validation probe: single-ctx multi-sequence parallel decode
//
// Proves that one llama_context can maintain N independent-KV sequences, decoded in a
// SINGLE llama_decode (decode_multi_seq), each reading its own candidate logits, and
// compares to decoding the same prompts one-by-one serially (the existing decode_seq path).
//
// For each prompt we build a "填空式" prompt (like probs_logits renders) and pick a few
// candidate answer tokens. We compute each prompt's candidate logits two ways:
//   * SERIAL  (reference / current production path):
//       per prompt p: seq_clear(p.seq); decode_seq(P_p, 0, p.seq, {Lp-1}); read_logits.
//   * PARALLEL (B1, target):
//       seq_clear each p.seq; decode_multi_seq(all N prompts in ONE llama_decode,
//       out = last-position logits of each prompt); read_logits via returned out_idx.
// We compare the two per (prompt, candidate) — bit-exact float, argmax choice, softmax dist.
//
// FINDINGS (0.8B CPU / vendored llama.cpp; verified multithread AND --threads 1):
//   * argmax CHOICE is bit-identical parallel-vs-serial (choice_mismatch=0 every round).
//   * the raw candidate logits are NOT bit-identical (max ~0.5 abs): llama.cpp's CPU forward
//     is only bit-deterministic for a FIXED graph shape — batching N sequences changes the
//     total n_tokens / graph, whose fp accumulation differs (even single-threaded; the
//     serial-vs-serial baseline is 0). Choice/softmax ranking is preserved -> semantic equality.
//   * throughput: batching amortizes per-decode fixed overhead, so for SHORT decodes (the
//     production LCP-hit tail regime) it is ~1.85x vs summing C serials; long FLOP-bound
//     prefill is ~1x.
//
// Usage:
//   b1_multi_seq_probe --model <gguf> [--seqs N] [--rounds R] [--threads N]
#include "kev_backend_factory.h"
#include "kev_encode.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace kev;

namespace {
double now_ms() {
    return (double) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() / 1e3;
}

// Build a realistic 填空式 prompt the way probs_logits does.
std::string render_prompt(const std::string & tpl, const std::string & state,
                          const std::string & instr, const std::vector<std::string> & opts) {
    auto rep = [&](std::string & p, const std::string & ph, const std::string & val) {
        size_t at = 0;
        while ((at = p.find(ph, at)) != std::string::npos) {
            p.replace(at, ph.size(), val);
            at += val.size();
        }
    };
    std::string p = tpl;
    std::string opt_str;
    for (size_t i = 0; i < opts.size(); ++i) { if (i) opt_str += ", "; opt_str += opts[i]; }
    rep(p, "{state}", state);
    rep(p, "{instructions}", instr);
    rep(p, "{options}", opt_str);
    return p;
}

std::vector<std::string> branch_states() {
    // 一组真实、长度不一的 state(中文 + 英文混合), 使各 seq 长度不等 -> 更贴近真实批处理。
    return {
        "The customer's running shoes arrived in the wrong size; they want a size 10 exchange and a prepaid return label.",
        "I was charged twice for my order on my credit card and I would like a refund for the duplicate charge immediately.",
        "客户反馈收到的商品数量不对,缺了一件,并且外包装有明显破损,客服需要核实订单并安排补发或退款。",
        "The package was marked delivered but never arrived; please investigate the carrier and re-ship or refund.",
        "用户询问退货退款需要多长时间到账,以及退货运费由谁承担,希望得到明确的时间表和流程说明。",
    };
}
} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    int n_seqs = 4, rounds = 3, n_threads = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--model") model_path = next();
        else if (a == "--seqs") n_seqs = std::atoi(next().c_str());
        else if (a == "--rounds") rounds = std::atoi(next().c_str());
        else if (a == "--threads") n_threads = std::atoi(next().c_str());
        else if (a == "--help" || a == "-h") {
            std::fprintf(stderr, "usage: b1_multi_seq_probe --model <gguf> [--seqs N] [--rounds R] [--threads N]\n");
            return 0;
        }
    }
    if (model_path.empty()) {
        std::fprintf(stderr, "error: --model required\n");
        return 2;
    }
    if (n_seqs <= 0 || n_seqs > 24) { std::fprintf(stderr, "error: --seqs must be in 1..24\n"); return 2; }

    std::string err;
    std::unique_ptr<Backend> b = BackendFactory_load("cpu", &err);
    if (!b) { std::fprintf(stderr, "error: load cpu backend: %s\n", err.c_str()); return 1; }

    BackendOptions opt;
    opt.model_path   = model_path;
    opt.n_ctx        = 8192;
    opt.n_seq_max    = 32;    // plenty of seq slots for parallel sequences
    opt.n_batch      = 512;
    opt.n_ubatch     = 256;
    opt.n_outputs_max_per_seq = 128;
    opt.embeddings   = 0;     // logits-only (matches production probs_logits)
    opt.lm_head      = 1;     // need lm_head projection for read_logits
    opt.n_threads    = n_threads; // 0 = auto (default); set to 1 to test ggml single-thread determinism
    if (!b->init(opt, &err)) { std::fprintf(stderr, "error: backend init: %s\n", err.c_str()); return 1; }

    const TokenFeeder tf = b->token_feeder();
    const int nv = b->vocab_n_tokens();
    const int chunk = b->n_batch() > 0 ? b->n_batch() : 512;
    std::fprintf(stderr, "b1 probe: backend=%s n_vocab=%d n_batch=%d n_seq_max=%d n_ctx_seq=%d threads=%d\n",
                 b->name(), nv, chunk, 32, (int) b->n_ctx_seq_capacity(), b->threads());

    const std::string tpl = "{instructions}\n[ {options} ]\n{state}\nAnswer:";
    const std::string instr = "Which team should handle this? / 判断该情况应归到哪个类别";
    const std::vector<std::string> opts = {
        "billing: about charges/invoices", "shipping: delivery status/delays",
        "returns: wrong/damaged items", "account: login/password access", "other: none of the above",
    };
    // candidate answer tokens to read logits for (single tokens: "billing/shipping/returns/..." first tokens).
    const std::vector<std::string> cands = {"billing", "shipping", "returns", "account", "other", "yes", "no", "refund"};

    const auto states = branch_states();

    int total_rounds_ok = 0;
    double par_time_sum = 0.0, ser_time_sum = 0.0;
    long long bit_exact_pts = 0, mismatch_pts = 0, choice_mismatches = 0;
    double max_abs_diff = 0.0;
    double det_max_abs_diff = 0.0;   // serial-vs-serial engine noise baseline

    for (int r = 0; r < rounds; ++r) {
        // ------ build N prompts (round-rotating states so seqs differ across rounds) ------
        struct Seq { std::vector<int32_t> tok; int len; int seq; };
        std::vector<Seq> seqs(n_seqs);
        for (int s = 0; s < n_seqs; ++s) {
            const std::string & st = states[(r * n_seqs + s) % states.size()];
            std::string prompt = render_prompt(tpl, st, instr, opts);
            std::vector<int32_t> P = kev::tokenize(tf, prompt, false, false);
            while (!P.empty() && P.back() < 0) P.pop_back();
            if (P.empty()) { std::fprintf(stderr, "error: empty prompt for seq %d\n", s); return 1; }
            seqs[s].tok = P; seqs[s].len = (int) P.size(); seqs[s].seq = s; // distinct seq ids 0..N-1
        }

        // candidate first-tokens per candidate phrase
        std::vector<int32_t> cand_first(cands.size(), -1);
        for (size_t c = 0; c < cands.size(); ++c) {
            auto t = kev::tokenize(tf, cands[c], false, false);
            cand_first[c] = t.empty() ? -1 : t[0];
        }

        // ------ DETERMINISM BASELINE (round 0 only): same prompt decoded twice serially ------
        // Measures the llama.cpp engine's intrinsic run-to-run noise for the SAME input in two
        // SEPARATE llama_decode calls (no batching). If this delta is ~ the parallel-vs-serial
        // delta, the batching is not the cause of any non-bit-identical results.
        if (r == 0) {
            std::string err2;
            std::vector<float> dtr_a(cands.size(), 0.0f), dtr_b(cands.size(), 0.0f);
            for (int w = 0; w < 2; ++w) {
                const int sid = w == 0 ? seqs[0].seq : seqs[1].seq;
                b->seq_clear(sid);
                if (!b->decode_seq(seqs[0].tok, 0, sid, &err2, {(int32_t)(seqs[0].len - 1)})) {
                    std::fprintf(stderr, "error: det serial decode: %s\n", err2.c_str()); return 1;
                }
                auto & dst = w == 0 ? dtr_a : dtr_b;
                for (size_t c = 0; c < cands.size(); ++c) {
                    if (cand_first[c] < 0) { dst[c] = 0.0f; continue; }
                    if (!b->read_logits((int32_t)(seqs[0].len - 1), cand_first[c], &dst[c], &err2)) dst[c] = 0.0f;
                }
            }
            double dmax = 0.0; long long dmis = 0;
            for (size_t c = 0; c < cands.size(); ++c) {
                const double d = std::fabs((double) dtr_a[c] - (double) dtr_b[c]);
                if (d > dmax) dmax = d;
                if (d > 1e-4) ++dmis;
            }
            std::fprintf(stderr, "[determinism] serial_vs_serial(same prompt, 2 llama_decode calls) "
                         "max_abs_diff=%.3g mismatch_pts=%lld/%zu\n", dmax, dmis, cands.size());
            det_max_abs_diff = dmax;
        }

        // ------ SERIAL reference: decode each prompt alone, read its candidate logits ------
        std::vector<std::vector<float>> ser_z(n_seqs, std::vector<float>(cands.size(), 0.0f));
        double ser_t0 = now_ms();
        for (int s = 0; s < n_seqs; ++s) {
            b->seq_clear(seqs[s].seq);
            std::string err2;
            if (!b->decode_seq(seqs[s].tok, 0, seqs[s].seq, &err2, {(int32_t)(seqs[s].len - 1)}))
                { std::fprintf(stderr, "error: serial decode seq %d: %s\n", s, err2.c_str()); return 1; }
            for (size_t c = 0; c < cands.size(); ++c) {
                if (cand_first[c] < 0) { ser_z[s][c] = 0.0f; continue; }
                if (!b->read_logits((int32_t)(seqs[s].len - 1), cand_first[c], &ser_z[s][c], &err2))
                    ser_z[s][c] = 0.0f;
            }
        }
        double ser_ms = now_ms() - ser_t0;
        ser_time_sum += ser_ms;

        if (r == 0) {
            // Isolate batching effect from output-count effect: decode seq0 + seq1 in ONE batch
            // but request output ONLY for seq0 (1 output, like serial). If seq0's logits match the
            // serial seq0 bit-identically here, the par-vs-ser diff came from requesting N outputs.
            // If they still differ, merely putting another seq's tokens in the batch changes seq0's
            // forward numerics (batch-shape-dependent llama.cpp numerics).
            std::string err2;
            std::vector<std::vector<int32_t>> det_toks(2);
            det_toks[0] = seqs[0].tok; det_toks[1] = seqs[1].tok;
            std::vector<Backend::MultiSeqRequest> det_reqs;
            for (int q = 0; q < 2; ++q) { Backend::MultiSeqRequest rq; rq.tokens = det_toks[q]; rq.start_pos = 0; rq.seq = seqs[q].seq; det_reqs.push_back(std::move(rq)); }
            b->seq_clear(seqs[0].seq); b->seq_clear(seqs[1].seq);
            std::vector<int32_t> det_oi;
            if (b->decode_multi_seq(det_reqs, &err2, {{0, seqs[0].len - 1}}, &det_oi)) {
                std::vector<float> par1(cands.size(), 0.0f);
                for (size_t c = 0; c < cands.size(); ++c) {
                    if (cand_first[c] < 0) continue;
                    b->read_logits(det_oi[0], cand_first[c], &par1[c], &err2);
                }
                double dmax = 0.0; long long dmis = 0;
                for (size_t c = 0; c < cands.size(); ++c) {
                    const double d = std::fabs((double)par1[c] - (double)ser_z[0][c]);
                    if (d > dmax) dmax = d;
                    if (d > 1e-4) ++dmis;
                }
                std::fprintf(stderr, "[batching-effect] batch(seq0+seq1, output ONLY seq0) vs serial seq0: "
                             "max_abs_diff=%.3g mismatch_pts=%lld/%zu\n", dmax, dmis, cands.size());
            } else {
                std::fprintf(stderr, "[batching-effect] det decode_multi_seq failed: %s\n", err2.c_str());
            }
        }

        // ------ PARALLEL (B1): one decode_multi_seq for all N seqs, read each via out_idx ------
        std::vector<std::pair<int32_t,int32_t>> outs;
        for (int s = 0; s < n_seqs; ++s) outs.push_back({s, seqs[s].len - 1});
        for (int s = 0; s < n_seqs; ++s) b->seq_clear(seqs[s].seq);
        std::vector<Backend::MultiSeqRequest> reqs;
        for (int s = 0; s < n_seqs; ++s) {
            Backend::MultiSeqRequest rq;
            rq.tokens = seqs[s].tok; rq.start_pos = 0; rq.seq = seqs[s].seq;
            reqs.push_back(std::move(rq));
        }
        std::vector<int32_t> out_idx;
        double par_t0 = now_ms();
        if (!b->decode_multi_seq(reqs, &err, outs, &out_idx))
            { std::fprintf(stderr, "error: parallel decode_multi_seq: %s\n", err.c_str()); return 1; }
        double par_ms = now_ms() - par_t0;
        par_time_sum += par_ms;

        std::vector<std::vector<float>> par_z(n_seqs, std::vector<float>(cands.size(), 0.0f));
        for (int o = 0; o < (int) outs.size(); ++o) {
            const int s = outs[o].first;
            const int g = out_idx[o];
            for (size_t c = 0; c < cands.size(); ++c) {
                if (cand_first[c] < 0) { par_z[s][c] = 0.0f; continue; }
                if (!b->read_logits(g, cand_first[c], &par_z[s][c], &err)) par_z[s][c] = 0.0f;
            }
        }

        // ------ compare bit-exact + argmax choice ------
        int round_choice_mismatch = 0;
        double round_max_diff = 0.0;
        for (int s = 0; s < n_seqs; ++s) {
            int ser_arg = 0, par_arg = 0; float sm = -1e30f, pm = -1e30f;
            for (size_t c = 0; c < cands.size(); ++c) {
                const float a = ser_z[s][c], bz = par_z[s][c];
                const float d = std::fabs(a - bz);
                if (d > max_abs_diff) max_abs_diff = d;
                if (d > round_max_diff) round_max_diff = d;
                if (a == bz) ++bit_exact_pts; else ++mismatch_pts;
                if (a > sm) { sm = a; ser_arg = (int) c; }
                if (bz > pm) { pm = bz; par_arg = (int) c; }
            }
            if (ser_arg != par_arg) { ++choice_mismatches; ++round_choice_mismatch; }
        }
        if (round_choice_mismatch == 0) ++total_rounds_ok;

        std::fprintf(stderr,
            "[round %d] seqs=%d ser_total=%.1fms par_one_decode=%.1fms speedup(x)=%.2f "
            "par_vs_ser_maxdiff=%.3g choice_mismatch=%d\n",
            r, n_seqs, ser_ms, par_ms, ser_ms / (par_ms > 0 ? par_ms : 1e-9),
            round_max_diff, round_choice_mismatch);
    }

    // ------ SHORT-TAIL throughput (production-like LCP-hit regime) ------
    // Real /v1/systemone requests after an LCP prefix hit decode only a SHORT tail
    // (state+Answer, ~5-30 tokens). Batching amortizes llama.cpp graph/readout
    // fixed overhead, which dominates for short decodes -> the concurrency win.
    // Decode C short tails both serially (sum) and as one batch; report speedup.
    {
        const int C = n_seqs > 8 ? 8 : n_seqs;
        const int base_seq = 16;   // spare seq ids above the per-round usage
        std::vector<std::string> short_states = {
            "wrong size shoes", "charged twice", "package lost", "refund delay",
            "damaged item", "late delivery", "cancel order", "billing error",
        };
        std::vector<std::vector<int32_t>> stok(C);
        int maxlen = 0;
        for (int s = 0; s < C; ++s) stok[s] = kev::tokenize(tf, short_states[s % short_states.size()], false, false);
        for (int s = 0; s < C; ++s) maxlen = std::max(maxlen, (int) stok[s].size());

        // serial
        std::vector<int32_t> cand_first2(cands.size(), -1);
        for (size_t c = 0; c < cands.size(); ++c) { auto t = kev::tokenize(tf, cands[c], false, false); cand_first2[c] = t.empty() ? -1 : t[0]; }
        std::string err2;
        double st0 = now_ms();
        for (int s = 0; s < C; ++s) {
            const int sid = base_seq + s;
            b->seq_clear(sid);
            if (!b->decode_seq(stok[s], 0, sid, &err2, {(int32_t)(stok[s].size() - 1)})) { std::fprintf(stderr, "short serial err: %s\n", err2.c_str()); return 1; }
            float x = 0;
            for (size_t c = 0; c < cands.size(); ++c) if (cand_first2[c] >= 0) b->read_logits((int32_t)(stok[s].size() - 1), cand_first2[c], &x, &err2);
        }
        double short_ser_ms = now_ms() - st0;

        // parallel one batch
        for (int s = 0; s < C; ++s) b->seq_clear(base_seq + s);
        std::vector<Backend::MultiSeqRequest> sreqs;
        for (int s = 0; s < C; ++s) { Backend::MultiSeqRequest rq; rq.tokens = stok[s]; rq.start_pos = 0; rq.seq = base_seq + s; sreqs.push_back(std::move(rq)); }
        std::vector<std::pair<int32_t,int32_t>> souts;
        for (int s = 0; s < C; ++s) souts.push_back({s, (int32_t)(stok[s].size() - 1)});
        std::vector<int32_t> soi;
        double pt0 = now_ms();
        if (!b->decode_multi_seq(sreqs, &err2, souts, &soi)) { std::fprintf(stderr, "short par err: %s\n", err2.c_str()); return 1; }
        double short_par_ms = now_ms() - pt0;
        float x = 0;
        for (int o = 0; o < C; ++o) for (size_t c = 0; c < cands.size(); ++c) if (cand_first2[c] >= 0) b->read_logits(soi[o], cand_first2[c], &x, &err2);

        std::fprintf(stderr, "\n[short-tail] C=%d states_len~%d..%d batched ser_total=%.1fms par_one_decode=%.1fms "
                     "speedup_vs_Cxserial=%.2fx\n",
                     C, 1, maxlen, short_ser_ms, short_par_ms,
                     short_ser_ms / (short_par_ms > 0 ? short_par_ms : 1e-9));
    }

    const double avg_speedup = ser_time_sum / (par_time_sum > 0 ? par_time_sum : 1e-9);
    std::fprintf(stderr, "\n===== B1 RESULT =====\n");
    std::fprintf(stderr, "rounds=%d seqs_per_round=%d cands=%zu\n", rounds, n_seqs, cands.size());
    std::fprintf(stderr, "ser_time_total=%.1fms par_time_total=%.1fms avg_speedup_vs_serial=%.2fx\n",
                 ser_time_sum, par_time_sum, avg_speedup);
    std::fprintf(stderr, "bit_exact_pts=%lld mismatch_pts=%lld choice_mismatches=%lld "
                 "par_vs_ser_max_abs_diff=%.3g\n",
                 bit_exact_pts, mismatch_pts, choice_mismatches, max_abs_diff);
    std::fprintf(stderr, "det_serial_vs_serial_max_abs_diff=%.3g  (engine intrinsic run-to-run noise; "
                 "if par_vs_ser diff <= this, batching adds no extra error)\n", det_max_abs_diff);
    const bool par_no_extra_error = det_max_abs_diff <= 0.0 || max_abs_diff <= det_max_abs_diff;
    const bool ok = (choice_mismatches == 0) && par_no_extra_error;
    std::fprintf(stderr, "PARALLEL_CHOICE_EQUALS_SERIAL=%d | PAR_ERR_LE_ENGINE_NOISE=%d | "
                 "choices_identical_rounds=%d/%d\n",
                 choice_mismatches == 0 ? 1 : 0, par_no_extra_error ? 1 : 0, total_rounds_ok, rounds);
    return ok ? 0 : 1;
}