// m3_bench.cpp — M3 状态前缀缓存基准: 同 state 二次请求命中 + 概率不变 + 耗对比。
//   m3_bench --model <gguf> --head head.bin --req stateqs.json [--n 3] [--lru 2]
// req JSON: {"state": str, "questions":[{"instr":str,"options":[str,...]}, ...]}
#include "kev_model.h"
#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using nlohmann::json;

static kev::KevRequest read_req(const std::string & path) {
    std::ifstream f(path);
    json j = json::parse(f);
    kev::KevRequest r;
    r.state = j["state"].get<std::string>();
    for (const auto & q : j["questions"]) {
        kev::KevQuestion kq;
        kq.instr = q["instr"].get<std::string>();
        for (const auto & o : q["options"]) kq.options.push_back(o.get<std::string>());
        r.questions.push_back(std::move(kq));
    }
    if (j.contains("labels")) r.labels = j["labels"].get<std::vector<int32_t>>();
    return r;
}

static double max_dp(const std::vector<std::vector<float>> & a,
                     const std::vector<std::vector<float>> & b) {
    double m = 0;
    for (size_t q = 0; q < a.size(); ++q)
        for (size_t k = 0; k < a[q].size(); ++k)
            m = std::max(m, (double) std::fabs((double)a[q][k] - (double)b[q][k]));
    return m;
}

int main(int argc, char ** argv) {
    std::string model_path, head, reqfile;
    int lru = 2;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--model") model_path = next();
        else if (a == "--head") head = next();
        else if (a == "--req") reqfile = next();
        else if (a == "--lru") lru = std::atoi(next().c_str());
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || reqfile.empty()) {
        std::fprintf(stderr, "usage: m3_bench --model model.gguf [--head head.bin] --req stateqs.json [--lru n]\n");
        return 2;
    }

    kev::KevModelOptions opt;
    opt.model_path = model_path; opt.head_path = head; opt.lru_capacity = lru; opt.n_ctx = 8192;
    kev::KevModel model;
    std::string err;
    if (!model.init(opt, &err)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 1; }

    auto req = read_req(reqfile);
    // build a second request with a DIFFERENT state (to exercise eviction)
    auto reqB = req; reqB.state = std::string("A totally different state for eviction. ") + req.state;

    auto run = [&](const kev::KevRequest & r) {
        auto res = model.probs(r, &err);
        if (!err.empty()) { std::fprintf(stderr, "probs: %s\n", err.c_str()); }
        return res;
    };

    std::printf("seq layout: n_seq_max=%d scratch=%d (state slots %d..%d)\n",
                model.n_seq_max(), model.scratch_seq(), 1, 1 + lru - 1);

    // pass1 (cold, miss)
    auto p1 = run(req);
    std::printf("pass1(cold) : state_hit=%s  state=%.1fms  rows=%.1fms  total=%.1fms\n",
                p1.state_hit ? "YES" : "no", p1.t_state_ms, p1.t_rows_ms, p1.t_total_ms);
    // pass2 (same state, expect hit)
    auto p2 = run(req);
    std::printf("pass2(same) : state_hit=%s  state=%.1fms  rows=%.1fms  total=%.1fms\n",
                p2.state_hit ? "YES" : "no", p2.t_state_ms, p2.t_rows_ms, p2.t_total_ms);
    // pass3 (different state, miss)
    auto p3 = run(reqB);
    std::printf("pass3(other): state_hit=%s  state=%.1fms  rows=%.1fms  total=%.1fms\n",
                p3.state_hit ? "YES" : "no", p3.t_state_ms, p3.t_rows_ms, p3.t_total_ms);
    // pass4 (back to original state; hit again unless evicted)
    auto p4 = run(req);
    std::printf("pass4(orig) : state_hit=%s  state=%.1fms  rows=%.1fms  total=%.1fms\n",
                p4.state_hit ? "YES" : "no", p4.t_state_ms, p4.t_rows_ms, p4.t_total_ms);

    // probability parity
    const double dp12 = max_dp(p1.probs, p2.probs);
    const double dp14 = max_dp(p1.probs, p4.probs);
    std::printf("probs: pass1 vs pass2 max|dp|=%.3e   pass1 vs pass4 max|dp|=%.3e\n", dp12, dp14);
    std::printf("probs(q0): "); for (float p : p1.probs.empty()? std::vector<float>(): p1.probs[0]) std::printf("%.4f ", (double)p); std::printf("\n");

    // speedup of hit vs cold
    if (p1.t_total_ms > 0 && p2.t_total_ms > 0)
        std::printf("speedup(hit vs cold total) = %.2fx  (rows-only pass2 == %.1fms)\n",
                    p1.t_total_ms / p2.t_total_ms, p2.t_rows_ms);

    const bool hit2 = p2.state_hit, hit4 = p4.state_hit;
    const bool probs_same = (dp12 < 1e-4) && (dp14 < 1e-4);
    std::printf("RESULT: %s (pass2 hit=%s pass4 hit=%s probs_same=%s lru_cap=%d)\n",
                (hit2 && hit4 && probs_same) ? "PASS" : "CHECK",
                hit2 ? "Y" : "N", hit4 ? "Y" : "N", probs_same ? "Y" : "N", lru);
    model.close();
    return (hit2 && hit4 && probs_same) ? 0 : 1;
}