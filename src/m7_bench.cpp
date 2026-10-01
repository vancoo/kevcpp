// m7_bench.cpp — M7: CPU 慢定位。用 TypeSafe systemone 请求做分段计时复现。
//   m7_bench --model gguf --head head.bin [--lora lora.gguf] --req req_systemone.json
//           [--ctx N] [--threads N] [--lru 2] [--warm]
// 打印: cold/warm 的 state_hit / state_ms / rows_ms / 每问 rows_ms / total_ms / threads。
#include "kev_api.h"
#include "kev_model.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <algorithm>

using nlohmann::json;

static json read_json(const std::string & path) {
    std::ifstream f(path);
    return json::parse(f);
}

int main(int argc, char ** argv) {
    std::string model_path, head, lora_path, reqfile;
    int ctx = 8192, threads = 0, lru = 2;
    bool warm = false;
    int iters = 1;   // kevcpp graph-cache A/B: repeat the (warm) request N times
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--model") model_path = next();
        else if (a == "--head") head = next();
        else if (a == "--lora") lora_path = next();
        else if (a == "--req") reqfile = next();
        else if (a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--lru") lru = std::atoi(next().c_str());
        else if (a == "--warm") warm = true;
        else if (a == "--iters") iters = std::max(1, std::atoi(next().c_str()));
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (warm && iters < 2) iters = 2;   // --warm implies at least a cold + one warm run
    if (model_path.empty() || reqfile.empty()) {
        std::fprintf(stderr, "usage: m7_bench --model gguf --head head.bin [--lora lora.gguf] --req systemone.json [--ctx 8192] [--threads 0] [--lru 2] [--warm] [--iters N]\n");
        return 2;
    }

    kev::KevModelOptions opt;
    opt.model_path = model_path; opt.head_path = head; opt.lru_capacity = lru; opt.n_ctx = ctx;
    opt.n_threads = threads;
    if (!lora_path.empty()) { opt.lora_path = lora_path; opt.lora_scale = 1.0f; }
    kev::KevModel model;
    std::string err;
    if (!model.init(opt, &err)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 1; }

    json body = read_json(reqfile);
    kev::ParsedSystemOne ps;
    if (!kev::parse_systemone(body, ps, &err)) { std::fprintf(stderr, "parse: %s\n", err.c_str()); return 1; }

    // 诊断: branch token 长度(决定 decode 规模)
    {
        auto enc = model.encode(ps.rec, &err);
        if (!enc.decide_idx.empty()) {
            auto rows = model.rows_of(enc, &err);
            std::printf("branch token lengths per question:");
            for (const auto & r : rows.rows) std::printf(" %zu", r.ids.size());
            std::printf("   (state tokens=%zu)\n", rows.state_ids.size());
        }
    }

    std::printf("config: n_ctx=%d threads(set)=%d lru=%d\n", ctx, threads, lru);
    std::printf("llama.cpp actual: n_threads=%d n_threads_batch=%d\n",
                model.threads_actual(), model.threads_batch_actual());
    std::printf("state_len=%zu tokens, questions=%zu\n", ps.rec.state.size(), ps.rec.questions.size());

    auto run_one = [&](bool cold) {
        std::string e2;
        auto res = model.probs(ps.rec, &e2);
        if (res.probs.empty()) { std::fprintf(stderr, "probs: %s\n", e2.c_str()); return; }
        std::printf("%s: state_hit=%s  state=%.1fms  rows=%.1fms  total=%.1fms  (threads=%d)\n",
                    cold ? "cold" : "warm", res.state_hit ? "Y" : "N",
                    res.t_state_ms, res.t_rows_ms, res.t_total_ms, model.threads());
        std::printf("  rows_ms_per_q:");
        for (double x : res.rows_ms_per_q) std::printf(" %.1f", x);
        std::printf("\n");
        std::printf("  probs:"); for (float p : res.probs[0]) std::printf(" %.4f", (double)p); std::printf("\n");
    };

    run_one(true);                    // cold (state miss)
    if (warm) {
        for (int it = 1; it < std::max(iters, 2); ++it) {
            std::string e2;
            auto res = model.probs(ps.rec, &e2);
            std::printf("warm%d: state_hit=%s  state=%.1fms  rows=%.1fms  total=%.1fms\n",
                        it, res.state_hit ? "Y" : "N",
                        res.t_state_ms, res.t_rows_ms, res.t_total_ms);
        }
    }
    model.close();
    return 0;
}