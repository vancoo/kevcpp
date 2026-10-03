// m8_bench.cpp — P.4 对照: 等长多问共享前缀"批量 decode" vs "逐问 decode"逐位一致性 + 耗时。
//   m8_bench --model gguf --head head.bin [--lora lora.gguf] --req multi.json [--ctx N] [--threads N] [--lru 2] [--warm]
// 对同一多问请求分别以 batch_enabled=true / false 跑 probs(), 比较:
//   - 每题 probs(float) 逐位相等
//   - 每题 decide hidden(float[1024]) 逐位相等
//   - 平均/每题 耗时(建议 best-of-3 见脚本)
#include "kev_api.h"
#include "kev_model.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using nlohmann::json;

static json read_json(const std::string & path) {
    std::ifstream f(path);
    return json::parse(f);
}

static bool check_bit(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;   // 逐位 float 相等
    }
    return true;
}

static float max_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float m = 0.0f;
    size_t n = a.size() < b.size() ? a.size() : b.size();
    for (size_t i = 0; i < n; ++i) {
        float d = (float)std::fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char ** argv) {
    std::string model_path, head, lora_path, reqfile;
    int ctx = 8192, threads = 0, lru = 2, reps = 3;
    bool warm = false;
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
        else if (a == "--reps") reps = std::atoi(next().c_str());
        else if (a == "--warm") warm = true;
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || reqfile.empty()) {
        std::fprintf(stderr, "usage: m8_bench --model gguf --head head.bin [--lora lora.gguf] --req multi.json\n");
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

    // 诊断 branch 长度(确认是否有等长组)
    {
        auto enc = model.encode(ps.rec, &err);
        auto rows = model.rows_of(enc, &err);
        std::printf("questions=%zu  branch lens:", rows.rows.size());
        for (const auto & r : rows.rows) std::printf(" %zu", r.ids.size());
        std::printf("  (state=%zu)\n", rows.state_ids.size());
    }

    auto run = [&](bool batched) {
        model.set_batch_enabled(batched);
        // 预热到 state 命中
        std::string e;
        if (warm) { auto w = model.probs(ps.rec, &e); (void)w; }
        double t_sum = 0, rows_sum = 0;
        std::vector<std::vector<float>> probs, hidden;
        for (int r = 0; r < reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            auto res = model.probs(ps.rec, &e);
            double dt = (double)std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0).count() / 1e3;
            if (res.probs.empty()) { std::fprintf(stderr, "probs failed: %s\n", e.c_str()); }
            if (r == reps - 1) { probs = res.probs; hidden = res.hidden_decide; }
            t_sum += dt; rows_sum += res.t_rows_ms;
        }
        std::printf("  batch=%s  avg_total=%.1fms  rows=%.1fms (reps=%d)\n",
                    batched ? "ON" : "OFF ", t_sum / (double)reps, rows_sum / (double)reps, reps);
        return std::make_pair(probs, hidden);
    };

    std::printf("== batched path ==\n");
    auto bn = run(true);
    std::printf("== per-question path ==\n");
    auto bo = run(false);

    const size_t Q = bn.first.size();
    std::printf("\n== bit-exact compare (batch ON vs OFF) ==\n");
    bool all_ok = true;
    for (size_t q = 0; q < Q; ++q) {
        bool p_ok = q < bo.first.size() && check_bit(bn.first[q], bo.first[q]);
        bool h_ok = q < bo.second.size() && check_bit(bn.second[q], bo.second[q]);
        float dp = (q < bo.first.size()) ? max_diff(bn.first[q], bo.first[q]) : -1.0f;
        all_ok &= p_ok && h_ok;
        std::printf("  q%zu  probs_bit=%s hidden_bit=%s  max|dp|=%.3e\n",
                    q, p_ok ? "EQ " : "NE ", h_ok ? "EQ " : "NE ", (double)dp);
        if (!p_ok) {
            std::printf("    on :"); for (float p : bn.first[q]) std::printf(" %.6f", (double)p);
            std::printf("\n    off:"); for (float p : bo.first[q]) std::printf(" %.6f", (double)p);
            std::printf("\n");
        }
    }
    std::printf("\nRESULT: %s\n", all_ok ? "BIT-EXACT (batch == per-question, all questions)" : "MISMATCH");
    model.close();
    return all_ok ? 0 : 1;
}