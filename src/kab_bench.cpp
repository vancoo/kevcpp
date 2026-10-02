// kab_bench.cpp — KEV perf-gemv A/B 对拍工具 (argmax 一致率)。
// 对一组 systemone 请求文件, 用 KevModel.probs 算每问概率, 输出稳定 dump:
//   ARGMAX <fileIdx> <qIdx> <argmax> <maxP>  p0 p1 ... pK-1
// 在 env KEV_Q8_AVX512 设为 开/关 下各跑一遍, 对拍 argmax 一致率。
#include "kev_api.h"
#include "kev_model.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

using nlohmann::json;

static json read_json(const std::string & path) {
    std::ifstream f(path);
    return json::parse(f);
}

int main(int argc, char ** argv) {
    std::string model_path, head, lora_path;
    std::vector<std::string> reqs;
    int ctx = 8192, threads = 0, lru = 2;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--model") model_path = next();
        else if (a == "--head") head = next();
        else if (a == "--lora") lora_path = next();
        else if (a == "--req") reqs.push_back(next());
        else if (a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--lru") lru = std::atoi(next().c_str());
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || head.empty() || reqs.empty()) {
        std::fprintf(stderr, "usage: kab_bench --model gguf --head head.bin [--lora lora] [--req f.json]* [--ctx N] [--threads N] [--lru 2]\n");
        return 2;
    }

    kev::KevModelOptions opt;
    opt.model_path = model_path; opt.head_path = head; opt.lru_capacity = lru; opt.n_ctx = ctx;
    opt.n_threads = threads;
    if (!lora_path.empty()) { opt.lora_path = lora_path; opt.lora_scale = 1.0f; }
    kev::KevModel model;
    std::string err;
    if (!model.init(opt, &err)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 1; }

    const char * kev_env = getenv("KEV_Q8_AVX512");
    std::printf("# KEV_Q8_AVX512=%s (%s path)\n",
                kev_env ? kev_env : "<unset>", (kev_env && *kev_env == '0') ? "OLD/scalar" : "NEW/avxvnni");

    for (size_t f = 0; f < reqs.size(); ++f) {
        json body = read_json(reqs[f]);
        kev::ParsedSystemOne ps;
        if (!kev::parse_systemone(body, ps, &err)) { std::fprintf(stderr, "parse %s: %s\n", reqs[f].c_str(), err.c_str()); return 1; }
        kev::KevModel::ProbsResult res = model.probs(ps.rec, &err);
        if (res.probs.empty()) { std::fprintf(stderr, "probs %s: %s\n", reqs[f].c_str(), err.c_str()); return 1; }
        for (size_t q = 0; q < res.probs.size(); ++q) {
            const auto & v = res.probs[q];
            int am = (int)(std::max_element(v.begin(), v.end()) - v.begin());
            std::printf("ARGMAX %zu %zu %d %f", f, q, am, (double)v[am]);
            for (float p : v) std::printf(" %f", (double)p);
            std::printf("\n");
        }
    }
    model.close();
    return 0;
}