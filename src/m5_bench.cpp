// m5_bench.cpp — M5: 真实 LoRA 对拍。跑一次 base 或 base+LoRA, 输出每题概率并把每题
// decide 隐藏向量[1024]落盘(二进制), 供 Python 层对比 base vs base+lora.
//   m5_bench --model gguf --head head.bin --req stateqs.json [--lora kev-lora.gguf] [--hidden-out dump.bin]
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

int main(int argc, char ** argv) {
    std::string model_path, head, reqfile, lora_path, hidden_out;
    float lora_scale = 1.0f;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--model") model_path = next();
        else if (a == "--head") head = next();
        else if (a == "--req") reqfile = next();
        else if (a == "--lora") lora_path = next();
        else if (a == "--lora-scale") lora_scale = (float) std::atof(next().c_str());
        else if (a == "--hidden-out") hidden_out = next();
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || reqfile.empty()) {
        std::fprintf(stderr, "usage: m5_bench --model gguf [--head head.bin] --req req.json [--lora lora.gguf] [--lora-scale f] [--hidden-out dump.bin]\n");
        return 2;
    }
    kev::KevModelOptions opt;
    opt.model_path = model_path; opt.head_path = head; opt.lru_capacity = 2; opt.n_ctx = 8192;
    opt.lora_path = lora_path; opt.lora_scale = lora_scale;
    kev::KevModel model;
    std::string err;
    if (!model.init(opt, &err)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 1; }

    auto req = read_req(reqfile);
    auto res = model.probs(req, &err);
    if (res.probs.empty()) { std::fprintf(stderr, "probs failed: %s\n", err.c_str()); model.close(); return 1; }

    for (size_t q = 0; q < res.probs.size(); ++q) {
        std::printf("probs(q%zu):", q);
        for (float p : res.probs[q]) std::printf(" %.6f", (double)p);
        std::printf("\n");
    }
    if (!res.hidden_decide.empty()) {
        // max |hidden| + norm for sanity
        for (size_t q = 0; q < res.hidden_decide.size(); ++q) {
            double m = 0, s = 0;
            for (float x : res.hidden_decide[q]) { m = std::max(m, (double) std::fabs(x)); s += (double) x * x; }
            std::printf("hidden(q%zu): len=%zu max|.|=%.5f l2=%.3f\n",
                        q, res.hidden_decide[q].size(), m, std::sqrt(s));
        }
        if (!hidden_out.empty()) {
            std::ofstream fo(hidden_out, std::ios::binary);
            int32_t nq = (int32_t) res.hidden_decide.size();
            int32_t dim = (int32_t) res.hidden_decide[0].size();
            fo.write((const char *) &nq, sizeof(nq));
            fo.write((const char *) &dim, sizeof(dim));
            for (const auto & h : res.hidden_decide)
                fo.write((const char *) h.data(), h.size() * sizeof(float));
            std::printf("wrote hidden dump: %s (%d x %d)\n", hidden_out.c_str(), nq, dim);
        }
    }
    model.close();
    return 0;
}