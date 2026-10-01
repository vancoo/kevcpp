// lora_ab.cpp — merged-vs-separate LoRA 对拍与性能 A/B 工具。
//   lora_ab --model base.gguf --head head.bin --lora lora.gguf \
//           [--merge] [--req file.json ...] [--ctx 8192] [--threads 0] [--lru 2]
//           [--timing N] [--scale 1.0]
// 输出:
//   - 每问一行 "PR <reqIdx> <questionIdx> <argmax> <p1> <p2> ..." (概率逐选项)
//   - 若 --timing N, 对第一/全部请求做 N 次 cold 计时, 输出 "TIME cold state=.. rows=.. total=.."
// 两次运行(分离 / merged)分别收集输出, 由 python 脚本做 argmax 一致率 / max|Δp| / avg|Δp| 比较。
#include "kev_api.h"
#include "kev_model.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using nlohmann::json;

static json read_json(const std::string & path) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "cannot open request file %s\n", path.c_str()); std::exit(2); }
    return json::parse(f, nullptr, true);
}

// 把请求 JSON 规范化为 KevRequest。支持:
//   (a) systemone 格式 {state:..., model, questions:{id:{type,...}}}
//   (b) 朴素 {state, questions:[{instr, options:[...]}]}
static bool load_request(const json & body, kev::KevRequest & rec, std::string * err) {
    if (!body.contains("state") || !body.contains("questions")) {
        if (err) *err = "request missing state/questions"; return false;
    }
    const json & qs = body["questions"];
    // 识别格式: systemone 用的是 object(value 有 type 字段); 朴素用 array(value 有 options)
    if (qs.is_array()) {
        rec.state = body["state"].is_string() ? body["state"].get<std::string>() : body["state"].dump();
        for (const auto & q : qs) {
            if (!q.contains("instr") || !q.contains("options")) { if (err) *err="bad plain question"; return false; }
            kev::KevQuestion kq;
            kq.instr = q["instr"].get<std::string>();
            for (const auto & o : q["options"]) kq.options.push_back(o.get<std::string>());
            rec.questions.push_back(std::move(kq));
        }
        return true;
    }
    // systemone
    kev::ParsedSystemOne ps;
    if (!kev::parse_systemone(body, ps, err)) return false;
    rec = ps.rec;
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path, head, lora_path;
    std::vector<std::string> reqfiles;
    bool merge = false;
    int ctx = 8192, threads = 0, lru = 2, timing = 0;
    float scale = 1.0f;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--model") model_path = next();
        else if (a == "--head") head = next();
        else if (a == "--lora") lora_path = next();
        else if (a == "--req") reqfiles.push_back(next());
        else if (a == "--merge") merge = true;
        else if (a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--lru") lru = std::atoi(next().c_str());
        else if (a == "--timing") timing = std::atoi(next().c_str());
        else if (a == "--scale") scale = (float)std::atof(next().c_str());
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || head.empty() || reqfiles.empty()) {
        std::fprintf(stderr, "usage: lora_ab --model base.gguf --head head.bin [--lora lora.gguf] [--merge] --req f.json ...\n");
        return 2;
    }

    kev::KevModelOptions opt;
    opt.model_path = model_path; opt.head_path = head; opt.lru_capacity = lru; opt.n_ctx = ctx;
    opt.n_threads = threads;
    opt.merge_lora = merge;
    if (!lora_path.empty()) { opt.lora_path = lora_path; opt.lora_scale = scale; }
    kev::KevModel model;
    std::string err;
    if (!model.init(opt, &err)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    std::fprintf(stderr, "[mode=%s]\n", merge ? "merged" : "separate");

    // 加载请求
    std::vector<kev::KevRequest> reqs;
    for (auto & p : reqfiles) {
        json body = read_json(p);
        kev::KevRequest rec;
        if (!load_request(body, rec, &err)) { std::fprintf(stderr, "load %s: %s\n", p.c_str(), err.c_str()); return 1; }
        reqs.push_back(std::move(rec));
    }

    int reqidx = 0;
    for (auto & rec : reqs) {
        std::string e2;
        auto res = model.probs(rec, &e2);
        if (res.probs.empty()) { std::fprintf(stderr, "probs %s\n", e2.c_str()); return 1; }
        // 首问必为 cold(state miss); 输出其分段时间供 fresh-process best-of-N 计时。
        if (reqidx == 0)
            std::printf("COLD state=%.2f rows=%.2f total=%.2f state_hit=%d\n",
                        res.t_state_ms, res.t_rows_ms, res.t_total_ms, res.state_hit ? 1 : 0);
        for (size_t qi = 0; qi < res.probs.size(); ++qi) {
            const auto & p = res.probs[qi];
            int argmax = 0;
            for (size_t k = 1; k < p.size(); ++k) if (p[k] > p[argmax]) argmax = (int)k;
            std::printf("PR %d %zu %d", reqidx, qi, argmax);
            for (float x : p) std::printf(" %.9g", (double)x);
            std::printf("\n");
        }
        ++reqidx;
    }

    // cold timing(best-of-n 直接打印每次)
    if (timing > 0 && !reqs.empty()) {
        for (int t = 0; t < timing; ++t) {
            std::string e3;
            auto res = model.probs(reqs[0], &e3);
            std::printf("TIME cold state=%.2f rows=%.2f total=%.2f state_hit=%d\n",
                        res.t_state_ms, res.t_rows_ms, res.t_total_ms, res.state_hit ? 1 : 0);
        }
    }

    model.close();
    return 0;
}