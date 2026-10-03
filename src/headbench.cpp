// headbench.cpp — M1 对拍工具
//
// 两个模式:
//   1) headbench (默认): 纯 PointerHead 数学对拍。读 cases.json(隐藏向量+参考概率) + head.bin,
//      用 C++ kev_head 算概率并与参考概率比对，报告 max|dp| 与 argmax 一致率。
//      用例:   headbench --head head.bin --cases cases.json
//   2) readout: 端到端。加载 GGUF, 对 req.json 的 token 序列 prefill, 在 decide/opt 绝对位置取
//      hidden state(1024), 过 PointerHead 得概率, 写 probs.json; 也可 --hidden-out 导出隐藏向量
//      供 Python 用同一 head 独立重算参考概率做端到端对拍。
//      用例:   headbench --model model.gguf --req req.json --head head.bin
//                [--probs-out probs.json] [--hidden-out hidden.json]
//
// JSON 用 nlohmann/json(third_party/llama.cpp/vendor/nlohmann).
#include "llama.h"
#include "kev_head.h"
#include "kev_encode.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using nlohmann::json;

namespace {

std::vector<float> read_floats(const json & j) {
    std::vector<float> v;
    v.reserve(j.size());
    for (const auto & x : j) v.push_back(x.get<float>());
    return v;
}

void print_probs(const std::vector<std::vector<float>> & ps) {
    std::printf("probs per question:\n");
    for (size_t q = 0; q < ps.size(); ++q) {
        std::printf("  q%zu  -> ", q);
        for (float p : ps[q]) std::printf("%.6f ", (double)p);
        std::printf("\n");
    }
}

// ---------------- headbench mode ----------------
int run_headbench(const std::string & headfile, const std::string & casesfile) {
    kev::PointerHead h;
    if (!h.load_from_file(headfile)) return 1;

    std::ifstream f(casesfile);
    if (!f) { std::fprintf(stderr, "headbench: cannot open cases '%s'\n", casesfile.c_str()); return 1; }
    json root = json::parse(f);

    const int dim = root.value("dim", 1024);
    const std::vector<json> & cases = root["cases"];

    double max_dp = 0.0; int argmax_match = 0, n = 0;
    for (const auto & c : cases) {
        auto h_decide = read_floats(c["h_decide"]);
        std::vector<const float *> opts;
        std::vector<std::vector<float>> opts_storage;
        for (const auto & o : c["h_opts"]) { opts_storage.push_back(read_floats(o)); }
        for (auto & s : opts_storage) opts.push_back(s.data());

        std::vector<float> ref = read_floats(c["ref_probs"]);
        std::vector<float> got = h.probs(h_decide.data(), opts);

        int ref_idx = (int)(std::max_element(ref.begin(), ref.end()) - ref.begin());
        int got_idx = (int)(std::max_element(got.begin(), got.end()) - got.begin());
        argmax_match += (ref_idx == got_idx);
        for (size_t i = 0; i < ref.size(); ++i)
            max_dp = std::max(max_dp, (double) std::fabs((double)got[i] - (double)ref[i]));
        n++;
    }

    std::printf("headbench: %d cases\n", n);
    std::printf("  max |dp|      = %.10f\n", max_dp);
    std::printf("  argmax match  = %d / %d\n", argmax_match, n);
    std::printf("  RESULT        : %s\n",
                (max_dp < 1e-5 && argmax_match == n) ? "PASS" : "CHECK");
    return 0;
}

// ---------------- readout mode ----------------
int run_readout(const std::string & model_path, const std::string & reqfile,
                const std::string & headfile, const std::string & probs_out,
                const std::string & hidden_out) {
    std::ifstream rf(reqfile);
    if (!rf) { std::fprintf(stderr, "readout: cannot open req '%s'\n", reqfile.c_str()); return 1; }
    json req = json::parse(rf);
    const auto ids64 = req["ids"].get<std::vector<int64_t>>();
    std::vector<int32_t> ids(ids64.begin(), ids64.end());
    std::vector<int32_t> decide = req["decide_idx"].get<std::vector<int32_t>>();
    const auto & optj = req["opt_idx"];
    std::vector<std::vector<int32_t>> opts;
    for (const auto & o : optj) opts.push_back(o.get<std::vector<int32_t>>());

    kev::PointerHead head;
    if (!head.load_from_file(headfile)) return 1;
    if ((int) head.dim_in() != 1024) { std::fprintf(stderr, "readout: head dim_in=%d (expect 1024)\n", head.dim_in()); return 1; }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { std::fprintf(stderr, "readout: failed to load model\n"); llama_backend_free(); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = (uint32_t) std::max(ids.size() * 2 + 256, (size_t) 2048);
    cparams.n_batch     = 512;
    cparams.n_ubatch    = 512;
    cparams.n_outputs_max_per_seq = (uint32_t) ids.size();
    cparams.embeddings  = true;
    cparams.lm_head     = false;   // Rank-1: readout only reads embeddings (llama_get_embeddings_ith), never logits
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cparams.n_threads   = 0; cparams.n_threads_batch = 0;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { std::fprintf(stderr, "readout: failed to init context\n"); llama_model_free(model); llama_backend_free(); return 1; }

    llama_batch batch = llama_batch_get_one(ids.data(), (int32_t) ids.size());
    if (llama_decode(ctx, batch)) { std::fprintf(stderr, "readout: decode failed\n"); llama_free(ctx); llama_model_free(model); llama_backend_free(); return 1; }

    const int dim = 1024;
    // 每个 decide/opt 位置的 hidden state
    auto hidden_at = [&](int32_t idx) {
        float * p = llama_get_embeddings_ith(ctx, idx);
        if (!p) { std::fprintf(stderr, "readout: no embedding at %d\n", idx); return std::vector<float>(); }
        return std::vector<float>(p, p + dim);
    };

    std::vector<std::vector<float>> dec_h;                // [Q][1024]
    std::vector<std::vector<std::vector<float>>> opt_h;   // [Q][K][1024]
    std::vector<std::vector<float>> probs;                // [Q][K]
    bool ok = true;
    for (size_t q = 0; q < decide.size(); ++q) {
        auto hd = hidden_at(decide[q]);
        if (hd.empty()) { ok = false; break; }
        dec_h.push_back(hd);
        std::vector<std::vector<float>> oh;
        for (int32_t oi : opts[q]) { auto v = hidden_at(oi); if (v.empty()) { ok = false; break; } oh.push_back(v); }
        if (!ok) break;
        // NOTE: build pointers into the stable storage AFTER filling `oh`; the locals are destroyed
        // each iteration, so taking .data() of `v` (a local copy) mid-loop would dangle.
        std::vector<const float *> optp;
        for (auto & s : oh) optp.push_back(s.data());
        opt_h.push_back(oh);
        probs.push_back(head.probs(dec_h.back().data(), optp));
    }

    if (ok) {
        print_probs(probs);
        json out;
        std::vector<int> am;
        for (auto & p : probs) am.push_back((int)(std::max_element(p.begin(), p.end()) - p.begin()));
        out["argmax"] = am;
        out["probs"] = probs;
        if (!probs_out.empty()) { std::ofstream of(probs_out); of << out.dump(2); }
        std::printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
    }

    if (!hidden_out.empty() && ok) {
        json hid;
        hid["dim"] = dim;
        hid["decide"] = dec_h;
        hid["opts"] = opt_h;
        std::ofstream of(hidden_out); of << hid.dump();
        std::printf("wrote hidden states -> %s\n", hidden_out.c_str());
    }

    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return ok ? 0 : 1;
}

// ---------------- tokens mode (tokenizer parity probe: llama_tokenize vs HF fast) ----------------
int run_tokens(const std::string & model_path, const std::string & textfile,
               const std::string & add_special) {
    std::ifstream tf(textfile);
    if (!tf) { std::fprintf(stderr, "tokens: cannot open '%s'\n", textfile.c_str()); return 1; }
    std::string text((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>());
    const bool special = (add_special == "true");

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { std::fprintf(stderr, "tokens: failed to load model\n"); llama_backend_free(); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // llama_tokenize needs an upper-bound buffer; call once to size, then fill
    std::vector<llama_token> buf(text.size() * 4 + 16);
    const int32_t ntok = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
                                        buf.data(), (int32_t) buf.size(),
                                        /*add_special=*/special, /*parse_special=*/true);
    if (ntok < 0) { std::fprintf(stderr, "tokens: overflow (need %d)\n", -ntok); return 1; }

    // dump ids as JSON array on one line
    std::printf("[");
    for (int32_t i = 0; i < ntok; ++i) { if (i) std::printf(","); std::printf("%d", (int) buf[i]); }
    std::printf("]\n");
    std::fflush(stdout);

    llama_model_free(model); llama_backend_free();
    return 0;
}

// ---------------- encode mode (full kev.model.encode parity dump) ----------------
int run_encode(const std::string & model_path, const std::string & reqfile) {
    std::ifstream rf(reqfile);
    if (!rf) { std::fprintf(stderr, "encode: cannot open '%s'\n", reqfile.c_str()); return 1; }
    json req = json::parse(rf);

    kev::KevRequest kreq;
    kreq.state = req["state"].get<std::string>();
    for (const auto & qj : req["questions"]) {
        kev::KevQuestion q;
        q.instr = qj["instr"].get<std::string>();
        for (const auto & o : qj["options"]) q.options.push_back(o.get<std::string>());
        kreq.questions.push_back(std::move(q));
    }
    if (req.contains("labels")) kreq.labels = req["labels"].get<std::vector<int32_t>>();

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { std::fprintf(stderr, "encode: failed to load model\n"); llama_backend_free(); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::string err;
    auto enc = kev::encode(vocab, kreq, /*max_state=*/384, /*max_branch=*/1024, &err);
    auto rows = kev::rows_of(enc, &err);

    json out;
    out["ids"] = enc.ids;
    out["seg"] = enc.seg;
    out["pos"] = enc.pos;
    out["opt"] = enc.opt;
    out["decide_idx"] = enc.decide_idx;
    out["opt_idx"] = enc.opt_idx;
    out["ls"] = (int32_t) rows.state_ids.size();
    out["rows"] = json::array();
    for (const auto & r : rows.rows) {
        out["rows"].push_back({{"ids", r.ids}, {"pos", r.pos}, {"decide", r.decide}, {"opts", r.opts}});
    }
    if (req.contains("labels")) out["labels"] = enc.labels;
    std::printf("%s\n", out.dump().c_str());
    std::fflush(stdout);

    llama_model_free(model); llama_backend_free();
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path, reqfile, headfile, casesfile, probs_out, hidden_out, textfile, add_special = "false";
    std::string mode = "headbench";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--mode") mode = next();
        else if (a == "--head" || a == "--headfile") headfile = next();
        else if (a == "--model") model_path = next();
        else if (a == "--req") reqfile = next();
        else if (a == "--cases") casesfile = next();
        else if (a == "--probs-out") probs_out = next();
        else if (a == "--hidden-out") hidden_out = next();
        else if (a == "--text-file") textfile = next();
        else if (a == "--add-special") add_special = next();
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }

    if (mode == "headbench") {
        if (headfile.empty() || casesfile.empty()) {
            std::fprintf(stderr, "usage: headbench --head head.bin --cases cases.json\n");
            return 2;
        }
        return run_headbench(headfile, casesfile);
    }
    if (mode == "readout") {
        if (model_path.empty() || reqfile.empty() || headfile.empty()) {
            std::fprintf(stderr, "usage: headbench --mode readout --model model.gguf --req req.json --head head.bin [--probs-out probs.json] [--hidden-out hidden.json]\n");
            return 2;
        }
        return run_readout(model_path, reqfile, headfile, probs_out, hidden_out);
    }
    if (mode == "tokens") {
        if (model_path.empty() || textfile.empty()) {
            std::fprintf(stderr, "usage: headbench --mode tokens --model model.gguf --text-file t.txt [--add-special true|false]\n");
            return 2;
        }
        return run_tokens(model_path, textfile, add_special);
    }
    if (mode == "encode") {
        if (model_path.empty() || reqfile.empty()) {
            std::fprintf(stderr, "usage: headbench --mode encode --model model.gguf --req req.json\n");
            return 2;
        }
        return run_encode(model_path, reqfile);
    }
    std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
    return 2;
}