// kev_api.cpp — TypeSafe /v1/systemone 映射层(复刻 kev/api.py 语义)
#include "kev_api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using nlohmann::json;

namespace kev {

static constexpr int MAX_OPTIONS = 255;

// TRACE: 仅当环境变量 KEV_TRACE=1 时打印逐环节耗时(性能拆解临时探针, 默认关闭, 不影响任何逻辑)。
static bool env_trace_enabled() {
    static const bool v = std::getenv("KEV_TRACE") != nullptr && std::string(std::getenv("KEV_TRACE")) == "1";
    return v;
}
static double now_us() {
    return (double) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() / 1e3;
}

// ---------- render / option_text (api.py) ----------
static std::string scalar_str(const json & v) {
    if (v.is_null()) return "";
    if (v.is_boolean()) return v.get<bool>() ? "True" : "False";
    if (v.is_number_integer()) return std::to_string(v.get<long long>());
    if (v.is_number_unsigned()) return std::to_string(v.get<unsigned long long>());
    if (v.is_number_float()) {
        double x = v.get<double>();
        if (std::isnan(x)) return "nan";
        if (std::isinf(x)) return x > 0 ? "inf" : "-inf";
        if (x == (double)(long long) x && std::fabs(x) < 1e17) { char b[64]; std::snprintf(b, sizeof b, "%.1f", x); return b; }
        for (int p = 1; p <= 17; ++p) { char b[64]; std::snprintf(b, sizeof b, "%.*g", p, x); if (std::stod(b) == x) return b; }
        char b[64]; std::snprintf(b, sizeof b, "%.17g", x); return b;
    }
    if (v.is_string()) return v.get<std::string>();
    return v.dump();
}

static std::string render_rec(const json & v, int indent);
static std::string lstrip(const std::string & s) {
    size_t i = 0; while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i; return s.substr(i);
}

static std::string render_rec(const json & v, int indent) {
    std::string pad(2 * indent, ' ');
    if (v.is_null()) return "";
    if (v.is_primitive()) return scalar_str(v);
    if (v.is_array()) {
        std::string out;
        bool first = true;
        for (const auto & x : v) {
            std::string r = lstrip(render_rec(x, indent + 1));
            std::string line = pad + "- " + r;
            if (!first) out += "\n";
            out += line; first = false;
        }
        return out;
    }
    std::string out; bool first = true;
    for (auto it = v.begin(); it != v.end(); ++it) {
        std::string line;
        const std::string & k = it.key();
        const json & x = it.value();
        if (x.is_object() || x.is_array()) line = pad + k + ":\n" + render_rec(x, indent + 1);
        else line = pad + k + ": " + scalar_str(x);
        if (!first) out += "\n";
        out += line; first = false;
    }
    return out;
}

std::string render(const json & v) { return render_rec(v, 0); }

static std::string option_text(const std::string & name, const json & desc) {
    if (desc.is_null() || (desc.is_string() && desc.get<std::string>().empty())) return name;
    return name + ": " + render_rec(desc, 0);
}

static std::vector<std::string> question_keys(const std::string & qtype, const json & criteria) {
    std::vector<std::string> k;
    if (qtype == "choice") {
        for (auto it = criteria.begin(); it != criteria.end(); ++it) k.push_back(it.key());
    } else if (qtype == "noul") {
        k = {"false", "true"};
    } else {
        for (int i = 0; i < (int) criteria.size(); ++i) k.push_back(std::to_string(i));
    }
    return k;
}

bool parse_systemone(const json & body, ParsedSystemOne & out, std::string * err) {
    auto errf = [&](const std::string & m) { if (err) *err = m; return false; };
    if (!body.contains("state")) return errf("missing 'state'");
    if (!body.contains("questions") || !body["questions"].is_object() || body["questions"].empty())
        return errf("missing or empty 'questions'");

    out.rec.state = render_rec(body["state"], 0);
    for (auto it = body["questions"].begin(); it != body["questions"].end(); ++it) {
        const std::string & qid = it.key();
        const json & q = it.value();
        if (!q.contains("type")) return errf("question '" + qid + "' missing type");
        std::string type = q["type"].get<std::string>();
        std::string instr = q.contains("instructions") ? render_rec(q["instructions"], 0) : std::string();

        SystemOneMeta m; m.id = qid; m.type = type;
        std::vector<std::string> opts;
        if (type == "noul") {
            json crit = (q.contains("criteria") && q["criteria"].is_object()) ? q["criteria"] : json::object();
            json nullv(nullptr);
            opts.push_back(option_text("no",  crit.contains("false") ? crit["false"] : nullv));
            opts.push_back(option_text("yes", crit.contains("true")  ? crit["true"]  : nullv));
        } else if (type == "choice") {
            if (!q.contains("criteria") || !q["criteria"].is_object() || q["criteria"].empty())
                return errf("choice '" + qid + "' needs criteria entries");
            for (auto cit = q["criteria"].begin(); cit != q["criteria"].end(); ++cit)
                opts.push_back(option_text(cit.key(), cit.value()));
        } else if (type == "score") {
            if (!q.contains("criteria") || !q["criteria"].is_array() || q["criteria"].empty())
                return errf("score '" + qid + "' needs criteria levels");
            for (const auto & x : q["criteria"]) opts.push_back(render_rec(x, 0));
            m.legend = opts;
        } else {
            return errf("unsupported question type: " + type);
        }
        if ((int) opts.size() < 1 || (int) opts.size() > MAX_OPTIONS)
            return errf(type + " '" + qid + "' needs 1.." + std::to_string(MAX_OPTIONS) + " options");
        m.keys = question_keys(type, type == "score" ? q["criteria"] : q["criteria"]);

        KevQuestion kq; kq.instr = instr; kq.options = opts;
        out.rec.questions.push_back(std::move(kq));
        out.meta.push_back(std::move(m));
    }
    out.rec.labels.assign(out.rec.questions.size(), 0);
    return true;
}

// ---------- confidence / to_answers (api.py) ----------
static std::vector<double> normalize(const std::vector<double> & p) {
    double t = 0; for (double x : p) t += x;
    if (t == 0) { std::vector<double> u(p.size(), 1.0 / p.size()); return u; }
    std::vector<double> n; for (double x : p) n.push_back(x / t); return n;
}
static double choice_confidence(const std::vector<double> & p) {
    int K = (int) p.size(); if (K == 1) return 1.0;
    auto n = normalize(p);
    double mx = *std::max_element(n.begin(), n.end());
    return (mx - 1.0 / K) / (1.0 - 1.0 / K);
}
static double score_confidence(const std::vector<double> & p) {
    int L = (int) p.size(); if (L == 1) return 1.0;
    auto n = normalize(p);
    int mode = 0; for (int i = 1; i < L; ++i) if (n[i] > n[mode]) mode = i;
    double D = 0; for (int i = 0; i < L; ++i) D += std::fabs(i - (L - 1) / 2.0); D /= L;
    double e = 0; for (int i = 0; i < L; ++i) e += n[i] * std::fabs(i - mode);
    return std::max(0.0, 1.0 - e / D);
}
static double round_prob(double x) {
    double s = std::copysign(1.0, x);
    x = std::fabs(x) * 1e4;
    double f = std::floor(x), diff = x - f;
    if (diff > 0.5) f += 1.0;
    else if (diff == 0.5 && std::fmod(f, 2.0) != 0.0) f += 1.0;
    return s * f / 1e4;
}

json to_answers(const std::vector<std::vector<float>> & probs,
                const std::vector<SystemOneMeta> & meta) {
    json answers = json::object();
    for (size_t q = 0; q < meta.size(); ++q) {
        const auto & M = meta[q];
        std::vector<double> p; for (float x : probs[q]) p.push_back(x);
        if (M.type == "noul") {
            answers[M.id] = {{"type", "noul"}, {"noul", round_prob(p[1])}};
        } else if (M.type == "choice") {
            int argmax = (int) (std::max_element(p.begin(), p.end()) - p.begin());
            json dist; for (size_t i = 0; i < M.keys.size(); ++i) dist[M.keys[i]] = round_prob(p[i]);
            answers[M.id] = {{"type", "choice"}, {"choice", M.keys[argmax]},
                             {"confidence", round_prob(choice_confidence(p))}, {"probabilities", std::move(dist)}};
        } else {
            double score = 0; for (size_t i = 0; i < p.size(); ++i) score += i * p[i];
            json legend; for (size_t i = 0; i < M.legend.size(); ++i) legend[std::to_string(i)] = M.legend[i];
            json dist; for (size_t i = 0; i < p.size(); ++i) dist[std::to_string(i)] = round_prob(p[i]);
            answers[M.id] = {{"type", "score"}, {"score", round_prob(score)}, {"legend", std::move(legend)},
                             {"probabilities", std::move(dist)}, {"confidence", round_prob(score_confidence(p))}};
        }
    }
    return answers;
}

json systemone_json(KevModel & model, const json & body, double * t_model_ms, std::string * err) {
    const bool TR = env_trace_enabled();
    double w = 0.0, t_parse_c = 0.0, t_enc_c = 0.0, t_probs_c = 0.0, t_ans_c = 0.0, t_ct_c = 0.0;
    if (TR) w = now_us();
    ParsedSystemOne ps;
    if (!parse_systemone(body, ps, err)) return json(nullptr);
    if (TR) { t_parse_c = now_us() - w; w = now_us(); }

    std::string enc_err;
    KevEnc enc = model.encode(ps.rec, &enc_err);
    int input_tokens = (int) enc.ids.size();
    if (TR) { t_enc_c = now_us() - w; w = now_us(); }

    // 防御性预检: 一个 sequence(branch) 需在单注意力流内放下完整 state 前缀(Ls)
    // + 自身 branch tokens(B), 即 input_tokens 格; 若超过每流容量 n_ctx_seq,
    // decode 必然 find_slot 失败。在真正 decode 前给出清晰错误, 取代原先
    // cryptic 的 {"detail":"could not answer request"}(由 "init_batch: failed to
    // prepare attention ubatches" 触发)。纯预检, 不触碰成功路径的 probs/数值。
    const int n_ctx_seq = model.n_ctx_seq_capacity();
    if (n_ctx_seq > 0 && (int) enc.ids.size() > n_ctx_seq) {
        if (t_model_ms) *t_model_ms = 0.0;
        if (err && err->empty())
            *err = "state too long for context: " + std::to_string(input_tokens)
                 + " tokens exceed per-stream capacity " + std::to_string(n_ctx_seq)
                 + " (raise --ctx N, e.g. --ctx 8192, or split the input)";
        if (TR) std::fprintf(stderr, "[TRACE] sysone rejected input=%d > n_ctx_seq=%d (pre-decode capacity guard)\n",
                             input_tokens, n_ctx_seq);
        return json(nullptr);
    }

    auto t0 = std::chrono::steady_clock::now();
    // P.1: 复用已算好的 enc, 避免 probs(req) 内部再 encode+rows_of 一遍(消除原两次重复 encode + 正则重写)。
    auto res = model.probs(enc, &enc_err);
    double model_ms = (double) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0).count() / 1e3;
    if (TR) t_probs_c = now_us() - w;
    if (t_model_ms) *t_model_ms = model_ms;
    if (res.probs.empty()) {
        // 失败时把内部错误上抛给调用方(维持原 422 detail 语义)。
        if (err && err->empty() && !enc_err.empty()) *err = enc_err;
        return json(nullptr);
    }

    std::string model_name = body.value("model", std::string("kev-latest"));
    if (TR) w = now_us();
    auto answers = to_answers(res.probs, ps.meta);
    std::string answers_str = answers.dump();
    if (TR) { t_ans_c = now_us() - w; w = now_us(); }
    int output_tokens = model.count_tokens(answers_str);
    if (TR) t_ct_c = now_us() - w;

    if (TR) {
        std::fprintf(stderr, "[TRACE] sysone parse_systemone=%.2fms encode=%.2fms probs(model_ms)=%.2fms "
                             "to_answers+dump=%.2fms count_tokens=%.2fms answers_str_len=%zu Q=%zu out_tok=%d\n",
                     t_parse_c, t_enc_c, t_probs_c, t_ans_c, t_ct_c, answers_str.size(),
                     ps.meta.size(), output_tokens);
    }

    json resp;
    resp["model"] = model_name;
    resp["answers"] = std::move(answers);
    resp["usage"] = {{"input_tokens", input_tokens}, {"output_tokens", output_tokens}};
    resp["latency_ms"] = std::round(model_ms * 10.0) / 10.0;
    return resp;
}

} // namespace kev