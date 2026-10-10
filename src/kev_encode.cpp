// kev_encode.cpp — 复刻 kev.model encode()/user_tokens()/rows_of()
#include "kev_encode.h"

#include <algorithm>
#include <cstdio>
#include <regex>
#include <string>

namespace kev {

// 从决策配置解析 5 个 marker 的 token id; 任一缺失/非原子 token 则 *err 给出清晰错误并返回 false。
// 模型无关: 一个合法 marker 必须是词表中的"单个原子 token"(即真正的 special token, 如
// <|box_end|> tokenize 得 1 个 id)。若配置的 marker 在词表里不存在, tokenize 会把字面量
// 拆成多个普通 BPE token(fallback), 这里拒绝之并给出可操作的错误信息。
namespace {
struct MarkerIds { int32_t state, question, opt_start, opt_end, decide; };
bool resolve_markers(const TokenFeeder & tf, const KevDecisionConfig & cfg,
                     MarkerIds & ids, std::string * err) {
    auto need = [&](const char * name, const std::string & marker, int32_t & out) -> bool {
        auto v = tokenize(tf, marker, /*add_special=*/false, /*parse_special=*/true);
        if (v.size() != 1) {
            if (err && err->empty()) {
                *err = std::string("model lacks decision marker '") + name + "' = \"" + marker
                     + "\": it is not a single token in this model's vocabulary (tokenized to "
                     + std::to_string((long) v.size()) + " token(s)). "
                       "Configure an existing marker via --decide-config / --opt-end / --decide-token / KevDecisionConfig.";
            }
            return false;
        }
        out = v[0];
        return true;
    };
    ids.state     = -1; ids.question = -1; ids.opt_start = -1; ids.opt_end = -1; ids.decide = -1;
    return need("state",       cfg.state_marker,    ids.state)
        && need("question",    cfg.question_marker, ids.question)
        && need("option_start",cfg.option_start,    ids.opt_start)
        && need("option_end",  cfg.option_end,      ids.opt_end)
        && need("decide",      cfg.decide_marker,   ids.decide);
}
} // namespace

std::vector<int32_t> tokenize(const TokenFeeder & tf, const std::string & text,
                              bool add_special, bool parse_special) {
    if (!tf.tokenize) return {};
    // size by a call, then fill: tokenize returns count, or -required on overflow.
    int32_t cap = (int32_t) text.size() * 4 + 16;
    for (;;) {
        if (cap <= 0) return {};
        std::vector<int32_t> buf((size_t) cap);
        const int32_t n = tf.tokenize(tf.ud, text.c_str(), (int32_t) text.size(),
                                      add_special, parse_special, buf.data(), cap);
        if (n >= 0) { buf.resize((size_t) n); return buf; }
        cap = -n;   // exact capacity required by the tokenizer
    }
}

int32_t special_token_id(const TokenFeeder & tf, const char * special) {
    auto v = tokenize(tf, special, /*add_special=*/false, /*parse_special=*/true);
    if (v.empty()) return -1;
    return v.back();   // the <|name|> special token is the last token (preceded by a pre-token artifact)
}

std::vector<int32_t> user_tokens(const TokenFeeder & tf, const std::string & text) {
    // kev.user_tokens: <|x|> -> <¦x¦> then add_special_tokens=False
    static const std::regex re(R"(<\|([A-Za-z0-9_]+)\|>)");
    std::string rewritten = std::regex_replace(text, re, "<¦$1¦>");
    return tokenize(tf, rewritten, /*add_special=*/false, /*parse_special=*/true);
}

KevEnc encode(const TokenFeeder & tf, const KevRequest & rec,
              int max_state, int max_branch, std::string * err, const KevDecisionConfig * cfg) {
    KevEnc e;
    // 模型无关: marker 来自注入配置(默认 Qwen3.5)。缺失则报清晰错误。
    const KevDecisionConfig & dc = cfg ? *cfg : default_decision_config();
    MarkerIds mid;
    if (!resolve_markers(tf, dc, mid, err)) return e;

    // state
    auto state_tokens = user_tokens(tf, rec.state);
    if ((int) state_tokens.size() + 1 > max_state) {
        if (err) *err = "state too long";
        state_tokens.resize((size_t) std::max(0, max_state - 1));
        e.labels.resize(rec.questions.size(), -1);
    }
    const int32_t S0 = mid.state;
    const int32_t q_id = mid.question;
    const int32_t o_id = mid.opt_start;
    const int32_t c_id = mid.opt_end;
    const int32_t d_id = mid.decide;

    e.ids.push_back(S0);
    e.seg.push_back(0);
    e.pos.push_back(0);
    e.opt.push_back(OPT_NONE);
    for (size_t i = 0; i < state_tokens.size(); ++i) {
        e.ids.push_back(state_tokens[i]);
        e.seg.push_back(0);
        e.pos.push_back((int32_t)(i + 1));
        e.opt.push_back(OPT_NONE);
    }
    const int32_t Ls = (int32_t) e.ids.size();

    for (size_t k = 0; k < rec.questions.size(); ++k) {
        const auto & q = rec.questions[k];
        auto instr_t = user_tokens(tf, q.instr);
        std::vector<std::vector<int32_t>> spans;
        spans.reserve(q.options.size());
        for (const auto & o : q.options) {
            auto sp = user_tokens(tf, o);
            spans.push_back(sp);
        }
        std::vector<int32_t> br;
        br.push_back(q_id);
        br.insert(br.end(), instr_t.begin(), instr_t.end());
        for (const auto & sp : spans) { br.push_back(o_id); br.insert(br.end(), sp.begin(), sp.end()); br.push_back(c_id); }
        br.push_back(d_id);

        if ((int32_t) br.size() > max_branch - Ls) {
            if (err) *err = "branch too long";
            // truncate to fit (kev raises; here keep consistency with non-strict partial)
            br.resize((size_t) std::max(1, max_branch - Ls));
        }

        const int32_t base = (int32_t) e.ids.size();
        const int32_t p0 = Ls;
        std::vector<int32_t> br_opt;
        br_opt.assign((size_t) 1 + instr_t.size(), OPT_NONE);   // q_id + instr
        for (size_t j = 0; j < spans.size(); ++j) br_opt.insert(br_opt.end(), (int32_t)(2 + spans[j].size()), (int32_t) j); // o_id+content+c_id
        br_opt.push_back(OPT_DECIDE);

        for (int32_t i = 0; i < (int32_t) br.size(); ++i) {
            e.ids.push_back(br[i]);
            e.seg.push_back((int32_t)(k + 1));
            e.pos.push_back(p0 + i);
            e.opt.push_back(br_opt[i]);
        }

        // ends: relative </opt> positions -> absolute
        std::vector<int32_t> ends;
        int32_t cursor = (int32_t)(1 + instr_t.size());   // after q_id+instr
        for (const auto & sp : spans) {
            cursor += (2 + (int32_t) sp.size());           // o_id + content + c_id
            ends.push_back(cursor - 1);                    // c_id is the </opt>
        }
        e.decide_idx.push_back(base + (int32_t) br.size() - 1);
        std::vector<int32_t> opt_abs; for (int32_t en : ends) opt_abs.push_back(base + en);
        e.opt_idx.push_back(std::move(opt_abs));
    }
    e.labels = rec.labels;
    return e;
}

KevRows rows_of(const KevEnc & enc, std::string * err) {
    KevRows out;
    int32_t Ls = 0;
    for (int32_t s : enc.seg) if (s == 0) ++Ls;
    out.state_ids.assign(enc.ids.begin(), enc.ids.begin() + Ls);
    out.state_pos.assign(enc.pos.begin(), enc.pos.begin() + Ls);

    int32_t start = Ls;
    for (size_t k = 0; k < enc.decide_idx.size(); ++k) {
        int32_t d = enc.decide_idx[k];
        int32_t end = d + 1;
        if (start >= (int32_t) enc.ids.size() || end > (int32_t) enc.ids.size() ||
            enc.seg[start] != (int32_t)(k + 1) || enc.seg[end - 1] != (int32_t)(k + 1)) {
            if (err) *err = "branch layout mismatch";
            break;
        }
        KevRow r;
        r.ids.assign(enc.ids.begin() + start, enc.ids.begin() + end);
        r.pos.assign(enc.pos.begin() + start, enc.pos.begin() + end);
        r.decide = d - start;
        for (int32_t o : enc.opt_idx[k]) r.opts.push_back(o - start);
        out.rows.push_back(std::move(r));
        start = end;
    }
    return out;
}

} // namespace kev
