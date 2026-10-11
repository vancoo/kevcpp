// kev_config.cpp — 决策配置的轻量 JSON 加载
#include "kev_config.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

namespace kev {

// 从外部 JSON 文件加载 few-shot 示例库到 out(追加)。文件根可为数组或 {"examples": [...]},
// 每项 {"state_text": "...", "answer_key": "..."}。失败返 false 且 *err 说明原因。
bool load_fewshot_examples_file(const std::string & file,
                                std::vector<std::pair<std::string, std::string>> & out,
                                std::string * err);

// KevDecisionConfig::FewShot::render — 把示例库渲染成放回 prompt 的 few-shot 段。
// 模型无关: 纯文本逐行拼装; 固定样式 -> 对每请求一致 -> 可并入稳定公共前缀而不破坏 P2 缓存。
std::string KevDecisionConfig::FewShot::render() const {
    std::string s;
    s.reserve(64 + examples.size() * 96);
    if (!examples.empty()) s += "示例：\n";
    for (const auto & e : examples) {
        s += "- ";
        s += e.first;
        s += " → ";
        s += e.second;
        s += '\n';
    }
    return s;
}

namespace {

// 极小 JSON 对象解析器, 只支持决策配置 schema 需要的形状:
//   顶层对象, 可含 string / number / bool / null 标量字段, 以及一个嵌套对象字段。
//   不支持数组/嵌套多层 (决策配置不需要)。解析失败返回 false 并置 *err。
struct TinyJson {
    // 顶层对象解析。每遇到一个叶子键值对回调 on_pair(ud, path, rawValue, isString);
    // path 形如 "markers.option_end"(嵌套用 '.' 平铺)。返回是否成功。
    bool parse(const std::string & text,
               int (*on_pair)(void * ud, const std::string & path, const std::string & raw_value, bool is_string),
               void * ud, std::string * err) {
        src_ = &text; pos_ = 0; err_ = err;
        skip_ws();
        bool ok = parse_object(on_pair, ud, "");
        if (ok) { skip_ws(); if (pos_ < text.size()) ok = fail("trailing content after JSON root"); }
        return ok;
    }

    // 指定 few_shot.examples 的落地目标向量(parse 前调用)。
    void set_examples_target(std::vector<std::pair<std::string, std::string>> * out) { ex_out_ = out; }

    // 解析一个"示例库文档"(根为数组或 {"examples": [...]}): 结果追加到 ex_out_。
    // 用于 few_shot.examples_file 加载。
    bool parse_examples_document(const std::string & text) {
        src_ = &text; pos_ = 0;
        skip_ws();
        if (pos_ >= src_->size()) return fail("empty examples file");
        if ((*src_)[pos_] == '[') return parse_examples_array();
        if ((*src_)[pos_] != '{') return fail("expected '[' or '{' in examples file");
        ++pos_;  // {
        skip_ws();
        if (eat('}')) return true;  // empty object
        for (;;) {
            std::string key;
            if (!parse_string(key)) return false;
            if (!eat(':')) return fail("expected ':'");
            skip_ws();
            if (pos_ < src_->size() && (*src_)[pos_] == '[') {
                if (key == "examples") { if (!parse_examples_array()) return false; }
                else return fail("unsupported array key in examples file (expected 'examples')");
            } else if (pos_ < src_->size() && (*src_)[pos_] == '{') {
                if (!parse_object(nullptr, nullptr, key)) return false;
            } else if (pos_ < src_->size() && (*src_)[pos_] == '"') {
                std::string v; if (!parse_string(v)) return false;
            } else {
                std::string raw; if (!parse_scalar(raw)) return false;
            }
            if (eat(',')) continue;
            if (eat('}')) return true;
            return fail("expected ',' or '}'");
        }
    }

private:
    const std::string * src_ = nullptr;
    size_t pos_ = 0;
    std::string * err_ = nullptr;
    // few-shot examples 落地目标(由调用方在 parse 前设为目标向量) — 仅 few_shot.examples 使用。
    std::vector<std::pair<std::string, std::string>> * ex_out_ = nullptr;

    bool fail(const std::string & m) { if (err_ && err_->empty()) *err_ = m; return false; }
    void skip_ws() { while (pos_ < src_->size() && std::isspace((unsigned char)(*src_)[pos_])) ++pos_; }

    bool eat(char c) { skip_ws(); if (pos_ < src_->size() && (*src_)[pos_] == c) { ++pos_; return true; } return false; }

    bool parse_string(std::string & out) {
        skip_ws();
        if (pos_ >= src_->size() || (*src_)[pos_] != '"') return fail("expected string");
        ++pos_;
        out.clear();
        while (pos_ < src_->size()) {
            char c = (*src_)[pos_++];
            if (c == '"') return true;
            if (c == '\\') {
                if (pos_ >= src_->size()) return fail("bad escape");
                char e = (*src_)[pos_++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        if (pos_ + 4 > src_->size()) return fail("bad \\u escape");
                        unsigned v = 0;
                        for (int k = 0; k < 4; ++k) {
                            char h = (*src_)[pos_++];
                            int d;
                            if (h >= '0' && h <= '9') d = h - '0';
                            else if (h >= 'a' && h <= 'f') d = h - 'a' + 10;
                            else if (h >= 'A' && h <= 'F') d = h - 'A' + 10;
                            else return fail("bad \\u hex");
                            v = v * 16 + (unsigned) d;
                        }
                        if (v < 0x80) out.push_back((char) v);
                        else if (v < 0x800) { out.push_back((char)(0xC0 | (v >> 6))); out.push_back((char)(0x80 | (v & 0x3F))); }
                        else { out.push_back((char)(0xE0 | (v >> 12))); out.push_back((char)(0x80 | ((v >> 6) & 0x3F))); out.push_back((char)(0x80 | (v & 0x3F))); }
                        break;
                    }
                    default: return fail("bad escape char");
                }
            } else {
                out.push_back(c);
            }
        }
        return fail("unterminated string");
    }

    bool parse_scalar(std::string & raw) {
        skip_ws();
        size_t start = pos_;
        while (pos_ < src_->size()) {
            char c = (*src_)[pos_];
            if (c == ',' || c == '}' || c == ']' || c == ' ' || c == '\t' || c == '\n' || c == '\r') break;
            ++pos_;
        }
        if (pos_ == start) return fail("expected value");
        raw = src_->substr(start, pos_ - start);
        return true;
    }

    // 解析 "few_shot.examples" 数组: [ { "state_text": "...", "answer_key": "..." }, ... ]。
    // 数组是决策配置里唯一支持的数组形状; 其它位置遇到 '[' 保持"not supported"报错。
    // 结果追加到 ex_out_(由调用方在 parse 前设为目标 examples 向量)。
    bool parse_examples_array() {
        if (!eat('[')) return fail("expected '['");
        skip_ws();
        if (eat(']')) return true;  // empty examples
        for (;;) {
            skip_ws();
            if (!eat('{')) return fail("expected '{' in examples element");
            skip_ws();
            std::string state_text, answer_key;
            if (pos_ < src_->size() && (*src_)[pos_] == '}') {
                ++pos_;  // empty object -> example with empty fields
            } else {
                for (;;) {
                    std::string key;
                    if (!parse_string(key)) return false;
                    if (!eat(':')) return fail("expected ':' in example element");
                    std::string val;
                    if (!parse_string(val)) return false;
                    if (key == "state_text")       state_text  = val;
                    else if (key == "answer_key")  answer_key  = val;
                    if (eat(',')) continue;
                    if (eat('}')) break;
                    return fail("expected ',' or '}' in example element");
                }
            }
            if (ex_out_) ex_out_->emplace_back(state_text, answer_key);
            if (eat(',')) continue;
            if (eat(']')) return true;
            return fail("expected ',' or ']' in examples array");
        }
    }

    bool parse_object(int (*on_pair)(void *, const std::string &, const std::string &, bool), void * ud, const std::string & prefix) {
        skip_ws();
        if (pos_ >= src_->size() || (*src_)[pos_] != '{') return fail("expected '{'");
        ++pos_;  // {
        skip_ws();
        if (pos_ < src_->size() && (*src_)[pos_] == '}') { ++pos_; return true; }  // empty object
        for (;;) {
            std::string key;
            if (!parse_string(key)) return false;
            if (!eat(':')) return fail("expected ':'");
            skip_ws();
            std::string path = prefix.empty() ? key : (prefix + "." + key);
            if (pos_ < src_->size() && (*src_)[pos_] == '{') {
                if (!parse_object(on_pair, ud, path)) return false;
            } else if (pos_ < src_->size() && (*src_)[pos_] == '"') {
                std::string val;
                if (!parse_string(val)) return false;
                if (on_pair) on_pair(ud, path, val, true);
            } else if (pos_ < src_->size() && (*src_)[pos_] == '[') {
                if (path == "few_shot.examples") {
                    if (!parse_examples_array()) return false;
                } else {
                    return fail("arrays not supported in decision config (only few_shot.examples)");
                }
            } else {
                std::string raw;
                if (!parse_scalar(raw)) return false;
                if (on_pair) on_pair(ud, path, raw, false);
            }
            if (eat(',')) continue;
            if (eat('}')) return true;
            return fail("expected ',' or '}'");
        }
    }
};

} // namespace

bool load_decision_config(const std::string & file, KevDecisionConfig & cfg, std::string * err) {
    std::ifstream f(file, std::ios::binary);
    if (!f) {
        if (err) *err = "cannot open decision config: " + file;
        return false;
    }
    std::stringstream ss; ss << f.rdbuf();
    const std::string text = ss.str();

    TinyJson j;
    j.set_examples_target(&cfg.fewshot.examples);
    bool ok = j.parse(text,
        [](void * ud, const std::string & path, const std::string & raw, bool is_string) -> int {
            auto * c = static_cast<KevDecisionConfig *>(ud);
            if (is_string) {
                if (path == "markers.state")              c->state_marker    = raw;
                else if (path == "markers.question")      c->question_marker = raw;
                else if (path == "markers.option_start")  c->option_start    = raw;
                else if (path == "markers.option_end")    c->option_end      = raw;
                else if (path == "markers.decide")        c->decide_marker   = raw;
                else if (path == "logits.prompt_template") c->prompt_template = raw;
                else if (path == "logits.aggregation")    c->aggregation     = raw;
                else if (path == "logits.logits_type")    c->logits_type     = raw;
                else if (path == "few_shot.examples_file") c->fewshot.examples_file = raw;
            } else {
                if (path == "temperature") {
                    float v = (float) std::atof(raw.c_str());
                    if (v > 0.0f) c->temperature = v;
                } else if (path == "logits.enable") {
                    bool v = raw == "true" || raw == "1";
                    c->enable_logits = v;
                } else if (path == "logits.cache_prefix") {
                    bool v = raw == "true" || raw == "1";
                    c->cache_prefix = v;
                } else if (path == "few_shot.enabled") {
                    bool v = raw == "true" || raw == "1";
                    c->fewshot.enabled = v;
                }
            }
            return 1;
        },
        (void *) &cfg, err);
    if (!ok && err && err->empty()) {
        if (err) *err = "invalid decision config: " + file;
    }
    // few_shot.examples_file(可选): 从外部 JSON 加载示例库(根为数组或 {"examples": [...]},
    // 每项 {"state_text","answer_key"})。失败则整个配置加载失败(清晰报错)。
    if (ok && cfg.fewshot.enabled && !cfg.fewshot.examples_file.empty()) {
        std::string fe;
        if (!load_fewshot_examples_file(cfg.fewshot.examples_file, cfg.fewshot.examples, &fe)) {
            if (ok && err && err->empty()) *err = fe;
            ok = false;
        }
    }
    return ok;
}

bool load_fewshot_examples_file(const std::string & file,
                                std::vector<std::pair<std::string, std::string>> & out,
                                std::string * err) {
    std::ifstream f(file, std::ios::binary);
    if (!f) {
        if (err) *err = "cannot open few-shot examples file: " + file;
        return false;
    }
    std::stringstream ss; ss << f.rdbuf();
    const std::string text = ss.str();

    TinyJson j;
    j.set_examples_target(&out);
    if (!j.parse_examples_document(text)) {
        if (err && err->empty()) *err = "invalid few-shot examples file: " + file;
        return false;
    }
    return true;
}

} // namespace kev