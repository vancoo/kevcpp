// kev_encode.cpp — logits 主通道共用: llama-free 分词器访问 (模型无关)
//
// 原 kev 的 KEV marker 布局 encode()/user_tokens()/rows_of() 已随「剔除 kev 兼容、logits
// 通道为主线」删除; 此处仅保留 logits 主通道(KevModel 的 probs_logits)依赖的分词接口。
#include "kev_encode.h"

#include <cstdint>
#include <string>
#include <vector>

namespace kev {

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

} // namespace kev
