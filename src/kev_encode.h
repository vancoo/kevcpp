// kev_encode.h — logits 判别主通道的共享类型 + llama-free 分词器访问 (模型无关)
//
// 在「以 logits 判别通道为主线」的重构后, 不再保留原 kev 的 KEV 指针头/KEV marker 布局
// encode(KevEnc/rows_of/SPECIAL/OPT_* 等)。这里只保留 logits 主通道(Model + API layer)
// 需要的基础:
//   - TokenFeeder:   llama-free 分词器访问器(单一 llama 模块负责真实分词, 见
//                    docs/portable_backend_deployment.md; host 经 backend ops 转发)。
//   - KevRequest / KevQuestion: 一次判别请求的模型无关结构化表示(state + 每问指令/候选/keys)。
//   - tokenize / special_token_id: llama-free 分词接口(logits 主通道用)。
#pragma once

#include "kev_config.h"

#include <cstdint>
#include <string>
#include <vector>

namespace kev {

// llama-free tokenizer accessor (B-3). host 不触碰 llama 类型: 调用方提供 TokenFeeder,
// 由其转发到 backend 的 tokenize / special_token_id ops(单一 llama 规则, dll 拥有 llama_vocab)。
//
// tokenize returns count written (>=0), or -required-capacity (negative) when
// `cap` is too small (writes nothing).
struct TokenFeeder {
    void * ud = nullptr;
    int32_t (*tokenize)(void * ud, const char * text, int32_t text_len,
                        bool add_special, bool parse_special,
                        int32_t * out_ids, int32_t cap);
    int32_t (*special_token_id)(void * ud, const char * special);
};

struct KevQuestion {
    std::string instr;
    std::vector<std::string> options;
    // 每题判别的候选 key(choice: criteria keys; noul: ["false","true"]; score: ["0".."L-1"])。
    // 由 API 层(parse_systemone)填充, 供「生成式 logits 判别」通道用作候选答案 token 来源;
    // 空时 logits 通道会按 options 推导。
    std::vector<std::string> keys;
};

struct KevRequest {
    std::string state;
    std::vector<KevQuestion> questions;
    std::vector<int32_t> labels;   // 每问的 label(仅透传, 判别逻辑不用于算术)
};

// 分词器抽象: 返回 text 的 token ids。
std::vector<int32_t> tokenize(const TokenFeeder & tf, const std::string & text,
                              bool add_special, bool parse_special);

// 查找特殊 token id: 对 "<|name|>" parse_special=true 分词取最后一个 id(前面的伪 token 忽略)。
int32_t special_token_id(const TokenFeeder & tf, const char * special);

} // namespace kev
