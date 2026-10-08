// kev_encode.h — 复刻 kev.model.encode() 编码层 (M2)
//
// 目标: 与 E:\van\projects\kev\kev\model.py 的 encode()/user_tokens()/rows_of() 逐 token/逐索引
// 对齐。分词用 GGUF 自带 llama_tokenize(等价于 HF fast tokenizer, 已对拍确认)。
//
//   SPECIAL = ["<|fim_prefix|>","<|fim_middle|>","<|box_start|>","<|box_end|>","<|fim_suffix|>"]
//       SPECIAL[0]=state 开头; q/o/c/d = SPECIAL[1..3]; d(decide)=SPECIAL[4].
//   user_tokens(tok,text): <|x|> -> <¦x¦> 后 add_special_tokens=False 分词.
//   encode(): [<state>...] 每个 question [<q> instr <o> opt </o>... <decide>].
//       ids/seg/pos/opt/decide_idx/opt_idx 与 Python 一致. opt: OPT_NONE=-1(指令), j(选项span), OPT_DECIDE=-2(decide).
//   rows_of(enc): 拆成 state_ids/state_pos + 每问 branch row(ids/pos/decide/opts 相对偏移).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kev {

// llama-free tokenizer accessor (B-3). kev_encode never touches llama types: the
// caller supplies a TokenFeeder that performs tokenize / special-token lookup.
//
//  * cpu/gpu plugin path: built from a kev::Backend (kev_backend_factory.h),
//    forwarding to the backend's tokenize / special_token_id ops (single-llama:
//    the dll owns the llama_vocab).
//  * legacy static bench tool path (headbench, etc., which link llama directly):
//    built from a const llama_vocab* in a llama-linking TU (see kev_encode.h's
//    llama-free contract — the adapter itself lives where llama is statically
//    linked, not in kev_encode.cpp).
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

constexpr int32_t OPT_NONE  = -1;
constexpr int32_t OPT_DECIDE = -2;

const char * const SPECIAL[5] = {
    "<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"};

struct KevOption { std::string text; };

struct KevQuestion {
    std::string instr;
    std::vector<std::string> options;
};

struct KevRequest {
    std::string state;
    std::vector<KevQuestion> questions;
    std::vector<int32_t> labels;   // 每问的 label(仅透传, encode 不用于算术)
};

struct KevEnc {
    std::vector<int32_t> ids;         // full packed ids
    std::vector<int32_t> seg;         // 0=state, k=question k
    std::vector<int32_t> pos;
    std::vector<int32_t> opt;
    std::vector<int32_t> decide_idx;  // [Q] 每个 </decision>/<decide> 在 ids 中的绝对索引
    std::vector<std::vector<int32_t>> opt_idx; // [Q][K] 每个 </opt> 在 ids 中的绝对索引
    std::vector<int32_t> labels;
};

// 一个 branch row: question k 的 branch tokens 及其相对 decide/opts 偏移
struct KevRow {
    std::vector<int32_t> ids;
    std::vector<int32_t> pos;
    int32_t decide = 0;                 // decide 在本 row 内的偏移
    std::vector<int32_t> opts;          // 各 </opt> 在本 row 内的偏移
};

struct KevRows {
    std::vector<int32_t> state_ids;
    std::vector<int32_t> state_pos;
    std::vector<KevRow>  rows;          // 每个 question 一个
};

// 分词器抽象: 返回 text (已按需改写) 的 token ids, 序列 = HF fast tokenizer(addr_special=False)
//   parse_special 语义等同 HF split_special_tokens; 对 Qwen3.5 已对拍 == HF fast.
std::vector<int32_t> tokenize(const TokenFeeder & tf, const std::string & text,
                              bool add_special, bool parse_special);

// 查找特殊 token id: 对 "<|name|>" parse_special=true 分词取最后一个 id(前面的伪 token 忽略).
int32_t special_token_id(const TokenFeeder & tf, const char * special);

// user_tokens: <|x|> -> <¦x¦> 再 add_special=False 分词
std::vector<int32_t> user_tokens(const TokenFeeder & tf, const std::string & text);

// encode(): 完整打包, 与 kev.model.encode(option_isolation=False) 对齐.
// max_state/max_branch 语义: 返回状态是否被截断; 超限置 state_truncated / throw_on_overflow.
KevEnc encode(const TokenFeeder & tf, const KevRequest & rec,
              int max_state = 384, int max_branch = 1024,
              std::string * err = nullptr);

// rows_of(enc): 拆成 state + per-question rows.
KevRows rows_of(const KevEnc & enc, std::string * err = nullptr);

} // namespace kev
