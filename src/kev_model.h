// kev_model.h — 组装层: llama 上下文 + 编码器 + PointerHead + 状态前缀缓存 (M3)
//
// 对应 kev 的 DecisionModel 接口子集(probs / encode / prefix 缓存), 纯 CPU, 混合架构 hybrid 走 row form:
//   - state 只 prefill 一次(缓存到 LRU slot 的 KV + Gated DeltaNet 反复式状态)。
//   - 每个 question 作为 continue row: llama_memory_seq_cp(slot -> scratch), 只解码 branch,
//     于 decide/</opt> 取 hidden, 过 PointerHead 得概率。
//   - prefix_min_tokens=0(hybrid)。
//   - 同 state 的后续请求命中缓存, 不重跑 state prefill。
#pragma once

#include "kev_encode.h"
#include "kev_head.h"
#include "kev_prefix.h"
#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

namespace kev {

struct KevModelOptions {
    std::string model_path;
    std::string head_path;
    std::string lora_path;      // 可选 llama.cpp GGUF lora 适配器 (M5)
    float       lora_scale = 1.0f;
    int n_ctx          = 4096;
    int n_threads      = 0;      // 0 = auto
    int lru_capacity   = 2;      // 状态缓存 slot 数
    // 派生: n_seq_max = lru_capacity + 1(scratch)
};

class KevModel {
public:
    KevModel() = default;
    ~KevModel();
    KevModel(const KevModel &) = delete;
    KevModel & operator=(const KevModel &) = delete;

    bool init(const KevModelOptions & opt, std::string * err = nullptr);
    void close();

    // 组装好的 encode (M2)
    KevEnc encode(const KevRequest & req, std::string * err = nullptr) const;
    KevRows rows_of(const KevEnc & enc, std::string * err = nullptr) const;

    // 一次请求的概率(含状态缓存命中/未命中与计时)
    struct ProbsResult {
        std::vector<std::vector<float>> probs;  // [Q][K]
        std::vector<std::vector<float>> hidden_decide; // [Q][1024] decide 隐藏向量 (调试/对拍)
        std::vector<double> rows_ms_per_q;      // [Q] 每问 branch decode(+seq_cp 复制) 耗时 ms (M7)
        bool    state_hit = false;
        double  t_state_ms  = 0.0;   // ensure_state 耗时(命中态=缓存查找;未命中=prefill)
        double  t_rows_ms   = 0.0;   // 全部 question branch decode + head
        double  t_total_ms  = 0.0;
    };
    ProbsResult probs(const KevRequest & req, std::string * err = nullptr);

    // 诊断
    const PrefixLRU & lru() const { return lru_; }
    int n_seq_max() const { return n_seq_max_; }
    int scratch_seq() const { return scratch_seq_; }
    int threads() const { return n_threads_; }
    int threads_actual() const { return ctx_ ? llama_n_threads(ctx_) : 0; }
    int threads_batch_actual() const { return ctx_ ? llama_n_threads_batch(ctx_) : 0; }

    // 用 GGUF 分词器统计文本 token 数(output_tokens 计费用: api.output_tokens)
    int count_tokens(const std::string & text) const;

    // 汇报 head 维度与温度(供 /v1/models)
    int   dim_in()  const { return head_.dim_in(); }
    float temperature() const { return head_.temperature(); }

private:
    // 确保 state 在缓存中; 返回其 slot seq(命中或刚 prefill)。ps_state 输出 prefill 耗时。
    int ensure_state(const std::vector<int32_t> & state_ids, double * ps_state, bool * hit);
    // 用缓存的 slot 对每问做 readout
    std::vector<std::vector<float>> rows_from_slot(const KevRows & rows, int slot, double * t_rows,
                                                   std::vector<std::vector<float>> * out_hidden = nullptr,
                                                   std::vector<double> * rows_ms = nullptr);
    bool decode_seq(const std::vector<int32_t> & tokens, int start_pos, llama_seq_id seq, std::string * err);
    std::vector<float> hidden_at_output(int out_idx, std::string * err);  // 从上次 decode 的 output 取 hidden[1024]

    llama_context  * ctx_   = nullptr;
    llama_model    * model_ = nullptr;
    llama_adapter_lora * lora_ = nullptr;   // M5: 可选 LoRA 适配器
    const llama_vocab * vocab_ = nullptr;
    PointerHead  head_;
    PrefixLRU    lru_{2, 1};
    int n_seq_max_   = 0;
    int scratch_seq_ = 1;
    int n_threads_   = 0;
    bool ok_         = false;
};

} // namespace kev