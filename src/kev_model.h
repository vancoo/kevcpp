// kev_model.h — 组装层: llama 上下文 + 编码器 + PointerHead + 状态前缀缓存 (M3)
//
// 对应 kev 的 DecisionModel 接口子集(probs / encode / prefix 缓存), 纯 CPU, 混合架构 hybrid 走 row form:
//   - state 只 prefill 一次(缓存到 LRU slot 的 KV + Gated DeltaNet 反复式状态)。
//   - 每个 question 作为 continue row: llama_memory_seq_cp(slot -> scratch), 只解码 branch,
//     于 decide/</opt> 取 hidden, 过 PointerHead 得概率。
//   - prefix_min_tokens=0(hybrid)。
//   - 同 state 的后续请求命中缓存, 不重跑 state prefill。
//
// Pluggable-backend (ARCHIVE): KevModel 保留全部组装/编排(LRU、encode 组装、rows 规划、
// PointerHead、compute_mutex_、n_seq 布局 —— 均不扰动), 但 llama/ggml 算子耦合全部改经
// kev::Backend*(来自 kev_backend_factory/BACKEND C-ABI) 调用, 不再直接调 llama_*。见
// docs/portable_backend_deployment.md。encode/tokenize 也经后端: 由 backend 的
// token_feeder()(delegated tokenize/special_token_id, llama-free) 提供(single-llama: 见文档)。
#pragma once

#include "kev_encode.h"
#include "kev_head.h"
#include "kev_prefix.h"
#include "kev_backend_factory.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace kev {

struct KevModelOptions {
    std::string model_path;
    std::string head_path;
    std::string lora_path;      // 可选 llama.cpp GGUF lora 适配器 (M5)
    float       lora_scale = 1.0f;
    // merged-LoRA: 把 lora_path fold 进 model_path 的 Q8_0 权重, 前向走单份权重。
    bool        merge_lora = false;
    std::string merged_model_path;   // 可选: merged GGUF 的落盘位置
    int n_ctx          = 2048;
    int n_threads      = 0;      // 0 = auto
    int lru_capacity   = 2;      // 状态缓存 slot 数
    std::string backend = "cpu"; // pluggable backend name (default "cpu")
    // 派生: n_seq_max = lru_capacity + 1(scratch) + kMaxBatch(branch 池)。
};

class KevModel {
public:
    // P.4: 一次共享同 state 前缀、等长 branch 批量 decode 时最多并行 branch seq 数。
    static constexpr int kMaxBatch = 8;
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
    ProbsResult probs(const KevEnc & enc, std::string * err = nullptr);

    bool prewarm_state(const std::vector<int32_t> & state_ids);
    bool warm_graph(bool warm_branch = true, int state_tokens = 38, int branch_tokens = 151,
                    std::string * err = nullptr);

    // 诊断
    const PrefixLRU & lru() const { return lru_; }
    int n_seq_max() const { return n_seq_max_; }
    int scratch_seq() const { return scratch_seq_; }
    int branch_seq(int k) const { return branch_seq_base_ + k; }
    int max_batch() const { return kMaxBatch; }
    void set_batch_enabled(bool v) { batch_enabled_ = v; }
    bool batch_enabled() const { return batch_enabled_; }
    int threads() const { return n_threads_; }
    int threads_actual() const { return backend_ ? backend_->threads() : 0; }
    int threads_batch_actual() const { return backend_ ? backend_->threads_batch() : 0; }

    // 每流注意力 KV 容量(n_ctx_seq)。见旧注释: 供 API 层提前判定 request 是否超容量。
    int n_ctx_seq_capacity() const { return backend_ ? backend_->n_ctx_seq_capacity() : 0; }

    int count_tokens(const std::string & text) const;

    int   dim_in()  const { return head_.dim_in(); }
    float temperature() const { return head_.temperature(); }

    // Pluggable backend metadata (reported in /v1/models + startup log)
    const char * backend_name()   const { return backend_ ? backend_->name()   : "(none)"; }
    const char * backend_device() const { return backend_ ? backend_->device() : "(none)"; }
    const char * backend_version() const { return backend_ ? backend_->version() : "(none)"; }

private:
    int ensure_state(const std::vector<int32_t> & state_ids, double * ps_state, bool * hit);
    std::vector<std::vector<float>> rows_from_slot(const KevRows & rows, int slot, double * t_rows,
                                                   std::vector<std::vector<float>> * out_hidden = nullptr,
                                                   std::vector<double> * rows_ms = nullptr);

    // 解码 token 序列(续接在 start_pos 之后,写入 seq)。经 backend。
    bool decode_seq(const std::vector<int32_t> & tokens, int start_pos, int seq,
                    std::string * err, const std::vector<int32_t> & logits_pos);
    // P.4: 单次 decode 多个等长 branch(共享 state 前缀)。经 backend。
    bool decode_batch(const std::vector<BackendBranch> & branches, std::string * err);

    // 从上次 decode 的 output 取 hidden[1024](经 backend read_hidden)
    std::vector<float> hidden_at_output(int out_idx, std::string * err);

    std::unique_ptr<Backend> backend_;
    PointerHead  head_;
    PrefixLRU    lru_{2, 1};
    int n_seq_max_   = 0;
    int scratch_seq_ = 1;
    int branch_seq_base_ = 1;
    int n_threads_   = 0;
    bool batch_enabled_ = true;
    bool ok_         = false;
    // P0b: 串行化所有访问共享 llama ctx/lru/scratch 的计算段。
    std::mutex compute_mutex_;
};

} // namespace kev
