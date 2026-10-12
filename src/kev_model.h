// kev_model.h — 组装层: llama 上下文 + 生成式 logits 判别主通道 + 稳定公共前缀缓存 (P2)
//
// logits-first 重构后, 本模块收敛为「以 logits 判别通道为主线」:
//   - 每问按决策配置的 prompt_template 拼装填空式 prompt, 把该问各候选 key 作为「续写答案
//     token」逐个 decode/logits 读取, 对全体候选做相对 softmax -> 各 key 概率。模型无关
//     (见 docs 与 kev_config.h)。
//   - 不再保留原 kev 的 KEV 指针头 / KEV marker 布局 encode / KEV 半边点积打分 / 状态
//     LRU(enable_logits=false 回退)等兼容路径。probs() 恒走 probs_logits。
//   - P2 最长公共前缀(LCP)前缀缓存(prefix_lru_): 缓存 key = 同任务稳定公共前缀(模板 {state} 前
//     的 instructions+options)的 token ids。匹配按 LCP 且受 M-RoPE 约束(须整体复用一个槽的完整
//     前缀): 同一 instructions+options({state} 在模板末尾)的多请求共享稳定前段 KV, 只续算
//     state+Answer; 同报文连发也命中。未命中时 prefill 新前缀到 LRU 槽并登记, 供后续跨请求复用。
//
// Pluggable-backend: KevModel 保留全部组装/编排(LRU、rows 规划、compute_mutex_、n_seq 布局),
// 但 llama/ggml 算子耦合全部改经 kev::Backend*(来自 kev_backend_factory) 调用, 不再直接调
// llama_*。encode/tokenize 也经后端: 由 backend 的 token_feeder()(delegated tokenize) 提供。
#pragma once

#include "kev_encode.h"
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
    std::string lora_path;      // 可选 llama.cpp GGUF lora 适配器(不经 kev_merge 折叠, 原样传给后端)
    float       lora_scale = 1.0f;
    // 决策配置(模型无关 marker + 温度 + logits 通道 prompt/aggregation)。原版 Qwen3.5 用默认;
    // 换模型改这里或 decision_config 文件。temperature / option_end 等旧 CLI 覆盖项也回填到这里。
    KevDecisionConfig decision;
    std::string decision_config;   // 可选 JSON 决策配置文件路径 (加载后覆盖 decision; 见 config/)
    // 便捷覆盖项(CLI): --temperature / --opt-end / --decide-token。非空/+0 则覆盖 decision。
    float       temperature_override  = 0.0f;  // >0 时覆盖 decision.temperature
    std::string option_end_override;          // 非空时覆盖 decision.option_end
    std::string decide_marker_override;       // 非空时覆盖 decision.decide_marker
    int n_ctx          = 2048;
    int n_threads      = 0;      // 0 = auto
    // P2 前缀缓存槽位数(缓存"稳定公共前缀"{instructions}\n[ {options} ]\n 的 KV)。同一类任务的
    // 多请求共享同一前缀 -> 只需 1~2 个槽即命中; 不同 instructions/options 任务各占一个 LRU 槽。
    int prefix_capacity = 4;
    std::string backend = "cpu"; // pluggable backend name (default "cpu")
    // 派生: n_seq_max = scratch(1) + prefix_capacity(前缀池)。
};

class KevModel {
public:
    KevModel() = default;
    ~KevModel();
    KevModel(const KevModel &) = delete;
    KevModel & operator=(const KevModel &) = delete;

    bool init(const KevModelOptions & opt, std::string * err = nullptr);
    void close();

    // 一次请求的概率(生成式 logits 判别主通道)
    struct ProbsResult {
        std::vector<std::vector<float>> probs;  // [Q][K]
        std::vector<double> rows_ms_per_q;      // [Q] 每问 decode+logits 耗时 ms
        double  t_rows_ms  = 0.0;   // 全部 question decode + logits
        double  t_total_ms  = 0.0;
    };
    ProbsResult probs(const KevRequest & req, std::string * err = nullptr);

    // 生成式 logits 判别主通道是否生效(enable_logits 且后端支持 read_logits)。
    bool logits_channel_active() const;

    bool warm_graph(int state_tokens = 38, int branch_tokens = 151,
                    std::string * err = nullptr);

    // 诊断
    int n_seq_max() const { return n_seq_max_; }
    int scratch_seq() const { return scratch_seq_; }
    int threads() const { return n_threads_; }
    int threads_actual() const { return backend_ ? backend_->threads() : 0; }
    int threads_batch_actual() const { return backend_ ? backend_->threads_batch() : 0; }
    int n_embd() const { return backend_ ? backend_->model_n_embd() : 0; }
    int backend_n_batch() const { return backend_ ? backend_->n_batch() : 0; }

    // 每流注意力 KV 容量(n_ctx_seq)。供 API 层提前判定 request 是否超容量。
    int n_ctx_seq_capacity() const { return backend_ ? backend_->n_ctx_seq_capacity() : 0; }

    int count_tokens(const std::string & text) const;

    float temperature() const { return decision_.temperature; }

    // logits 通道该请求的「输入 token 数」= 各问中最长 prompt 的 token 数(供 API 层 token
    // 计数 + 容量预检; 语义与 probs_logits 实际拼装一致)。
    int logits_input_tokens(const KevRequest & req) const;

    // 已解析的决策配置(marker + 温度 + logits 通道), 供调用方/日志/model_card 读取。
    const KevDecisionConfig & decision_config() const { return decision_; }

    // P2 前缀缓存(稳定公共前缀 LRU)。
    const PrefixLRU & prefix_cache() const { return prefix_lru_; }

    // Pluggable backend metadata (reported in /v1/models + startup log)
    const char * backend_name()   const { return backend_ ? backend_->name()   : "(none)"; }
    const char * backend_device() const { return backend_ ? backend_->device() : "(none)"; }
    const char * backend_version() const { return backend_ ? backend_->version() : "(none)"; }

private:
    // 生成式 logits 判别主通道(模型无关, 填空式 prompt + 候选答案 token logits + softmax)。
    // req 的 questions[].keys 提供每问候选答案 key; 见决策配置 enable_logits/prompt_template/aggregation。
    ProbsResult probs_logits(const KevRequest & req, std::string * err);

    // 解码 token 序列(续接在 start_pos 之后,写入 seq)。经 backend。
    bool decode_seq(const std::vector<int32_t> & tokens, int start_pos, int seq,
                    std::string * err, const std::vector<int32_t> & logits_pos);

    // n_batch 安全解码: 把 ids 切成 <= chunk 的若干段, 逐段 decode_seq 续写同一 seq,
    // 使单次 llama_decode 的 token 数不超过后端 cparams.n_batch(llama.cpp 会 GGML_ASSERT
    // n_tokens_all <= n_batch, 长 prompt + 多候选时会把总 token 顶爆而崩溃)。所有要读的
    // 输出位置(0-base 于 ids 内的 out_global)必须落在最后一段; 成功时把它们的 read_logits
    // 索引(g 减该段起点)写回 out_local。
    bool decode_seq_chunked(const std::vector<int32_t> & ids, int start_pos, int seq,
                            std::string * err, const std::vector<int32_t> & out_global,
                            std::vector<int32_t> & out_local, int chunk);

    // 填空式 prompt 渲染({state}/{instructions}/{options} 占位替换)。
    static std::string render_prompt(const std::string & tpl, const KevQuestion & q,
                                     const KevRequest & req, const std::vector<std::string> & opts);

    std::unique_ptr<Backend> backend_;
    KevDecisionConfig decision_;   // 已解析的决策配置(marker + 温度 + logits 通道)
    PrefixLRU    prefix_lru_{4, 1}; // P2: LCP 前缀 LRU(logits 通道; base/capacity 在 init 覆写)
    int n_seq_max_   = 0;
    int scratch_seq_ = 1;
    int prefix_seq_base_ = 1;      // P2: 前缀 slot 的起始 seq(在 scratch 之后)
    int prefix_capacity_ = 4;      // P2: 前缀槽位数
    int n_threads_   = 0;
    bool ok_         = false;
    // P0b: 串行化所有访问共享 llama ctx/scratch/前缀槽 的计算段。
    // P0 (concurrency): mutable, 以使 count_tokens()/logits_input_tokens() 等 const 方法
    // 也能把锁外四处运行的 kev::tokenize 对共享 llama_vocab 的访问纳入同一把互斥锁, 消除
    // 跨线程并发llama_tokenize 的资料竞争(崩溃/UB)。二者均在锁内不重入, 无死锁。
    mutable std::mutex compute_mutex_;
};

} // namespace kev
