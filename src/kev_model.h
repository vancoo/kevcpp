// kev_model.h — 组装层: llama 上下文 + 编码器 + 无权重 KEV 打分器 + 状态前缀缓存 (M3)
//
// 对应 kev 的 DecisionModel 接口子集(probs / encode / prefix 缓存), 纯 CPU, 混合架构 hybrid 走 row form:
//   - state 只 prefill 一次(缓存到 LRU slot 的 KV + Gated DeltaNet 反复式状态)。
//   - 每个 question 作为 continue row: llama_memory_seq_cp(slot -> scratch), 只解码 branch,
//     于 decide/</opt> 取 hidden, 过无权重 KEV 打分(KevKevHead::probs) 得概率。
//   - prefix_min_tokens=0(hybrid)。
//   - 同 state 的后续请求命中缓存, 不重跑 state prefill。
//
// Pluggable-backend (ARCHIVE): KevModel 保留全部组装/编排(LRU、encode 组装、rows 规划、
// KEV 打分、compute_mutex_、n_seq 布局 —— 均不扰动), 但 llama/ggml 算子耦合全部改经
// kev::Backend*(来自 kev_backend_factory/BACKEND C-ABI) 调用, 不再直接调 llama_*。见
// docs/portable_backend_deployment.md。encode/tokenize 也经后端: 由 backend 的
// token_feeder()(delegated tokenize/special_token_id, llama-free) 提供(single-llama: 见文档)。
#pragma once

#include "kev_encode.h"
#include "kev_head.h"
#include "kev_kevhead.h"
#include "kev_prefix.h"
#include "kev_backend_factory.h"
#include "kev_calibrate.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace kev {

struct KevModelOptions {
    std::string model_path;
    std::string head_path;      // 兼容旧客户代码: 保留字段但路线 A 不再强制加载/使用。
    std::string lora_path;      // 可选 llama.cpp GGUF lora 适配器 (M5)
    float       lora_scale = 1.0f;
    // merged-LoRA: 把 lora_path fold 进 model_path 的 Q8_0 权重, 前向走单份权重。
    bool        merge_lora = false;
    std::string merged_model_path;   // 可选: merged GGUF 的落盘位置
    // 决策配置(模型无关 marker + temperature)。原版 Qwen3.5 用默认; 换模型改这里或
    // decision_config 文件。temperature / option_end 等旧 CLI 覆盖项也回填到这里。
    KevDecisionConfig decision;
    std::string decision_config;   // 可选 JSON 决策配置文件路径 (加载后覆盖 decision; 见 config/)
    // 便捷覆盖项(CLI): --temperature / --opt-end / --decide-token。非空/+0 则覆盖 decision。
    // 优先级: decision_config 文件 < 如下 CLI 覆盖项 (CLI 显式覆盖始终胜出)。
    float       temperature_override  = 0.0f;  // >0 时覆盖 decision.temperature
    std::string option_end_override;          // 非空时覆盖 decision.option_end
    std::string decide_marker_override;       // 非空时覆盖 decision.decide_marker
    int n_ctx          = 2048;
    int n_threads      = 0;      // 0 = auto
    int lru_capacity   = 2;      // 状态缓存 slot 数
    // P2 前缀缓存槽位数(缓存"稳定公共前缀"{instructions}\n[ {options} ]\n 的 KV)。同一类任务的
    // 多请求共享同一前缀 -> 只需 1~2 个槽即命中; 不同 instructions/options 任务各占一个 LRU 槽。
    int prefix_capacity = 4;
    std::string backend = "cpu"; // pluggable backend name (default "cpu")
    // 派生: n_seq_max = lru_capacity + 1(scratch) + kMaxBatch(branch 池) + prefix_capacity(前缀池)。
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

    // 方案3: 取「生成式 logits 判别通道」每个候选 key 的原始 logits(softmax 缩放前), [Q][K]。
    // 供校准/评测 harness 离线统计使用(无需 re-decode 即可网格搜温度/去偏置)。模型无关。
    // err 非空且失败时在其中说明原因; 返回 false。
    bool raw_logits(const KevRequest & req, std::vector<std::vector<float>> & out,
                    std::string * err = nullptr);

    // 生成式 logits 判别主通道是否生效(enable_logits 且后端支持 read_logits)。
    bool logits_channel_active() const;

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

    int   dim_in()  const { return kevhead_.dim(); }
    float temperature() const { return kevhead_.temperature(); }

    // 已解析的决策配置(marker + 温度), 供调用方/日志/model_card 读取。
    const KevDecisionConfig & decision_config() const { return decision_; }

    // 方案3: 当前生效的 logits 决策映射校准器(默认恒等)。供决策层(API)读取 drop_low_conf /
    // 归一化置信, 应用低置信拒绝。
    const LogitsCalibrator & calibrator() const { return calibrator_; }
    // 按 decision_.calib 重建校准器(含加载 params_file)。供 init 与校准 harness 在拟合后调。
    void rebuild_calibrator(std::string * err = nullptr);

    // Pluggable backend metadata (reported in /v1/models + startup log)
    const char * backend_name()   const { return backend_ ? backend_->name()   : "(none)"; }
    const char * backend_device() const { return backend_ ? backend_->device() : "(none)"; }
    const char * backend_version() const { return backend_ ? backend_->version() : "(none)"; }

private:
    int ensure_state(const std::vector<int32_t> & state_ids, double * ps_state, bool * hit);
    std::vector<std::vector<float>> rows_from_slot(const KevRows & rows, int slot, double * t_rows,
                                                   std::vector<std::vector<float>> * out_hidden = nullptr,
                                                   std::vector<double> * rows_ms = nullptr);

    // 生成式 logits 判别主通道(模型无关, 填空式 prompt + 候选答案 token logits + softmax)。
    // req 的 questions[].keys 提供每问候选答案 key; 见决策配置 enable_logits/prompt_template/aggregation。
    // 可选 *raw_out: 非空时把每问"候选 key 的原始 logits"([Q][K], 校准/缩放前)写入。
    ProbsResult probs_logits(const KevRequest & req, std::string * err,
                             std::vector<std::vector<float>> * raw_out = nullptr);

    // 解码 token 序列(续接在 start_pos 之后,写入 seq)。经 backend。
    bool decode_seq(const std::vector<int32_t> & tokens, int start_pos, int seq,
                    std::string * err, const std::vector<int32_t> & logits_pos);
    // P.4: 单次 decode 多个等长 branch(共享 state 前缀)。经 backend。
    bool decode_batch(const std::vector<BackendBranch> & branches, std::string * err);

    // n_batch 安全解码: 把 ids 切成 <= chunk 的若干段, 逐段 decode_seq 续写同一 seq,
    // 使单次 llama_decode 的 token 数不超过后端 cparams.n_batch(llama.cpp 会 GGML_ASSERT
    // n_tokens_all <= n_batch, 长 prompt + 多候选时会把总 token 顶爆而崩溃)。所有要读的
    // 输出位置(0-base 于 ids 内的 out_global)必须落在最后一段; 成功时把它们的 read_logits/
    // read_hidden 索引(g 减该段起点)写回 out_local。
    bool decode_seq_chunked(const std::vector<int32_t> & ids, int start_pos, int seq,
                            std::string * err, const std::vector<int32_t> & out_global,
                            std::vector<int32_t> & out_local, int chunk);

    // 从上次 decode 的 output 取 hidden[1024](经 backend read_hidden)
    std::vector<float> hidden_at_output(int out_idx, std::string * err);

    // P2 前缀缓存(logits 通道): 稳定公共前缀 slot 查找/分配(返回缓存该前缀 token ids 的 seq)。
    // 命中返回已缓存的前缀 seq; 未命中返回 -1 并经 *evict_seq 给出"需先 seq_clear 再 prefill"的
    // 旧槽(调用方 decode P[0..Dp) 到该槽后调 register_)。不可缓存(无后端/前缀空)返回 -1。
    int prefix_slot(const std::vector<int32_t> & prefix_ids, int * evict_seq);

    std::unique_ptr<Backend> backend_;
    // 路线 A: 无权重 KEV 打分器(替代旧 PointerHead)。kevhead_.dim/temperature 由 init 设入。
    KevKevHead   kevhead_;
    KevDecisionConfig decision_;   // 已解析的决策配置(marker + 温度)
    PrefixLRU    lru_{2, 1};       // state-prefix LRU(KEV 回退通道)
    PrefixLRU    prefix_lru_{4, 1}; // P2: 稳定公共前缀 LRU(logits 通道; base/capacity 在 init 覆写)
    int n_seq_max_   = 0;
    int scratch_seq_ = 1;
    int branch_seq_base_ = 1;
    int prefix_seq_base_ = 1;      // P2: 前缀 slot 的起始 seq(高于 scratch + branch 池)
    int prefix_capacity_ = 4;      // P2: 前缀槽位数
    int n_threads_   = 0;
    bool batch_enabled_ = true;
    bool ok_         = false;
    // 方案3: logits 决策映射层校准器(默认恒等; 由 decision_.calib 构建)。
    LogitsCalibrator calibrator_;
    // P0b: 串行化所有访问共享 llama ctx/lru/scratch 的计算段。
    std::mutex compute_mutex_;
};

} // namespace kev
