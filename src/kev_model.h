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
#include "kev_batch.h"

#include <atomic>
#include <cstdint>
#include <map>
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
    // 解码批大小 (llama n_batch / n_ubatch, 默认 512 / 256)。n_batch 是单次 llama_decode
    // 的「逻辑上限」token 数(超了会触发 llama GGML_ASSERT); llama.cpp 内部把一次 n_batch 的
    // 前向按 n_ubatch 切子批、每个子批重建并运行一次计算图(真正的 prefill 块)。n_ubatch 必须
    // <= n_batch。0 = 用后端各默认(512/256)。两者一起加大对长 prompt / 高并发合并有明显影响。
    int n_batch        = 512;
    int n_ubatch       = 256;
    // P2 前缀缓存槽位数(缓存"稳定公共前缀"{instructions}\n[ {options} ]\n 的 KV)。同一类任务的
    // 多请求共享同一前缀 -> 只需 1~2 个槽即命中; 不同 instructions/options 任务各占一个 LRU 槽。
    int prefix_capacity = 4;
    std::string backend = "cpu"; // pluggable backend name (default "cpu")
    // B2: 并发批处理。enable_batching=true 且后端支持 decode_multi_seq 时, 解码经
    // BatchScheduler 单 worker 集中: 并发到达的请求各自用独立 sequence 槽, worker 把它们的
    // 单条 decode 合并进同一次 decode_multi_seq(摊销 per-decode 固定开销)。false 时退回
    // 原「compute_mutex_ 全程串行」路径(可回退/基准)。
    bool enable_batching = true;
    // B2: 并发 scratch sequence 槽位数(同时 in-flight 的请求数上限)。在后缀槽之后分配。
    int max_concurrency = 8;
    // B2: batch 合并窗口(微秒)。B3(continuous): 仅当队列稀疏(未满一个 batch)时才等这么短的一
    // 段时间以攒到更多并发请求; 队列满时立即 decode(不加延迟)。默认 400us。
    long batch_aggregation_us = 400;
    // 派生: n_seq_max = scratch(1) + prefix_capacity(前缀池) + max_concurrency(并发槽)。
    // MoE 专家 offload(透传给后端; 见 kev_backend.h)。
    int n_gpu_layers = 0;   // 0 = 全 CPU; 负 = 全 GPU; 正 = N 层 GPU
    int moe_offload  = 0;   // 1 = MoE 专家张量强制留 host RAM, 其余按 n_gpu_layers
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

    // ---- 调研实验: hidden 聚合判别通道(零训练, KEV_DECIDE=hidden|hidden_state) ----
    // 读隐层读入表示向量, 在 hidden 空间聚合/打分, 替代"仅取候选首 token 的 vocab logits"。
    //   use_state_agg=true  → 先用 h_decide 对 {state} 各 token hidden 做软注意力加权聚合,
    //                         得 h_agg(即"用注意力找相关内容再 sum")。
    //   score[c] = cos(query, h_cand[c]), query = h_decide 或 h_agg。零训练, 模型无关。
    ProbsResult probs_hidden(const KevRequest & req, bool use_state_agg, std::string * err);

    // ---- 调研实验: kev 风格 system-one 读取(零训练, KEV_DECIDE=decide) ----
    // 对标 kev PointerHead 的 _readout: 同一次 forward 读 "决定位 <decide> 的 hidden" 与每个
    // 选项闭合位 <o_e> 的 hidden, 用零训练去质心余弦替代训过的 Wq/Wk 做 cross-match。
    // score[j] = cos(h_decide - centroid, h_opts[j] - centroid)。模型无关, 不生成 token。
    ProbsResult probs_decide(const KevRequest & req, std::string * err);

    // ---- 调研实验: 标签 token 打分 + 顺序轮换 (零训练, KEV_LBL=abcd|yn + KEV_ROTATE=n) ----
    // 落实用户两条思路: (1) 构造约束 prompt 让答案 token 能被直接输出 —— 选项统一成可直接
    // 输出的标签 token(A/B/C… | 否/是 | 1..L), 打分只读这些标签 token 的 logit、只对标签集
    // softmax(空格/thinking/解释 token 不计入, 等于在打分层完成约束);(2) 顺序轮换抵消
    // 字母/位置 bias(2308.11483 顺序可差 13–75%): 对 choice 把 标签→选项 的映射轮换 r 次,
    // 各打分一次, 对"原选项 key"做概率平均(或多数), 抵消系统性标签位置偏置。
    // 零训练、模型无关。默认 off(走 probs_logits)。
    ProbsResult probs_label(const KevRequest & req, const std::string & mode, int rotate,
                            std::string * err);

    // Decode `tokens` 续接在 seq, 在 out_pos 各位置读 hidden 向量(read_hidden), 返回
    // hidden[out][n_embd]。串行路径(调用方须已持 compute_mutex_)。copy_src>=0 表示从缓存
    // 前缀 continue。out_pos 使用与 decode_one_to_logits 相同的全局位置语义。
    bool decode_one_to_hidden(const std::vector<int32_t> & tokens, int32_t start_pos,
                              int32_t seq, int32_t copy_src,
                              const std::vector<int32_t> & out_pos,
                              std::vector<std::vector<float>> & hidden,
                              std::string * err);

    // ---- B2: batched-decode helpers ----
    // Decode `tokens` (possibly continued from cached prefix `copy_src`) into `seq`,
    // read candidate logits at `out_pos`, and return them in `logits[out][cand]`.
    // Works in both modes: batched path submits to schedule_mu_ worker; serial path
    // (compute_mutex_ already held) calls decode_seq directly.
    bool decode_one_to_logits(const std::vector<int32_t> & tokens, int32_t start_pos,
                              int32_t seq, int32_t copy_src,
                              const std::vector<int32_t> & out_pos,
                              const std::vector<std::vector<int32_t>> & out_cands,
                              std::vector<std::vector<float>> & logits,
                              std::string * err);

    // acquire/release a per-request scratch sequence slot (concurrent pool).
    int  acquire_scratch_slot();
    void release_scratch_slot(int seq);

    // ---- 调研实验: 上下文校准 (contextual calibration; KEV_CALIB=1, 零训练) ----
    // Zhao et al., "Calibrate Before Use", ICML 2021 (arXiv:2102.09690)。
    // 治"模型对某答案有恒定先验偏置"(如 noul always-yes): 用内容中性输入估计每个候选的
    // 系统性偏置 logit, 真实请求在 softmax 前做 z'[c] = z[c] - z_bias[c]。
    // 缓存按 (候选 key 列表 + aggregation) 键控, 惰性估算一次。归 compute_mutex_ 串行保护
    // (校准开启时 probs_logits 强制走串行路径)。不引入任何训练产物。
    std::vector<float> estimate_answer_bias(const std::vector<std::string> & keys);
    void apply_calibration(const KevQuestion & q, std::vector<float> & z);
    bool calibrate_ = false;                       // KEV_CALIB=1 时开启
    std::map<std::string, std::vector<float>> calib_cache_;  // key -> bias (compute_mutex_ 保护)

    std::unique_ptr<Backend> backend_;
    KevDecisionConfig decision_;   // 已解析的决策配置(marker + 温度 + logits 通道)
    PrefixLRU    prefix_lru_{4, 1}; // P2: LCP 前缀 LRU(logits 通道; base/capacity 在 init 覆写)
    int n_seq_max_   = 0;
    int scratch_seq_ = 1;
    int prefix_seq_base_ = 1;      // P2: 前缀 slot 的起始 seq(在 scratch 之后)
    int prefix_capacity_ = 4;      // P2: 前缀槽位数
    int concurrent_seq_base_ = 0;  // B2: 并发 scratch 槽起始 seq(在前缀槽之后)
    int max_concurrency_   = 0;    // B2: 并发槽数
    bool batching_enabled_ = false; // B2: 实际启用了 batch 调度器
    std::unique_ptr<BatchScheduler> scheduler_;  // B2
    std::vector<bool> slot_used_;   // B2: 并发槽占用表
    std::mutex slot_mutex_;         // B2: 槽位分配
    std::mutex tokenizer_mutex_;    // B2: llama_vocab 分词串行(替代原来整函数持 compute_mutex_)
    std::mutex prefix_mutex_;       // B2: LRU prepare/register + 前缀 prefill 串行
    int n_threads_   = 0;
    bool ok_         = false;
    // P0b: 串行化所有访问共享 llama ctx/scratch/前缀槽 的计算段。
    // P0 (concurrency): mutable, 以使 count_tokens()/logits_input_tokens() 等 const 方法
    // 也能把锁外四处运行的 kev::tokenize 对共享 llama_vocab 的访问纳入同一把互斥锁, 消除
    // 跨线程并发llama_tokenize 的资料竞争(崩溃/UB)。二者均在锁内不重入, 无死锁。
    mutable std::mutex compute_mutex_;
};

} // namespace kev
