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
    // merge_lora=true 且 lora_path 非空时, init 会把 LoRA 合并进基础模型(产出 merged GGUF,
    // 由 merged_model_path 指定或自动放在 model_path 同目录), 再加载 merged 作为普通模型,
    // 不再挂独立 LoRA adapter。默认读环境变量 KEV_MERGE_LORA(=1 开启), 否则走分离 LoRA。
    bool        merge_lora = false;
    std::string merged_model_path;   // 可选: merged GGUF 的落盘位置
    int n_ctx          = 2048;
    int n_threads      = 0;      // 0 = auto
    int lru_capacity   = 2;      // 状态缓存 slot 数
    // 派生: n_seq_max = lru_capacity + 1(scratch)
};

class KevModel {
public:
    // P.4: 一次共享同 state 前缀、等长 branch 批量 decode 时最多并行 branch seq 数。
    // 分配在 scratch 之前, 直接计入 n_seq_max(全注意力 KV 层 n_stream=n_seq_max, 故不宜过大)。
    // 超过此数的等长问会拆成多个组各自批量, 或回退逐问(行为不变, 仍逐位一致)。
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
    // 复用外部已算好的 encode(省掉一次重复的 encode+rows_of): 仍会 rows_of(enc)。
    // 与单参 probs(req) 数值完全一致(原本就是 encode->rows_of->ensure_state->rows_from_slot)。
    ProbsResult probs(const KevEnc & enc, std::string * err = nullptr);

    // 预热: 把 state_ids 预填进状态缓存(未命中则 prefill)。启动预热专用, 默认不调用不影响行为。
    bool prewarm_state(const std::vector<int32_t> & state_ids);

    // 启动预热建图(P0a): 用 scratch seq 做最小 dummy decode, 仅触发 llama.cpp 的
    // 图构建 + ggml_backend_sched_reserve(compute buffer 保留), 不触碰任何 LRU slot /
    // 缓存 state / 业务输出, 数值逐位一致。它只把"进程一次性的建图+245MB buffer 分配"
    // 移到服务监听之前, 使冷启动首个请求不再付这笔(~290ms)。
    // warm_branch: 是否额外把 branch(readout, 带 logits 输出) 形状也预建(保守, 两形状常驻)。
    // state_tokens / branch_tokens: 代表性 token 数(默认 38 / 151), 用于预留 buffer 宽度;
    //   因 n_ubatch=256 封顶, 任意 <=256 的形状都会沿用同一已预留 buffer。
    bool warm_graph(bool warm_branch = true, int state_tokens = 38, int branch_tokens = 151,
                    std::string * err = nullptr);

    // 诊断
    const PrefixLRU & lru() const { return lru_; }
    int n_seq_max() const { return n_seq_max_; }
    int scratch_seq() const { return scratch_seq_; }
    int branch_seq(int k) const { return branch_seq_base_ + k; }  // 第 k 个可批 branch seq
    int max_batch() const { return kMaxBatch; }
    // 在 init 后设置(对拍用): batch_enabled=false 强制逐问 decode(验证批量逐位一致)。
    void set_batch_enabled(bool v) { batch_enabled_ = v; }
    bool batch_enabled() const { return batch_enabled_; }
    int threads() const { return n_threads_; }
    int threads_actual() const { return ctx_ ? llama_n_threads(ctx_) : 0; }
    int threads_batch_actual() const { return ctx_ ? llama_n_threads_batch(ctx_) : 0; }

    // 每流注意力 KV 容量(n_ctx_seq)。一个 sequence(branch) 需在单流内放下
    // 完整 state 前缀(Ls 格) + 自身 tokens(B 格) = input_tokens 格, 故
    // Ls+B > n_ctx_seq 时会 decode 失败(find_slot 无空闲 cells)。供 API 层
    // 提前判定 request 是否超容量, 给出清晰错误而非 cryptic "could not answer request"。
    int n_ctx_seq_capacity() const { return ctx_ ? (int) llama_n_ctx_seq(ctx_) : 0; }

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
    // 解码 token 序列(续接在 start_pos 之后,写入 seq)。
    // logits_pos: 需要输出 hidden(=embeddings) 的 token 位置集合(分支的 <decide>/<opt> 处)。
    //             传空 vector = 不输出任何 hidden(用于 state prefill,省输出缓冲与 GET_ROWS 计算)。
    // 语义保证: 每个位置的 output row 索引 == 该位置在 tokens 中的下标(利用 output_ids[i]=i)。
    bool decode_seq(const std::vector<int32_t> & tokens, int start_pos, llama_seq_id seq,
                    std::string * err, const std::vector<int32_t> & logits_pos);

    // P.4: 在单次 llama_decode 中解码多个 branch(共享 state 前缀, 等长 n_seq_tokens)。
    // branches: [Q] 每项 = {branch token 序列, start_pos(=Ls), seq_id, decide 与 opts 的 branch 内位置}。
    // seq-major 布局: 第 s 个 branch 的第 t 个 token 是 batch 下标 s*L + t。
    // hidden_decide[s] 在第 (s*L + branches[s].decide) 行。
    struct BranchDec {
        std::vector<int32_t> ids;        // branch token (长度 L, 全部相等)
        int start_pos;                   // = Ls (state 长度)
        llama_seq_id seq;                // 该 branch 写入的 seq(branch seq 池之一)
        int decide;                      // branch 内 decide 位置
        std::vector<int32_t> opts;       // branch 内 opt 位置
    };
    bool decode_batch(const std::vector<BranchDec> & branches, std::string * err);

    // 从上次 decode 的 output 取 hidden[1024](out_idx = batch 内 token 下标)
    std::vector<float> hidden_at_output(int out_idx, std::string * err);

    llama_context  * ctx_   = nullptr;
    llama_model    * model_ = nullptr;
    llama_adapter_lora * lora_ = nullptr;   // M5: 可选 LoRA 适配器
    const llama_vocab * vocab_ = nullptr;
    PointerHead  head_;
    PrefixLRU    lru_{2, 1};
    int n_seq_max_   = 0;
    int scratch_seq_ = 1;
    int branch_seq_base_ = 1;  // base(cap slots) 之后, scratch 之前: 可批 branch seq 池
    int n_threads_   = 0;
    // P.4: 是否启用等长 branch 批量 decode。init 时读环境变量 KEV_NO_BATCH(=1 关闭)。
    bool batch_enabled_ = true;
    bool ok_         = false;
    // 并发加固(P0b): 串行化所有访问共享 llama ctx/lru/scratch 的计算段。
    // 不改变单请求计算顺序或数值, 仅把并发竞态降为严格串行。
    std::mutex compute_mutex_;
};

} // namespace kev