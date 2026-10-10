// kev_kevhead.h — 无权重 KEV 打分器 (路线 A: 零 LoRA / 零独立 PointerHead)
//
// 实现 KEV(joint head / decision-model) 半边点积打分。公式来源为 llama.cpp 最新版
// server (E:\van\projects\llama.cpp\tools\server):
//   - server-context.cpp:2381-2401  (KEV 分支)
//         n_pointer = model_n_embd_out / 2
//         embd_q    = question 的 <decide> 隐层 (pointer)
//         对每个 option 的 marker(如 <|box_end|>) 隐层 embd:
//             dot = Σ_{i<n_pointer} embd_q[i] * embd[n_pointer + i]
//             score = dot / sqrt(n_pointer)
//   - server-decision.cpp:1078-1088 (temperature softmax)
//         数值稳定 softmax((score - max) / temperature), 默认 temperature=1.0.
//
// 与旧 PointerHead(kev_head) 的区别: 完全无权重、无 .bin 加载. dim 直接由外部设入
// (通常 = model_n_embd). 这样可直接对"任意原版开源 GGUF"做 system-one 打分, 与模型无关。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kev {

class KevKevHead {
public:
    KevKevHead() = default;

    // 无任何加载文件。dim 由外部设入 (model_n_embd); temperature 可配置 (默认 1.0)。
    void set_dim(int d)            { dim_ = d; }
    void set_temperature(float t)  { temperature_ = t; }
    int   dim() const              { return dim_; }
    float temperature() const      { return temperature_; }

    // 无权重 KEV 打分 + 温度 softmax。见类头注释的公式来源。
    //   h_decide  : <decide> 隐层 [n_embd] (question 的 last-token hidden)
    //   h_opts    : n_opts 个指针数组, 每个 [n_embd]; h_opts[j] 是第 j 个 option 的
    //               marker (默认 <|box_end|>) 位置的隐层。
    //   n_embd    : 模型隐层维 (model_n_embd); m = n_embd/2 (奇数截断, 与 llama.cpp 一致)。
    //   temperature: softmax 温度 (传入 1.0 则纯 softmax)。
    //   out_probs : 长度 >= n_opts, 写入规范化概率。
    // 空 h_opts(或 n_opts==0) 时 out_probs 不写任何东西(视为恒空)。
    static void probs(const float * h_decide, const float * const * h_opts, int n_opts,
                      int n_embd, float temperature, float * out_probs);

private:
    int   dim_ = 0;
    float temperature_ = 1.0f;
};

} // namespace kev