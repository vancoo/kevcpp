// kev_kevhead.cpp — 无权重 KEV 打分实现
#include "kev_kevhead.h"

#include <cmath>

namespace kev {

void KevKevHead::probs(const float * h_decide, const float * const * h_opts, int n_opts,
                       int n_embd, float temperature, float * out_probs) {
    if (n_opts <= 0 || h_opts == nullptr || h_decide == nullptr || out_probs == nullptr) {
        return;
    }
    // KEV: n_pointer = n_embd / 2 (奇数截断)。与 llama.cpp server-context.cpp KEV 分支一致。
    const int m = n_embd / 2;
    if (m <= 0) {
        // 无法打分(n_embd < 2): 退化为均匀分布, 保证后续 softmax 不崩。
        const float uni = 1.0f / (float) n_opts;
        for (int j = 0; j < n_opts; ++j) out_probs[j] = uni;
        return;
    }

    // score_j = (Σ_{i<m} h_decide[i] * h_opts[j][m+i]) / sqrt(m)
    std::vector<float> s(n_opts);
    const float inv_sqrt_m = 1.0f / std::sqrt((float) m);
    for (int j = 0; j < n_opts; ++j) {
        float dot = 0.0f;
        const float * opt = h_opts[j];
        for (int i = 0; i < m; ++i) dot += h_decide[i] * opt[m + i];
        s[j] = dot * inv_sqrt_m;
    }

    // 数值稳定温度 softmax: exp((s_j - max) / T)
    float t = temperature;
    if (!(t > 0.0f) || !std::isfinite(t)) t = 1.0f;   // 防御非法温度
    const float invT = 1.0f / t;
    float mx = s[0];
    for (int j = 1; j < n_opts; ++j) if (s[j] > mx) mx = s[j];
    double sum = 0.0;
    for (int j = 0; j < n_opts; ++j) {
        float e = std::expf((s[j] - mx) * invT);
        out_probs[j] = e;
        sum += e;
    }
    if (sum > 0.0) {
        for (int j = 0; j < n_opts; ++j) out_probs[j] = (float) (out_probs[j] / sum);
    } else {
        const float uni = 1.0f / (float) n_opts;
        for (int j = 0; j < n_opts; ++j) out_probs[j] = uni;
    }
}

} // namespace kev