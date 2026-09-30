// kev_head.h — PointerHead (Kev 读出头) 的 C++ 实现
//
// 移植自 E:\van\projects\kev\kev\model.py 的 PointerHead(class, ~L182):
//   self.q, self.k = nn.Linear(d, dp)          // 隐藏 d=1024 -> 指针维 dp=256
//   self.scale     = 1 / sqrt(dp)              // = 1/16 for dp=256
//   forward(h_decide, h_opts):                  // h_decide[d], h_opts[K,d] -> logits[K]
//       z = (self.k(h_opts) @ self.q(h_decide)) * self.scale
//       z = z / self.temperature if (eval and temperature != 1.0) else z
// 概率 = softmax(z / T)。
//
// nn.Linear: y = x @ W^T + b, W[dp,d], b[dp], 因此
//   k(h_opt) = Wk @ h_opt + bk   ([dp])
//   q(h_dec) = Wq @ h_dec + bq   ([dp])
//   z[i] = (Wk @ h_opt_i + bk) · (Wq @ h_dec + bq) * scale
//
// 权重从自定 .bin 加载（见 load_from_file），格式由 tests/gen_head_weights.py 生成，
// 其来源是 torch 的 head.pt（q.weight/q.bias/k.weight/k.bias + temperature）。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kev {

// 一个线性层: W[out][in] (row-major), b[out]; y = W x + b
struct Linear {
    std::vector<float> W;   // out * in
    std::vector<float> b;   // out
    std::int32_t in  = 0;
    std::int32_t out = 0;
};

class PointerHead {
public:
    PointerHead() = default;

    // 从 .bin 加载 q/k/temperature。格式:
    //   char magic[4] = "KEVH"
    //   int32 dim_in                    (= d, 如 1024)
    //   int32 dim_proj                  (= dp, 如 256)
    //   float32 temperature
    //   float32 Wq[dp][dim_in] (row-major)  (nn.Linear.weight, nn.Linear 的 W 即 weight)
    //   float32 bq[dp]
    //   float32 Wk[dp][dim_in]
    //   float32 bk[dp]
    // 均为小端。返回 true 成功。
    bool load_from_file(const std::string & path);

    // 计算 logits: z[K]= ( k(h_opts) · q(h_decide) ) * scale  (未除温度, 与 model.py forward 一致)
    std::vector<float> logits(const float * h_decide,
                              const std::vector<const float *> & h_opts) const;

    // 概率: softmax(z / T)
    std::vector<float> probs(const float * h_decide,
                             const std::vector<const float *> & h_opts) const;

    int dim_in()    const { return q_.in; }   // d  (隐藏维)
    int dim_proj()  const { return q_.out; }  // dp (指针维)
    float temperature() const { return temperature_; }

    const Linear & q() const { return q_; }
    const Linear & k() const { return k_; }

private:
    std::vector<float> apply(const Linear & m, const float * x) const;
    Linear q_;
    Linear k_;
    float  scale_ = 0.0f;
    float  temperature_ = 1.0f;
    bool   loaded_ = false;
};

} // namespace kev