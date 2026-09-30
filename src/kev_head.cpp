// kev_head.cpp — PointerHead C++ 实现
#include "kev_head.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace kev {

std::vector<float> PointerHead::apply(const Linear & m, const float * x) const {
    std::vector<float> y(m.out);
    for (int o = 0; o < m.out; ++o) {
        float acc = m.b.empty() ? 0.0f : m.b[o];
        const float * row = m.W.data() + (size_t) o * m.in;
        for (int i = 0; i < m.in; ++i) {
            acc += row[i] * x[i];
        }
        y[o] = acc;
    }
    return y;
}

std::vector<float> PointerHead::logits(const float * h_decide,
                                       const std::vector<const float *> & h_opts) const {
    const int dp = q_.out;
    const auto qv = apply(q_, h_decide);           // [dp]
    std::vector<float> z(h_opts.size());
    for (size_t o = 0; o < h_opts.size(); ++o) {
        const auto kv = apply(k_, h_opts[o]);      // [dp]
        float acc = 0.0f;
        for (int j = 0; j < dp; ++j) acc += kv[j] * qv[j];
        z[o] = acc * scale_;
    }
    return z;
}

std::vector<float> PointerHead::probs(const float * h_decide,
                                      const std::vector<const float *> & h_opts) const {
    auto z = logits(h_decide, h_opts);
    if (z.empty()) return z;
    const float invT = 1.0f / temperature_;
    // softmax(z / T), numerically stable
    float m = -1e30f;
    for (float v : z) { v *= invT; if (v > m) m = v; }
    std::vector<float> e(z.size());
    double sum = 0.0;
    for (size_t i = 0; i < z.size(); ++i) {
        e[i] = std::expf((z[i] * invT) - m);
        sum += e[i];
    }
    for (float & v : e) v = (float) (v / sum);
    return e;
}

bool PointerHead::load_from_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "kev_head: cannot open '%s'\n", path.c_str());
        return false;
    }
    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "KEVH", 4) != 0) {
        std::fprintf(stderr, "kev_head: '%s' is not a KEVH head file (bad magic)\n", path.c_str());
        return false;
    }
    auto rd_i32 = [&f](std::int32_t & v) -> bool {
        f.read(reinterpret_cast<char *>(&v), 4); return f.good();
    };
    auto rd_f32 = [&f](float & v) -> bool {
        f.read(reinterpret_cast<char *>(&v), 4); return f.good();
    };
    auto rd_f32n = [&f](float * p, size_t n) -> bool {
        f.read(reinterpret_cast<char *>(p), (std::streamsize)(n * sizeof(float))); return f.good();
    };

    std::int32_t dim_in = 0, dim_proj = 0;
    if (!rd_i32(dim_in) || !rd_i32(dim_proj)) { std::fprintf(stderr, "kev_head: bad dims\n"); return false; }
    if (dim_in <= 0 || dim_proj <= 0) { std::fprintf(stderr, "kev_head: invalid dims %d x %d\n", dim_in, dim_proj); return false; }

    float temperature = 1.0f;
    if (!rd_f32(temperature)) { std::fprintf(stderr, "kev_head: bad temperature\n"); return false; }

    q_.in = dim_in; q_.out = dim_proj;
    k_.in = dim_in; k_.out = dim_proj;
    q_.W.resize((size_t) dim_proj * dim_in);
    q_.b.resize(dim_proj);
    k_.W.resize((size_t) dim_proj * dim_in);
    k_.b.resize(dim_proj);

    if (!rd_f32n(q_.W.data(), q_.W.size())) { std::fprintf(stderr, "kev_head: bad Wq\n"); return false; }
    if (!rd_f32n(q_.b.data(), q_.b.size())) { std::fprintf(stderr, "kev_head: bad bq\n"); return false; }
    if (!rd_f32n(k_.W.data(), k_.W.size())) { std::fprintf(stderr, "kev_head: bad Wk\n"); return false; }
    if (!rd_f32n(k_.b.data(), k_.b.size())) { std::fprintf(stderr, "kev_head: bad bk\n"); return false; }

    scale_       = 1.0f / std::sqrt((float) dim_proj);
    temperature_ = temperature;
    loaded_      = true;
    return true;
}

} // namespace kev