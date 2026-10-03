// kev_merge.cpp — merged-LoRA 折叠实现(自包含 GGUF Q8_0 patch)。
#include "kev_merge.h"

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace kev {

namespace {

constexpr uint32_t GGUF_MAGIC = 0x46554747u;        // "GGUF"
constexpr uint64_t GGUF_ALIGNMENT = 32;
constexpr uint32_t GGML_TYPE_F32  = 0;
constexpr uint32_t GGML_TYPE_Q8_0 = 8;
constexpr uint32_t GGML_TYPE_F16  = 1;

constexpr int    QK8_0 = 32;
constexpr size_t BLK_Q8_0_SIZE = 34;   // [fp16 d][int8 x32]
constexpr int32_t KV_TYPE_STRING = 8;

// ---- fp16 <-> fp32(位操作, 等价位如 ggml) ----
float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t man  = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) { f = sign; }
        else {
            exp = 126 - 15 + 1;
            while ((man & 0x400) == 0) { man <<= 1; --exp; }
            man &= 0x3ff;
            f = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1f) {
        f = 0x7f800000u | sign | (man << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float out; std::memcpy(&out, &f, 4); return out;
}

uint16_t fp32_to_f16(float x) {
    uint32_t f; std::memcpy(&f, &x, 4);
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((f >> 23) & 0xff) - 127 + 15;
    uint32_t man  = f & 0x7fffff;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t) sign;
        man |= 0x800000;
        int shift = 14 - exp;
        uint32_t r = man >> shift;
        uint32_t rem = man & ((1u << shift) - 1);
        uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (r & 1))) ++r;
        return (uint16_t)(sign | r);
    }
    uint32_t r = 0x1000 | (man >> 13);
    uint32_t rem = man & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) ++r;
    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((r - 0x1000) & 0x3ff));
}

// ---- 最小 GGUF ----
struct Reader {
    std::ifstream f;
    bool read_exact(void * dst, size_t n) {
        return (bool)f.read((char*)dst, (std::streamsize)n) && (size_t)f.gcount() == n;
    }
    template <typename T> bool read(T & v) { return read_exact(&v, sizeof(T)); }
    bool read_str(std::string & out) {
        uint64_t n; if (!read(n)) return false;
        if (n > (1u << 30)) return false;
        out.resize((size_t)n);
        if (n == 0) return true;
        return read_exact(&out[0], (size_t)n);
    }
    bool seek(uint64_t pos) { f.seekg((std::streamoff)pos); return (bool)f; }
};

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> ne;
    uint32_t type = 0;
    uint64_t offset = 0;     // 相对 data section
    uint64_t nbytes = 0;
};

// 跳过一串 KV(值类型已知长度; 数组/字符串逐项)。
bool skip_kv(Reader & r, std::string * err) {
    auto skip_str = [&]() -> bool { std::string s; return r.read_str(s); };
    uint32_t vt; if (!r.read(vt)) { if (err) *err="kv type"; return false; }
    switch (vt) {
        case 0: { uint8_t v; return r.read(v); }
        case 1: { int8_t v; return r.read(v); }
        case 2: { uint16_t v; return r.read(v); }
        case 3: { int16_t v; return r.read(v); }
        case 4: { uint32_t v; return r.read(v); }
        case 5: { int32_t v; return r.read(v); }
        case 6: { float v; return r.read(v); }
        case 7: { bool v; return r.read(v); }
        case 8: return skip_str();
        case 10: { uint64_t v; return r.read(v); }
        case 11: { int64_t v; return r.read(v); }
        case 12: { double v; return r.read(v); }
        case 9: { // array
            uint32_t at; if (!r.read(at)) return false;
            uint64_t n;   if (!r.read(n))   return false;
            size_t elem = 0;
            switch (at) {
                case 0: case 1: case 7: elem=1; break;
                case 2: case 3: elem=2; break;
                case 4: case 5: case 6: elem=4; break;
                case 10: case 11: case 12: elem=8; break;
                case 8: {
                    for (uint64_t k = 0; k < n; ++k) if (!skip_str()) return false;
                    return true;
                }
                default: if (err) *err="array elem type"; return false;
            }
            for (uint64_t k = 0; k < n; ++k) { uint8_t junk[8]; if (!r.read_exact(junk, elem)) return false; }
            return true;
        }
        default: if (err) *err="kv type "+std::to_string(vt); return false;
    }
}

bool parse_gguf(const std::string & path, uint64_t * data_offset,
                std::map<std::string, TensorInfo> * tensors, std::string * err) {
    Reader r; r.f.open(path, std::ios::binary);
    if (!r.f) { if (err) *err = "cannot open " + path; return false; }
    uint32_t magic, ver; uint64_t n_tensors, n_kv;
    if (!r.read(magic) || magic != GGUF_MAGIC) { if (err) *err="not a gguf: "+path; return false; }
    if (!r.read(ver)) { if (err) *err="bad gguf"; return false; }
    if (!r.read(n_tensors) || !r.read(n_kv)) { if (err) *err="bad gguf counts"; return false; }
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key; if (!r.read_str(key)) { if (err) *err="kv key"; return false; }
        std::string ke;
        if (!skip_kv(r, &ke)) { if (err) *err="kv value: "+ke; return false; }
    }
    tensors->clear();
    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorInfo ti;
        if (!r.read_str(ti.name)) { if (err) *err="tensor name"; return false; }
        uint32_t nd; if (!r.read(nd)) { if (err) *err="tensor nd"; return false; }
        if (nd > 4) { if (err) *err="tensor nd>4"; return false; }
        ti.ne.resize(nd);
        if (nd > 0 && !r.read_exact(ti.ne.data(), nd*sizeof(uint64_t))) { if (err) *err="tensor ne"; return false; }
        if (!r.read(ti.type)) { if (err) *err="tensor type"; return false; }
        if (!r.read(ti.offset)) { if (err) *err="tensor off"; return false; }
        int qk = 1; size_t es = 4;
        switch (ti.type) {
            case GGML_TYPE_F32: qk=1; es=4; break;
            case GGML_TYPE_F16: qk=1; es=2; break;
            case GGML_TYPE_Q8_0: qk=QK8_0; es=BLK_Q8_0_SIZE; break;
            default: if (err) *err="unsupported tensor type " + std::to_string(ti.type) + " for " + ti.name; return false;
        }
        uint64_t ne0 = ti.ne.empty() ? 1 : ti.ne[0];
        uint64_t nblk = ne0 / qk + (ne0 % qk ? 1 : 0);
        uint64_t row = (uint64_t)es * nblk;
        uint64_t nrows = 1;
        for (size_t d = 1; d < ti.ne.size(); ++d) nrows *= ti.ne[d];
        ti.nbytes = row * nrows;
        (*tensors)[ti.name] = ti;
    }
    uint64_t cur = (uint64_t)r.f.tellg();
    *data_offset = (cur + GGUF_ALIGNMENT - 1) & ~(GGUF_ALIGNMENT - 1);
    return true;
}

// 从 LoRA gguf 读取 adapter.lora.alpha(f32)。
float read_lora_alpha(const std::string & path) {
    Reader r; r.f.open(path, std::ios::binary);
    if (!r.f) return 0.0f;
    uint32_t magic, ver; uint64_t n_t, n_kv;
    if (!r.read(magic) || magic != GGUF_MAGIC) return 0.0f;
    if (!r.read(ver) || !r.read(n_t) || !r.read(n_kv)) return 0.0f;
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key; if (!r.read_str(key)) return 0.0f;
        if (key == "adapter.lora.alpha") {
            uint32_t vt; if (!r.read(vt)) return 0.0f;
            if (vt != 6) return 0.0f;
            float v; if (!r.read(v)) return 0.0f;
            return v;
        }
        std::string ke; if (!skip_kv(r, &ke)) return 0.0f;
    }
    return 0.0f;
}

void dequant_q8_block(const uint8_t * blk, float * out) {
    uint16_t dh; std::memcpy(&dh, blk, 2);
    const float d = fp16_to_f32(dh);
    const int8_t * q = (const int8_t*)(blk + 2);
    for (int j = 0; j < QK8_0; ++j) out[j] = q[j] * d;
}

// Q8_0 整块 quant。先用 amax/127 初始化, 再做数次最小二乘迭代(同时也是 ggml 前向
// dequant q*d 的最小误差拟合), 以低于 ggml 简单 quantize_row_q8_0_ref 的舍入误差折叠 LoRA。
inline void quant_q8_block(const float * in, uint8_t * blk) {
    float amax = 0.0f;
    for (int j = 0; j < QK8_0; ++j) amax = std::max(amax, std::fabs(in[j]));
    float d = (amax > 0.0f) ? amax / ((1 << 7) - 1) : 0.0f;

    // 迭代: 给定 d -> q=round(x/d); 给定 q -> d=sum(q*x)/sum(q^2) (用 fp16 舍入后的 d 参与)。
    for (int it = 0; it < 4; ++it) {
        float sum_qx = 0.0f, sum_qq = 0.0f;
        float id = (d > 0.0f) ? 1.0f / d : 0.0f;
        for (int j = 0; j < QK8_0; ++j) {
            float qv = std::floor(in[j] * id + 0.5f);
            if (qv > 127.0f) qv = 127.0f;
            if (qv < -127.0f) qv = -127.0f;
            if (qv != 0.0f) { sum_qx += qv * in[j]; sum_qq += qv * qv; }
        }
        if (sum_qq <= 0.0f) break;
        float d_new = sum_qx / sum_qq;
        d = (float)fp16_to_f32(fp32_to_f16(d_new));   // 让优化目标用前向实际用的 fp16 尺度
        if (it == 3) break;
    }

    const uint16_t dh = fp32_to_f16(d);
    std::memcpy(blk, &dh, 2);
    int8_t * q = (int8_t*)(blk + 2);
    const float d_eff = fp16_to_f32(dh);
    const float id = (d_eff > 0.0f) ? 1.0f / d_eff : 0.0f;
    for (int j = 0; j < QK8_0; ++j) {
        float rr = std::floor(in[j] * id + 0.5f);
        if (rr > 127.0f) rr = 127.0f;
        if (rr < -127.0f) rr = -127.0f;
        q[j] = (int8_t)rr;
    }
}

} // namespace

bool merge_lora_into_model(const std::string & base_path,
                           const std::string & lora_path,
                           const std::string & out_path,
                           float adapter_scale,
                           std::string * err) {
    uint64_t base_data_off = 0, lora_data_off = 0;
    std::map<std::string, TensorInfo> base, lora;
    if (!parse_gguf(base_path, &base_data_off, &base, err)) return false;
    if (!parse_gguf(lora_path, &lora_data_off, &lora, err)) return false;
    if (base_data_off == 0 || lora_data_off == 0) { if (err) *err="internal error: data offset 0"; return false; }
    if (base.empty() || lora.empty()) { if (err) *err="empty model/adapter"; return false; }

    // 收集 lora_a/lora_b 对
    struct AB { const TensorInfo * a = nullptr; const TensorInfo * b = nullptr; };
    std::map<std::string, AB> ab_map;
    for (const auto & kv : lora) {
        const std::string & nm = kv.first;
        std::string base_nm = nm;
        bool is_a=false, is_b=false;
        static const char * SA = ".lora_a", * SB = ".lora_b";
        if (nm.size() > 7 && nm.compare(nm.size()-7, 7, SA) == 0) { is_a=true; base_nm.resize(nm.size()-7); }
        else if (nm.size() > 7 && nm.compare(nm.size()-7, 7, SB) == 0) { is_b=true; base_nm.resize(nm.size()-7); }
        else continue;
        auto & ab = ab_map[base_nm];
        if (is_a) ab.a = &kv.second; else ab.b = &kv.second;
    }
    if (ab_map.empty()) { if (err) *err="no lora_a/lora_b pairs in "+lora_path; return false; }

    const float alpha = read_lora_alpha(lora_path);

    // 复制整个 base 文件 -> out(非适配字节原样保留)。
    {
        std::ifstream src(base_path, std::ios::binary);
        std::ofstream dst(out_path, std::ios::binary);
        if (!src || !dst) { if (err) *err="cannot create "+out_path; return false; }
        dst << src.rdbuf();
        dst.flush();
        if (!dst) { if (err) *err="write failed "+out_path; return false; }
    }
    // 再打开可写 for patch
    std::fstream out(out_path, std::ios::binary | std::ios::in | std::ios::out);
    if (!out) { if (err) *err="cannot reopen "+out_path; return false; }

    uint64_t filesize = 0;
    { std::ifstream src(base_path, std::ios::binary | std::ios::ate); filesize = (uint64_t)src.tellg(); }

    size_t patched = 0;
    for (auto & kv : ab_map) {
        const std::string & base_nm = kv.first;
        const AB & ab = kv.second;
        if (!ab.a || !ab.b) { if (err) *err="lora pair incomplete for "+base_nm; return false; }
        const TensorInfo & A = *ab.a;
        const TensorInfo & B = *ab.b;
        auto wIt = base.find(base_nm);
        if (wIt == base.end()) { if (err) *err="lora target '"+base_nm+"' not found in base model"; return false; }
        const TensorInfo & W = wIt->second;
        if (W.type != GGML_TYPE_Q8_0) { if (err) *err="base tensor '"+base_nm+"' not Q8_0 (type="+std::to_string(W.type)+"); unsupported"; return false; }
        if (A.type != GGML_TYPE_F32 || B.type != GGML_TYPE_F32) { if (err) *err="lora A/B not F32 (only F32 adapters supported)"; return false; }

        const uint64_t n_in  = W.ne.empty() ? 1 : W.ne[0];
        const uint64_t n_out = W.ne.size() > 1 ? W.ne[1] : 1;
        const uint64_t rank  = A.ne.size() > 1 ? A.ne[1] : 1;
        if (B.ne.size() < 1 || B.ne[0] != rank) { if (err) *err="rank mismatch in "+base_nm; return false; }
        if (A.ne[0] != n_in || (B.ne.size() < 2 || B.ne[1] != n_out)) { if (err) *err="dim mismatch in "+base_nm; return false; }

        const float scale = (alpha > 0.0f) ? adapter_scale * alpha / (float)rank : adapter_scale;

        // 读 A/B f32
        std::vector<float> a_data((size_t)(A.nbytes / 4)), b_data((size_t)(B.nbytes / 4));
        {
            Reader ra; ra.f.open(lora_path, std::ios::binary);
            if (!ra.seek(lora_data_off + A.offset) || !ra.read_exact(a_data.data(), A.nbytes)) { if (err) *err="read lora_a "+base_nm; return false; }
            Reader rb; rb.f.open(lora_path, std::ios::binary);
            if (!rb.seek(lora_data_off + B.offset) || !rb.read_exact(b_data.data(), B.nbytes)) { if (err) *err="read lora_b "+base_nm; return false; }
        }
        // 读 base W Q8_0 -> dequant
        std::vector<uint8_t> w_q((size_t)W.nbytes);
        {
            Reader rw; rw.f.open(base_path, std::ios::binary);
            if (!rw.seek(base_data_off + W.offset) || !rw.read_exact(w_q.data(), W.nbytes)) { if (err) *err="read base "+base_nm; return false; }
        }
        std::vector<float> w_f32((size_t)(n_out * n_in), 0.0f);
        const size_t qrow = (size_t)((n_in / QK8_0) * BLK_Q8_0_SIZE);
        for (uint64_t o = 0; o < n_out; ++o) {
            const uint8_t * rr = w_q.data() + (size_t)o * qrow;
            float * ro = w_f32.data() + (size_t)o * n_in;
            for (uint64_t b = 0; b < n_in / QK8_0; ++b) dequant_q8_block(rr + b*BLK_Q8_0_SIZE, ro + b*QK8_0);
        }
        // delta = scale * B * A ; W' = W + delta
        for (uint64_t o = 0; o < n_out; ++o) {
            float * ro = w_f32.data() + (size_t)o * n_in;
            const float * brow = b_data.data() + (size_t)o * rank;
            for (uint64_t i = 0; i < n_in; ++i) {
                float acc = 0.0f;
                for (uint64_t k = 0; k < rank; ++k) acc += brow[k] * a_data[(size_t)k * n_in + i];
                ro[i] += scale * acc;
            }
        }
        // requant -> Q8_0
        std::vector<uint8_t> wq_new((size_t)W.nbytes);
        for (uint64_t o = 0; o < n_out; ++o) {
            const float * ro = w_f32.data() + (size_t)o * n_in;
            uint8_t * rrq = wq_new.data() + (size_t)o * qrow;
            for (uint64_t b = 0; b < n_in / QK8_0; ++b) quant_q8_block(ro + b*QK8_0, rrq + b*BLK_Q8_0_SIZE);
        }
        const uint64_t patch_off = base_data_off + W.offset;
        if (patch_off + W.nbytes > filesize) { if (err) *err="corrupt base tensor "+base_nm; return false; }
        out.seekp((std::streamoff)patch_off);
        out.write((const char*)wq_new.data(), (std::streamsize)W.nbytes);
        if (!out) { if (err) *err="patch failed "+base_nm; return false; }
        ++patched;
    }
    out.flush(); out.close();

    if (err) err->clear();
    std::fprintf(stderr, "kev_merge: patched %zu adapted tensors into %s\n", patched, out_path.c_str());
    return true;
}

bool merge_lora_cached(const std::string & base_path,
                       const std::string & lora_path,
                       const std::string & out_path,
                       float adapter_scale,
                       std::string * err) {
    {
        std::ifstream f(out_path, std::ios::binary | std::ios::ate);
        if (f) return true;   // 已存在, 幂等跳过
    }
    return merge_lora_into_model(base_path, lora_path, out_path, adapter_scale, err);
}

} // namespace kev