// kev_backend_moe.h — shared helpers for MoE expert offload across backends.
//
// MoE 专家 offload 的加载侧机制: 用 llama_model_params.tensor_buft_overrides 把**专家权重张量**
// (GGUF 中命名含 `_exps` 的 ffn_gate_exps / ffn_up_exps / ffn_down_exps / ffn_gate_up_exps)
// 强制分配到 CPU(host RAM), 而 attention / shared / door 等其余张量仍按 n_gpu_layers 进 VRAM。
// 这样只把"模型体积大头"的专家留在系统内存, 显著降低显存占用。路由 gating 张量 ffn_gate_inp
// (命名不含 _exps)不受影响, 留在 device, 从而"按需算出要哪些专家"仍在 GPU 上。
//
// 该机制等价于 llama.cpp 的 `--n-cpu-moe`(PR #15077), 且不改动 vendored llama.cpp 源码——
// 仅通过 llama.h 公开的 tensor_buft_overrides 传参实现。
//
// ABI 注意: 本头只被各后端实现模块(kev_backend_{cpu,vulkan,gpu}.cpp)包含, 这些模块是进程内
// 唯一的 llama 链接方(single-llama rule), 所以这里可以直接用 llama.h / ggml-backend.h 类型。
#pragma once

#include "kev_backend.h"   // kev_backend_params (llama-free struct)

#include "llama.h"
#include "ggml-backend.h"

#include <vector>

namespace kev_backend_moe {

// 判断该模型的专家张量命名模式(用于 tensor_buft_overrides 正则匹配)。
// 专家权重张量统一含 `_exps` 后缀(如 blk.5.ffn_up_exps.weight); 路由/门控 ffn_gate_inp 不含。
inline const char * expert_tensor_pattern() {
    return ".*_exps.*";
}

// 构造"把专家留 host RAM"的 tensor_buft_overrides 数组。
// 仅当 moe_offload 且 n_gpu_layers != 0(即走 GPU 且需省显存)时返回非空数组;
// 否则返回空(专家跟随 n_gpu_layers, 由 llama.cpp 自行放置——保持各家后端 legacy 行为不变)。
inline std::vector<llama_model_tensor_buft_override> build_moe_overrides(int moe_offload, int n_gpu_layers) {
    std::vector<llama_model_tensor_buft_override> out;
    if (moe_offload && n_gpu_layers != 0) {
        llama_model_tensor_buft_override o{};
        o.pattern = expert_tensor_pattern();
        o.buft    = ggml_backend_cpu_buffer_type();
        out.push_back(o);
    }
    return out;
}

} // namespace kev_backend_moe