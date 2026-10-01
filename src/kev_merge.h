// kev_merge.h — merged-LoRA: 把 LoRA 适配器融合(fold)进基础模型权重, 输出一个
// 前向走单份权重的 GGUF(无独立 LoRA matmul)。
//
// 落地方式: 对基础模型 GGUF 逐 block 做 Q8_0 dequant -> f32 加到 LoRA delta(B*A*scale)
// -> 再 Q8_0 requant, 生成的 merged GGUF 与基础模型逐字节相同(非适配张量原样保留,
// 适配张量仅替换数据区)。Q8_0->Q8_0 nbytes 不变, 故数据偏移不变, GGUF 结构原样合法,
// 可被未改动的 llama.cpp 直接作为普通模型加载。
//
// 优点: 完全不动 vendored llama.cpp; 两种路径(分离 LoRA / merged)都可独立加载与 A/B。
//
// 正确性依据: base 权重是 Q8_0 block 紧凑存储(每 32 元素一个 [fp16 scale, int8x32] block)。
// LoRA delta = scale * lora_b * lora_a, scale = alpha / rank (rank = lora_b->ne[0])。
// 前向分离路径算 Q8(W)*x + f32(delta)*x; merged 算 Q8(W+delta)*x。误差同为 Q8 单次量化
// 误差阶别(delta << W), 故预期 argmax 一致率接近分离路径。
#pragma once

#include <string>
#include <vector>

namespace kev {

// 把一个 LoRA GGUF 适配器 fold 进基础模型, 写出 merged GGUF。
//   base_path : 基础模型 Q8_0 GGUF(仅支持被适配张量为 Q8_0)
//   lora_path : llama.cpp 原生 LoRA 适配器 GGUF
//   out_path  : 输出的 merged GGUF(== base, 仅适配张量数据被覆盖)
//   adapter_scale : 与 llama_set_adapters_lora 相同的 adapter scale(默认 1.0)。
// 返回 true 表示成功; 失败置 *err 并返回 false。
bool merge_lora_into_model(const std::string & base_path,
                           const std::string & lora_path,
                           const std::string & out_path,
                           float adapter_scale,
                           std::string * err = nullptr);

// 便捷: 在 out_path 不存在时执行 merge; 存在则跳过(幂等缓存)。用于模型加载前生成。
bool merge_lora_cached(const std::string & base_path,
                       const std::string & lora_path,
                       const std::string & out_path,
                       float adapter_scale,
                       std::string * err = nullptr);

} // namespace kev