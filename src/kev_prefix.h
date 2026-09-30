// kev_prefix.h — 状态前缀 LRU 索引 (M3)
//
// 缓存: 状态指纹(state token ids 序列) -> 一个 seq slot(该序列的 KV + Gated DeltaNet 反复式状态已在其中)。
// 触发: 命中直接用; 未命中把状态 prefill 到一个 slot(LRU 淘汰最久未用)。容量 = slot 数。
// kev 参考: model.py 的 state-prefix cache(prefix / probs_with_prefix / probs_and_prefix; hybrid 下 prefix_min_tokens=0)。
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace kev {

class PrefixLRU {
public:
    // capacity = 可并行缓存的状态数(= 分配的 seq slot 数); 每个 slot 优先从 base_seq 递增分配。
    explicit PrefixLRU(int capacity, int base_seq);

    // 查找 state ids 是否命中; 命中返回 (true, seq_id); 未命中返回 (false, -1)。
    // 未命中时由调用方在返回的 evict_seq(旧 slot, 需先清空) 上 prefill, 再 register.
    std::pair<bool, int> lookup(const std::vector<int32_t> & state_ids, int * evict_seq);

    // 登记: 指定 seq 已 cache 该 state ids(由调用方刚 prefill)。
    void register_(const std::vector<int32_t> & state_ids, int seq);

    // 移除一个 seq 的缓存(清 LRU 项)。
    void invalidate(int seq);

    int capacity() const { return capacity_; }
    int base_seq()  const { return base_seq_; }

    // 索引当前缓存 content (用于诊断/测试): seq -> fingerprint
    std::vector<std::pair<int, std::vector<int32_t>>> contents() const;

private:
    int capacity_;
    int base_seq_;
    struct Entry { std::vector<int32_t> ids; int seq; long long last_use; };
    std::vector<Entry> entries_;
    long long clock_ = 0;
};

} // namespace kev