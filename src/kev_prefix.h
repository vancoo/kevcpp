// kev_prefix.h — 最长公共前缀(LCP) LRU 索引 (P2)
//
// 缓存: 一段「稳定公共前缀」token ids -> 一个 seq slot(该前缀的 KV 已在其中)。
// 匹配语义为"最长公共前缀(LCP)", 且受 M-RoPE 约束 -> 只能整体复用一个槽的完整存储前缀:
//   * lookup(ids, min_lcp): 找到缓存中「其存的整段前缀是 ids 的前缀」且长度 >= min_lcp 的槽,
//     取长度最长者; 复用长度 = 该槽的完整存储长度(不能复用更长缓存序列的中间段, 否则 KV 位置
//     回退触发 M-RoPE X<Y 校验失败)。命中返回 {hit, seq, lcp=存储长度}; 否则未命中, 返回
//     evict_seq 供调用方 seq_clear 后 prefill 一个新前缀再 register。
// 触发: 命中直接用; 未命中把新前缀 prefill 到一个 slot(LRU 淘汰最久未用)。容量 = slot 数。
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace kev {

// 一次 LCP 查找的结果。
struct PrefixMatch {
    bool hit       = false;   // 命中: 找到一个缓存槽, 其存的整段前缀是请求 ids 的前缀(公共前缀语义)
    int  seq       = -1;      // 命中的缓存槽(base_seq 之上), KV 覆盖 [0..lcp)
    int  lcp       = 0;       // 命中时: 复用长度 = 该缓存槽的完整存储的前缀长度
    int  evict_seq = -1;      // 未命中时: 需先 seq_clear 再 prefill 的旧槽(可能为全新未用槽)
};

class PrefixLRU {
public:
    // capacity = 可并行缓存的前缀数(= 分配的 seq slot 数); 每个 slot 优先从 base_seq 递增分配。
    explicit PrefixLRU(int capacity, int base_seq);

    // 查找请求 ids(整条 prompt token ids)是否与某缓存槽有足够长公共前缀。
    // ids 恒为"待测请求的整条 prompt"; 对每个缓存槽逐位比对求 LCP, 取最长者。
    // min_lcp: 命中所需最小 LCP 长度。命中返回 {hit=true, seq, lcp=原 LCP}; 否则
    // 返回 {hit=false, evict_seq=待 prefill 的槽}(LRU 淘汰)供调用方 prefill + register。
    PrefixMatch lookup(const std::vector<int32_t> & ids, int min_lcp);

    // 登记: 指定 seq 已 cache 该 ids(由调用方刚 prefill)。
    void register_(const std::vector<int32_t> & ids, int seq);

    // 移除一个 seq 的缓存(清 LRU 项)。
    void invalidate(int seq);

    int capacity() const { return capacity_; }
    int base_seq()  const { return base_seq_; }

    // 索引当前缓存 content (用于诊断/测试): seq -> 缓存的 token ids
    std::vector<std::pair<int, std::vector<int32_t>>> contents() const;

private:
    int capacity_;
    int base_seq_;
    struct Entry { std::vector<int32_t> ids; int seq; long long last_use; };
    std::vector<Entry> entries_;
    long long clock_ = 0;
};

} // namespace kev