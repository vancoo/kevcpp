// kev_prefix.cpp
#include "kev_prefix.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>

namespace kev {

PrefixLRU::PrefixLRU(int capacity, int base_seq)
    : capacity_(capacity > 0 ? capacity : 1), base_seq_(base_seq) {
    entries_.reserve(capacity_);
}

// 判断 a 是否为 b 的前缀, 返回公共前缀(=a)长度; 不是前缀则返回 -1。
namespace {
int prefix_len_if_prefix(const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
    if (a.size() > b.size()) return -1;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) return -1;
    return (int) a.size();
}
} // namespace

PrefixMatch PrefixLRU::lookup(const std::vector<int32_t> & ids, int min_lcp) {
    PrefixMatch m;
    if (min_lcp < 1) min_lcp = 1;
    Entry * best = nullptr;
    int best_len = 0;
    for (auto & e : entries_) {
        // 只能整体复用(见 kev_model P2 注释: M-RoPE 要求新解码起始位置 > 缓存 KV 末尾位置,
        // 不能复用更长缓存序列的中间段)。故仅当「缓存的整段前缀是请求 ids 的前缀」才命中,
        // 复用长度 = 该缓存槽的完整存储长度。
        const int plen = prefix_len_if_prefix(e.ids, ids);
        if (plen >= min_lcp && plen > best_len) {
            best = &e;
            best_len = plen;
        }
    }
    if (best) {
        best->last_use = ++clock_;
        m.hit = true;
        m.seq = best->seq;
        m.lcp = best_len;
        return m;
    }
    // 未命中: 选一个槽(LRU 淘汰)供调用方 seq_clear + prefill + register。
    m.hit = false;
    if ((int) entries_.size() < capacity_) {
        m.evict_seq = base_seq_ + (int) entries_.size();   // 全新未用槽
        return m;
    }
    auto it = std::min_element(entries_.begin(), entries_.end(),
                               [](const Entry & a, const Entry & b) { return a.last_use < b.last_use; });
    m.evict_seq = it->seq;
    entries_.erase(it);
    return m;
}

void PrefixLRU::register_(const std::vector<int32_t> & ids, int seq) {
    // 防御: 若 entries_ 已含同 seq, 原位刷新并返回, 不产生重复项。
    // 2026-10-30: 这里以前是静默刷新。同 seq 重复登记说明上面的分配逻辑出了问题
    // (两条逻辑前缀指向同一个 KV 槽), 后者 prefill 会覆盖前者。改为告警, 让问题可见。
    for (auto & e : entries_) {
        if (e.seq == seq) {
            std::fprintf(stderr,
                "kev_prefix: WARN duplicate register for seq=%d "
                "(existing len=%zu, new len=%zu) -- seq slot reuse, KV may be shared\n",
                seq, e.ids.size(), ids.size());
            e.ids = ids; e.last_use = ++clock_;
            return;
        }
    }
    entries_.push_back({ids, seq, ++clock_});
}

void PrefixLRU::invalidate(int seq) {
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                  [&](const Entry & e) { return e.seq == seq; }),
                   entries_.end());
}

std::vector<std::pair<int, std::vector<int32_t>>> PrefixLRU::contents() const {
    std::vector<std::pair<int, std::vector<int32_t>>> v;
    for (const auto & e : entries_) v.emplace_back(e.seq, e.ids);
    return v;
}

} // namespace kev