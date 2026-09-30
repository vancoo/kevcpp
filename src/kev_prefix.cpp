// kev_prefix.cpp
#include "kev_prefix.h"

#include <algorithm>

namespace kev {

PrefixLRU::PrefixLRU(int capacity, int base_seq)
    : capacity_(capacity > 0 ? capacity : 1), base_seq_(base_seq) {
    entries_.reserve(capacity_);
}

std::pair<bool, int> PrefixLRU::lookup(const std::vector<int32_t> & ids, int * evict_seq) {
    *evict_seq = -1;
    // exact match
    for (auto & e : entries_) {
        if (e.ids == ids) {
            e.last_use = ++clock_;
            return {true, e.seq};
        }
    }
    // not cached: pick a slot (evict LRU if full)
    if ((int) entries_.size() < capacity_) {
        int seq = base_seq_ + (int) entries_.size();
        *evict_seq = seq;                     // caller prefill here
        return {false, -1};
    }
    // evict least-recently-used
    auto it = std::min_element(entries_.begin(), entries_.end(),
                               [](const Entry & a, const Entry & b) { return a.last_use < b.last_use; });
    *evict_seq = it->seq;
    entries_.erase(it);
    return {false, -1};
}

void PrefixLRU::register_(const std::vector<int32_t> & ids, int seq) {
    // 防御: 若 entries_ 已含同 seq(latent 隐患), 原位刷新(设 ids + last_use)并返回, 不产生重复项。
    for (auto & e : entries_) {
        if (e.seq == seq) {
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