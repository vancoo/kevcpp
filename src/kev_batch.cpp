// kev_batch.cpp — B2: single-ctx batch scheduler worker.
#include "kev_batch.h"

#include <algorithm>
#include <cstdio>

namespace kev {

BatchScheduler::BatchScheduler(Backend * backend, int32_t n_batch, int32_t max_jobs_per_decode)
    : backend_(backend), n_batch_(n_batch > 0 ? n_batch : 512),
      max_jobs_per_decode_(max_jobs_per_decode) {
    worker_ = std::thread(&BatchScheduler::worker_loop, this);
}

BatchScheduler::~BatchScheduler() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_queue_.notify_all();
    if (worker_.joinable()) worker_.join();
}

bool BatchScheduler::submit(std::shared_ptr<BatchJob> job) {
    if (!backend_ || !job) { if (job) job->err = "scheduler not ready"; return false; }
    {
        std::lock_guard<std::mutex> lk(mu_);
        pending_.push_back(job);
    }
    cv_queue_.notify_one();
    std::unique_lock<std::mutex> lk(job->m);
    job->cv.wait(lk, [&] { return job->done; });
    return job->err.empty();
}

// Decode one flat sequence (with optional prefix copy) possibly chunked to n_batch,
// reading outs (local positions) into logits[]. All outputs must reside in the LAST
// chunk; else this path is not supported and returns false.
namespace {
bool decode_req_chunked(Backend * backend, const Backend::MultiSeqRequest & req,
                        int copy_src, int32_t n_batch,
                        const std::vector<int32_t> & out_pos,
                        std::vector<int32_t> & out_idx,
                        std::string & err) {
    const int32_t L = (int32_t) req.tokens.size();
    out_idx.assign(out_pos.size(), 0);
    if (L <= n_batch) {
        // single decode_multi_seq
        std::vector<Backend::MultiSeqRequest> one(1, req);
        std::vector<int> one_cpy(1, copy_src);
        if (!backend->seq_clear(req.seq)) { err = "seq_clear failed"; return false; }
        if (copy_src >= 0 && !backend->seq_copy(copy_src, req.seq)) { err = "seq_copy failed"; return false; }
        std::vector<std::pair<int32_t,int32_t>> outs;
        for (int32_t p : out_pos) outs.push_back({0, p});
        std::vector<int32_t> oi;
        if (!backend->decode_multi_seq(one, &err, outs, &oi)) return false;
        if (oi.size() != out_pos.size()) { err = "decode: output index count mismatch"; return false; }
        for (size_t i = 0; i < oi.size(); ++i) out_idx[i] = oi[i];
        return true;
    }
    // Chunk: outputs must be in the last chunk.
    const int32_t first_out = out_pos.empty() ? 0 : *std::min_element(out_pos.begin(), out_pos.end());
    if (L - first_out > n_batch) { err = "batched regex outputs span beyond one chunk"; return false; }
    // Prepare the seq once: clear, and copy the cached prefix when this is a tail
    // continuation (KV must exist at positions [0, Dp) before any tail token at >= Dp).
    if (!backend->seq_clear(req.seq)) { err = "seq_clear failed"; return false; }
    if (copy_src >= 0 && !backend->seq_copy(copy_src, req.seq)) { err = "seq_copy failed"; return false; }
    // head chunks: no outputs (continue from copied prefix or from pos 0)
    int32_t pos = 0;
    while (pos < first_out) {
        const int32_t end = std::min(first_out, pos + n_batch);
        Backend::MultiSeqRequest seg;
        seg.tokens.assign(req.tokens.begin() + pos, req.tokens.begin() + end);
        seg.start_pos = req.start_pos + pos;
        seg.seq = req.seq;
        std::vector<Backend::MultiSeqRequest> one(1, seg);
        std::string e2;
        if (!backend->decode_multi_seq(one, &e2, {}, nullptr)) { err = "head chunk decode failed: " + e2; return false; }
        pos = end;
    }
    // tail chunk: outputs
    const int32_t tail_start = first_out;
    Backend::MultiSeqRequest seg;
    seg.tokens.assign(req.tokens.begin() + tail_start, req.tokens.end());
    seg.start_pos = req.start_pos + tail_start;
    seg.seq = req.seq;
    std::vector<Backend::MultiSeqRequest> one(1, seg);
    std::vector<std::pair<int32_t,int32_t>> outs;
    std::vector<int32_t> local_pos;
    for (int32_t g : out_pos) { outs.push_back({0, g - tail_start}); local_pos.push_back(g - tail_start); }
    std::vector<int32_t> oi;
    if (!backend->decode_multi_seq(one, &err, outs, &oi)) return false;
    if (oi.size() != out_pos.size()) { err = "decode: output index count mismatch"; return false; }
    for (size_t i = 0; i < oi.size(); ++i) out_idx[i] = oi[i];
    return true;
}
} // namespace

void BatchScheduler::worker_loop() {
    std::vector<std::shared_ptr<BatchJob>> batch;

    for (;;) {
        batch.clear();
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_queue_.wait(lk, [&] { return stop_ || !pending_.empty(); });
            if (stop_) return;
            // B3: bounded dynamic batch. Take at most max_jobs_per_decode_ jobs
            // (0 = unlimited) — leaves the rest for the next drain (continuous).
            const size_t cap = max_jobs_per_decode_ > 0 ? (size_t)max_jobs_per_decode_ : pending_.size();
            const size_t take = std::min(cap, pending_.size());
            batch.assign(pending_.begin(), pending_.begin() + take);
            pending_.erase(pending_.begin(), pending_.begin() + take);
            // B3: wait a short bounded aggregation window ONLY when the queue is sparse
            // (fewer than a full batch). If the queue is already full we decode
            // immediately (no added latency under load).
            const bool full_batch = max_jobs_per_decode_ > 0 ? (batch.size() >= (size_t)max_jobs_per_decode_) : (batch.size() == pending_.size());
            if (!full_batch && agg_us_ > 0) {
                lk.unlock();
                std::this_thread::sleep_for(std::chrono::microseconds(agg_us_));
                lk.lock();
                const size_t cap2 = max_jobs_per_decode_ > 0 ? (size_t)max_jobs_per_decode_ : (size_t)-1;
                if (!pending_.empty() && batch.size() < cap2) {
                    const size_t more = std::min(cap2 - batch.size(), pending_.size());
                    const size_t base = batch.size();
                    batch.resize(base + more);
                    for (size_t i = 0; i < more; ++i) batch[base + i] = pending_[i];
                    pending_.erase(pending_.begin(), pending_.begin() + more);
                }
            }
        }

        // ---- Build one merged decode_multi_seq across ALL jobs' sequences ----
        // flat reqs across jobs, each tagged with owning job + local req index.
        struct Flat { Backend::MultiSeqRequest req; int copy_src; int job; int r; int ntok; };
        std::vector<Flat> flatin;
        for (size_t j = 0; j < batch.size(); ++j) {
            const auto & job = batch[j];
            std::unique_lock<std::mutex> jlk(job->m);
            if (job->done) continue;
            for (size_t r = 0; r < job->reqs.size(); ++r) {
                Flat f;
                f.req = job->reqs[r];
                f.copy_src = (r < job->copy_src.size()) ? job->copy_src[r] : -1;
                f.job = (int) j; f.r = (int) r;
                f.ntok = (int32_t) job->reqs[r].tokens.size();
                flatin.push_back(std::move(f));
            }
            jlk.unlock();
        }
        if (!flatin.empty()) {
            // Partition into chunks each with total tokens <= n_batch. A single huge req
            // (ntok > n_batch) is decoded by itself via chunked helper.
            std::vector<int> chunk_assignment(flatin.size(), -1);
            std::vector<int64_t> chunk_tok;
            std::vector<std::vector<int>> chunk_members;  // indices into flatin
            for (size_t i = 0; i < flatin.size(); ++i) {
                if (flatin[i].ntok > n_batch_) {
                    // give it its own chunk (chunked helper handles it alone)
                    chunk_assignment[i] = (int) chunk_members.size();
                    chunk_members.push_back({(int) i});
                    chunk_tok.push_back(flatin[i].ntok);
                    continue;
                }
                // try to fit in an existing chunk, else open one
                int placed = -1;
                for (size_t c = 0; c < chunk_members.size(); ++c) {
                    if (chunk_tok[c] + flatin[i].ntok <= n_batch_) { placed = (int) c; break; }
                }
                if (placed < 0) { placed = (int) chunk_members.size(); chunk_members.push_back({}); chunk_tok.push_back(0); }
                chunk_members[placed].push_back((int) i);
                chunk_tok[placed] += flatin[i].ntok;
                chunk_assignment[i] = placed;
            }

            // One decode_multi_seq per chunk; chunk of a single oversize req uses helper.
            for (size_t ci = 0; ci < chunk_members.size(); ++ci) {
                std::vector<Backend::MultiSeqRequest> creqs;
                std::vector<int> c_copy;
                std::vector<int> c_owner;   // flat index -> owner job
                std::vector<int> c_flat_member; // which flatin index
                int64_t total = 0;
                for (int m : chunk_members[ci]) {
                    creqs.push_back(flatin[m].req);
                    c_copy.push_back(flatin[m].copy_src);
                    c_owner.push_back(flatin[m].job);
                    c_flat_member.push_back(m);
                    total += flatin[m].ntok;
                }
                if (creqs.size() == 1 && flatin[chunk_members[ci][0]].ntok > n_batch_) {
                    // oversize single sequence: chunk it internally
                    const int m = chunk_members[ci][0];
                    auto & job = batch[flatin[m].job];
                    std::unique_lock<std::mutex> jlk(job->m);
                    // gather that job's outs that belong to this req
                    std::vector<int32_t> out_pos;
                    for (size_t o = 0; o < job->outs.size(); ++o) {
                        if (job->outs[o].first == flatin[m].r) out_pos.push_back(job->outs[o].second);
                    }
                    std::vector<int32_t> oi;
                    std::string e2;
                    if (decode_req_chunked(backend_, flatin[m].req, flatin[m].copy_src, n_batch_,
                                           out_pos, oi, e2)) {
                        // 防护(2026-10-30): oi 的长度由 out_pos 决定, 正常情况下与下面的
                        // 遍历范围一致。一旦后端回填的 oi 比预期短, oi[src_out] 就是越界读,
                        // 会踩坏 std::vector 控制块并在 CRT 里触发 0xc0000409。这里显式校验。
                        if (oi.size() != out_pos.size()) {
                            job->err = "oversize decode: out_idx size " + std::to_string(oi.size())
                                     + " != outputs " + std::to_string(out_pos.size());
                            jlk.unlock();
                            continue;
                        }
                        // write logits for each output of this req
                        job->logits.resize(job->outs.size(), {});
                        int src_out = 0;
                        for (size_t o = 0; o < job->outs.size(); ++o) {
                            if (job->outs[o].first != flatin[m].r) continue;
                            if (src_out >= (int) oi.size()) {
                                job->err = "oversize decode: out_idx exhausted at output "
                                         + std::to_string(o);
                                break;
                            }
                            job->logits[o].resize(job->out_cands[o].size());
                            std::string rd_err;
                            for (size_t c = 0; c < job->out_cands[o].size(); ++c) {
                                float v = 0.0f;
                                if (!backend_->read_logits(oi[src_out], job->out_cands[o][c], &v, &rd_err)) {
                                    if (job->err.empty()) job->err = "read_logits failed at output "
                                        + std::to_string(o) + " candidate " + std::to_string(c) + ": " + rd_err;
                                    break;
                                }
                                job->logits[o][c] = v;
                            }
                            if (!job->err.empty()) break;
                            ++src_out;
                        }
                    } else {
                        if (job->err.empty()) job->err = e2;
                    }
                    jlk.unlock();
                    continue;
                }

                // prepare all reqs in this chunk (clear + maybe prefix copy)
                bool ok = true;
                for (size_t r = 0; r < creqs.size(); ++r) {
                    if (!backend_->seq_clear(creqs[r].seq)) { ok = false; break; }
                    if (c_copy[r] >= 0 && !backend_->seq_copy(c_copy[r], creqs[r].seq)) { ok = false; break; }
                }
                if (!ok) {
                    for (int m : chunk_members[ci]) {
                        auto & job = batch[flatin[m].job];
                        std::lock_guard<std::mutex> jlk(job->m);
                        if (job->err.empty()) job->err = "seq prepare failed";
                    }
                    continue;
                }
                // build global outs (flat req idx within chunk) + read logits per owner job
                std::vector<std::pair<int32_t,int32_t>> gouts;
                std::vector<std::pair<int,int>> gout_owner; // (job idx, out idx in job)
                for (size_t r = 0; r < creqs.size(); ++r) {
                    const int jobi = c_owner[r];
                    auto & job = batch[jobi];
                    std::lock_guard<std::mutex> jlk(job->m);
                    for (size_t o = 0; o < job->outs.size(); ++o) {
                        if (flatin[c_flat_member[r]].r == job->outs[o].first) {
                            gouts.push_back({(int) r, job->outs[o].second});
                            gout_owner.push_back({jobi, (int) o});
                        }
                    }
                }
                std::vector<int32_t> goi;
                std::string de_err;
                if (!backend_->decode_multi_seq(creqs, &de_err, gouts, &goi) || goi.size() != gouts.size()) {
                    // mark all jobs in this chunk failed
                    for (int m : chunk_members[ci]) {
                        auto & job = batch[flatin[m].job];
                        std::lock_guard<std::mutex> jlk(job->m);
                        if (job->err.empty()) job->err = "decode_multi_seq failed: " + de_err;
                    }
                } else {
                    for (size_t i = 0; i < gouts.size(); ++i) {
                        const int jobi = gout_owner[i].first;
                        const int oi_ = gout_owner[i].second;
                        auto & job = batch[jobi];
                        std::lock_guard<std::mutex> jlk(job->m);
                        job->logits.resize(job->outs.size(), {});
                        job->logits[oi_].resize(job->out_cands[oi_].size());
                        std::string rd_err;
                        for (size_t c = 0; c < job->out_cands[oi_].size(); ++c) {
                            float v = 0.0f;
                            if (!backend_->read_logits(goi[i], job->out_cands[oi_][c], &v, &rd_err)) {
                                if (job->err.empty()) job->err = "read_logits failed at output "
                                    + std::to_string(oi_) + " candidate " + std::to_string(c) + ": " + rd_err;
                                break;
                            }
                            job->logits[oi_][c] = v;
                        }
                    }
                }
            }
        }

        // mark all done + notify
        for (auto & job : batch) {
            std::unique_lock<std::mutex> jlk(job->m);
            job->done = true;
            jlk.unlock();
            job->cv.notify_all();
        }
        cv_idle_.notify_all();
    }
}

} // namespace kev
