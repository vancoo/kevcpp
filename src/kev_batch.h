// kev_batch.h — B2: single-ctx batch scheduler for the logits decode step.
//
// The kevcpp llama_context is not thread-safe for concurrent llama_decode, so the
// P0 fix serialized every request on compute_mutex_ (throughput = 1 request at a
// time). B2 replaces that with a *batch scheduler*: per-request threads keep doing
// their orchestration (tokenize, LCP/prefix lookups, candidate selection), but the
// actual llama_decode + read_logits are submitted to ONE background worker. The
// worker coalesces the currently-pending decode jobs (across many requests) into a
// single decode_multi_seq call (B1 primitive), so C concurrent requests that were
// previously C serially-scheduled decodes now decode together in one batched pass,
// amortizing llama.cpp's per-decode fixed overhead.
//
// Safety model:
//   * ONLY the worker thread touches llama_decode / seq_clear / seq_copy /
//     read_logits — single-llama-context access is guaranteed.
//   * read_logits happens inside the worker immediately after the decode (while it
//     still owns the context), so a request never reads a logits buffer that a later
//     decode overwrote -> no data race.
//   * Each in-flight request holds a distinct sequence-id slot from a pool, so merged
//     batches never collide on KV.
//   * The worker merges at most `n_batch` total tokens per decode_multi_seq call; if a
//     single job exceeds n_batch it is run alone (still correct, just not coalesced).
#pragma once

#include "kev_backend_factory.h"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace kev {

// A single decode job submitted by a request to the batch scheduler.
//   * reqs[c]    = c-th sequence to decode in ONE llama_decode (a request may supply
//                  one sequence here; the scheduler merges it with other requests').
//   * copy_src[r]= prefix seq id whose KV must be copied into reqs[r].seq before the
//                  decode (LCP tail continuation), or -1 for a fresh seq.
//   * outs       = (req_idx, local_pos) output positions.
//   * out_cands  = per output, the token ids whose logits the worker should read and
//                  store (the worker reads them while it owns the context).
// On completion job.done == true and logits[o][c] holds the c-th candidate's logit at
// out o. err is set on failure.
struct BatchJob {
    std::vector<Backend::MultiSeqRequest> reqs;
    std::vector<int>                      copy_src;      // per req
    std::vector<std::pair<int32_t,int32_t>> outs;        // (req_idx, local_pos)
    std::vector<std::vector<int32_t>>     out_cands;     // per out: token ids to read

    std::vector<std::vector<float>> logits;              // [out][cand]
    std::string err;
    bool done = false;

    std::mutex              m;
    std::condition_variable cv;
};

// B2/B3: single-ctx decode worker. Construct with the Backend (owned by KevModel) and
// the max tokens per llama_decode (n_batch). submit() blocks until the worker has
// decoded the job's sequences (possibly batched with other pending jobs) and filled
// job.logits.
//
// B3 (continuous batching): the worker uses a *bounded dynamic batch* — it takes at
// most max_jobs_per_decode jobs per drain, and decodes IMMEDIATELY (no wait) whenever
// the pending queue already holds a full batch; it only waits a short, bounded
// aggregation window (aggregation_us) when the queue is sparse (fewer jobs than the
// cap), hoping to coalesce a few simultaneous arrivals. This keeps the batch size
// adaptive to load while adding minimal latency to sparse traffic (lower P95).
class BatchScheduler {
public:
    // max_jobs_per_decode > 0 caps jobs coalesced into one drain (default = all pending
    // up to n_batch tokens). Pass ~max_concurrency to bound batch size to available seqs.
    BatchScheduler(Backend * backend, int32_t n_batch, int32_t max_jobs_per_decode = 0);
    ~BatchScheduler();

    // Returns false (and sets job->err) if the job could not be decoded. Blocks until
    // the worker handles it. thread-safe; may be called from many request threads.
    bool submit(std::shared_ptr<BatchJob> job);

    // Max aggregation window (microseconds) the worker waits with a SPARSE queue hoping
    // for more jobs before decoding, all else equal. 0 = never wait. Default 400.
    void set_aggregation_us(long agg_us) { agg_us_ = agg_us; }
    long aggregation_us() const { return agg_us_; }

private:
    void worker_loop();

    Backend * backend_;
    int32_t   n_batch_;
    int32_t   max_jobs_per_decode_ = 0;
    long      agg_us_ = 400;        // bounded coalesce window (sparse queue only)

    std::mutex              mu_;
    std::condition_variable cv_queue_;
    std::condition_variable cv_idle_;
    std::vector<std::shared_ptr<BatchJob>> pending_;
    bool stop_ = false;
    std::thread worker_;
};

} // namespace kev