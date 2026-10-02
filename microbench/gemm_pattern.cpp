// gemm_pattern.cpp - standalone microbench: two Q8_0 weight *access pattern* bandwidth comparison.
//
// Question (minimal feasibility check, NOT a full tiled kernel):
//   kevcpp's Q8_0 GEMM falls back to legacy per-token vec_dot_q8_0_q8_0: each token re-reads the
//   whole trunk weight. With M=151 that is ~151 x 504MiB ~= 74GB implied read at ~183GB/s
//   (only ~52% of the ~353GB/s pure-read ceiling).
//   Candidate X = route Q8_0 into ggml's tiled/microtile path: weight read once, reused by M=151 rows.
//
//   Pattern A (per-token / legacy): outer loop over M tokens, each does GEMV over the whole weight
//                                    => weight read M times.
//   Pattern B (batch GEMM / microtile reuse): a weight tile is loaded once and reused across M tokens
//                                    => weight read once.
//   Both do identical MACs (= M*K*N*2); they differ only in memory traffic (A reads weight M x, B once).
//
// The inner dot is faithful to ggml: weight and activation are both Q8_0 (32 int8 + fp32 scale),
// dot = sum(qs_w[i]*qs_x[i])*(d_w*d_x) computed with AVX2 vpmaddubsw. Important: if the kernel is
// compute-bound (rather than DRAM-bound) then A and B will be equally slow - which is itself an honest
// signal that weight-reuse alone gives little (the real win must come from a faster kernel too).
//
// To stay DRAM-bound the total weight buffer is forced large (>=96MiB) via repeated projections
// (--gb), because a single k*n projection can fit in cache.
//
// Build (MSVC, standalone, no vendored changes):
//   call "...\vcvars64.bat"
//   cl /O2 /EHsc /arch:AVX2 gemm_pattern.cpp /Fe:bin\gemm_pattern.exe
//   multithreaded via std::thread (threads partition the N columns, no shared-write race).
//
// Usage:
//   gemm_pattern [--m 151] [--k 1024] [--n 1024] [--gb 0.5] [--threads 32] [--reps 7]
//                [--mode both|a|b]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>
#include <immintrin.h>

static double now_ms() {
    return (double) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() / 1e3;
}

static constexpr int QK8 = 32;
struct BlockQ8 { float d; int8_t qs[QK8]; };

struct Cfg {
    int    m = 151;
    int    k = 1024;
    int    n = 1024;
    double gb = 0.5;
    int    threads = 32;
    int    reps = 7;
    char   mode = 'x';
};

static void parse(Cfg& c, int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--m") c.m = atoi(next().c_str());
        else if (a == "--k") c.k = atoi(next().c_str());
        else if (a == "--n") c.n = atoi(next().c_str());
        else if (a == "--gb") c.gb = atof(next().c_str());
        else if (a == "--threads") c.threads = atoi(next().c_str());
        else if (a == "--reps") c.reps = atoi(next().c_str());
        else if (a == "--mode") { std::string v = next(); c.mode = v[0]; }
        else { std::fprintf(stderr, "bad arg: %s\n", a.c_str()); exit(2); }
    }
}

struct Wbuf {
    std::vector<int8_t> data;
    int64_t k = 0, n = 0, nproj = 0, blkK = 0, bytesPerProj = 0, totalBytes = 0;
    void alloc(int64_t K, int64_t N, int64_t nproj_) {
        k = K; n = N; nproj = nproj_;
        blkK = (k + QK8 - 1) / QK8;
        bytesPerProj = blkK * N * (int64_t)sizeof(BlockQ8);
        totalBytes = bytesPerProj * nproj;
        data.assign((size_t)totalBytes, 0);
    }
    BlockQ8* blk(int p, int iz, int64_t ncol) {
        int64_t off = ((int64_t)p * blkK + iz) * n * (int64_t)sizeof(BlockQ8) + ncol * (int64_t)sizeof(BlockQ8);
        return (BlockQ8*)(data.data() + off);
    }
    const BlockQ8* blk(int p, int iz, int64_t ncol) const {
        int64_t off = ((int64_t)p * blkK + iz) * n * (int64_t)sizeof(BlockQ8) + ncol * (int64_t)sizeof(BlockQ8);
        return (const BlockQ8*)(data.data() + off);
    }
};

// Activation Q8_0 blocks indexed [token][k-block].
struct Ax {
    std::vector<int8_t> data;
    int m = 0, k = 0, blkK = 0;
    void alloc(int m_, int k_) {
        m=m_; k=k_; blkK=(k_+QK8-1)/QK8;
        data.assign((size_t)m*blkK*(int64_t)sizeof(BlockQ8), 0);
    }
    BlockQ8* ab(int t, int iz) { return (BlockQ8*)(data.data() + ((size_t)t*blkK + iz)*(int64_t)sizeof(BlockQ8)); }
    const BlockQ8* ab(int t, int iz) const { return (const BlockQ8*)(data.data() + ((size_t)t*blkK + iz)*(int64_t)sizeof(BlockQ8)); }
};

// Q8_0 x Q8_0 block dot via vpmaddubsw (same as ggml vec_dot_q8_0_q8_0 AVX2 path).
static inline float dot_block_q8(const BlockQ8& w, const BlockQ8& x) {
    __m256i wa = _mm256_loadu_si256((const __m256i*)w.qs);
    __m256i xa = _mm256_loadu_si256((const __m256i*)x.qs);
    __m256i p  = _mm256_maddubs_epi16(xa, wa);
    __m256i s  = _mm256_madd_epi16(p, _mm256_set1_epi16(1));
    s          = _mm256_hadd_epi32(s, s);
    s          = _mm256_hadd_epi32(s, s);
    int32_t acc = _mm_cvtsi128_si32(_mm256_castsi256_si128(s));
    return w.d * x.d * (float)acc;
}

// Pattern A: per-token GEMV (legacy). Outer over tokens; weight re-read for each token.
static void run_pattern_a(const Wbuf& W, const Ax& A, int m, int tid, int nt) {
    const int64_t n = W.n, blkK = W.blkK;
    const int64_t lo = (n * tid) / nt, hi = (n * (tid + 1)) / nt;
    std::vector<float> acc(m, 0.f);
    for (int64_t ncol = lo; ncol < hi; ++ncol)
        for (int p = 0; p < W.nproj; ++p)
            for (int t = 0; t < m; ++t) {
                float a = 0.f;
                for (int64_t iz = 0; iz < blkK; ++iz)
                    a += dot_block_q8(*W.blk(p, iz, ncol), *A.ab(t, iz));
                acc[t] += a;
            }
    volatile float sink=0; for(int t=0;t<m;++t) sink+=acc[t]; (void)sink;
}

// Pattern B: batch GEMM / microtile reuse. Weight block loaded once, reused across M tokens.
static void run_pattern_b(const Wbuf& W, const Ax& A, int m, int tid, int nt) {
    const int64_t n = W.n, blkK = W.blkK;
    const int64_t lo = (n * tid) / nt, hi = (n * (tid + 1)) / nt;
    std::vector<float> acc(m, 0.f);
    for (int p = 0; p < W.nproj; ++p)
        for (int64_t ncol = lo; ncol < hi; ++ncol)
            for (int64_t iz = 0; iz < blkK; ++iz) {
                const BlockQ8* wb = W.blk(p, iz, ncol);          // weight read once
                for (int t = 0; t < m; ++t)                       // reused over M
                    acc[t] += dot_block_q8(*wb, *A.ab(t, iz));
            }
    volatile float sink=0; for(int t=0;t<m;++t) sink+=acc[t]; (void)sink;
}

int main(int argc, char** argv) {
    Cfg c; parse(c, argc, argv);
    if (c.k <= 0 || c.n <= 0 || c.m <= 0) { std::fprintf(stderr, "bad dims\n"); return 2; }

    int64_t blkK = (c.k + QK8 - 1) / QK8;
    int64_t bytesProj = blkK * (int64_t)c.n * (int64_t)sizeof(BlockQ8);
    int64_t nproj = std::max<int64_t>(1, (int64_t)(c.gb * (1024.0*1024.0*1024.0)) / bytesProj);
    int64_t minBytes = (int64_t)(96u << 20);
    nproj = std::max(nproj, (minBytes + bytesProj - 1) / bytesProj);

    Wbuf W; W.alloc(c.k, c.n, nproj);
    Ax A;   A.alloc(c.m, c.k);

    {
        uint32_t seed = 12345u;
        int8_t* base = W.data.data();
        for (int64_t i = 0; i < W.totalBytes; i += 16) { seed = seed*1664525u + 1013904223u; base[i] = (int8_t)(seed >> 24); }
        seed = 99u;
        for (int t = 0; t < c.m; ++t) for (int iz = 0; iz < (int)blkK; ++iz) {
            BlockQ8* b = A.ab(t, iz); b->d = 0.01f;
            for (int j = 0; j < QK8; ++j) { seed = seed*1664525u + 1013904223u; b->qs[j] = (int8_t)(seed >> 24); }
        }
    }
    volatile float sink = 0.f;
    for (int64_t o = 0; o < W.totalBytes; o += 4096) sink += (float)W.data[o];
    for (size_t i = 0; i < A.data.size(); i += 4096) sink += (float)A.data[i];

    const int threads = std::max(1, c.threads);
    double totA = (double)W.totalBytes * c.m;
    double totB = (double)W.totalBytes;

    std::printf("# gemm_pattern M=%d K=%d N=%d gb=%.2f threads=%d nproj=%lld totalBytes=%.1fMiB blkK=%lld mode=%c\n",
                c.m, c.k, c.n, c.gb, threads, (long long)nproj, W.totalBytes/1048576.0, (long long)blkK, c.mode);

    auto bench = [&](const char* name, int which) {
        double best=1e18, sum=0;
        for (int r = 0; r < c.reps; ++r) {
            std::vector<std::thread> pool;
            double t0 = now_ms();
            for (int t = 0; t < threads; ++t)
                pool.emplace_back([&, t]() {
                    if (which==1) run_pattern_a(W, A, c.m, t, threads);
                    else          run_pattern_b(W, A, c.m, t, threads);
                });
            for (auto& th : pool) th.join();
            double dt = now_ms() - t0;
            if (dt < best) best = dt; sum += dt;
        }
        double bytes = (which==1)?totA:totB;
        double gbps  = bytes/1e9/(best/1e3);
        double useful= (double)W.totalBytes*c.m/1e9/(best/1e3);
        std::printf("  %-8s best=%.1f ms  mean=%.1f ms  physW=%.2fGB  implicitBW=%.0f  usefulBW=%.0f GB/s\n",
                    name, best, sum/c.reps, bytes/1e9, gbps, useful);
        return best;
    };

    if (c.mode=='a') { bench("A_pertoken",1); return 0; }
    if (c.mode=='b') { bench("B_batch",2); return 0; }

    double bestA=0, bestB=0, accA=0, accB=0;
    for (int intc=0; intc<c.reps; ++intc) {
        bestA = bench("A_pertoken",1);
        bestB = bench("B_batch",2);
        accA += bestA; accB += bestB;
        double branchA = 425.0 * (W.totalBytes/1048576.0) / 504.0;
        std::printf("    [pair %d] speedup=%.2fx  scaled-504MiB branchA~%.0fms\n", intc, bestA/bestB, branchA);
    }
    double speedup = accA/accB;
    double branchA = 425.0 * (W.totalBytes/1048576.0) / 504.0;
    double branchB = branchA/speedup;
    std::printf("  => A/B speedup(mean) = %.2fx  (bestA_mean=%.1fms  bestB_mean=%.1fms)\n",
                speedup, accA/c.reps, accB/c.reps);
    std::printf("  => scaled-to-504MiB branch(151):  A~%.0fms  B~%.0fms\n", branchA, branchB);
    (void)sink;
    return 0;
}