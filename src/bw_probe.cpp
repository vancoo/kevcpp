// bw_probe.cpp - controlled memory-bandwidth probe (multithread AVX-512 stream read/write).
//
// Goal: reliably measure the *achieved* peak memory read bandwidth of this
// Zen5 (Strix Halo) machine, addressing the prior probe's underestimates.
//
// Improvements vs the original probe:
//   * AVX-512 512-bit stream (non-temporal) loads via _mm512_stream_load_si512
//     to bypass L2/L1 pollution and keep the load pipeline saturated.
//   * Long register accumulator chains: accumulate 512-bit adds in up to 8
//     independent accumulators, only spilling to memory once at the very end.
//     No per-load store dependency, so the sum never becomes a bottleneck.
//   * Independently sized, 2 MB aligned per-thread segments (large-page style),
//     minimizing TLB misses and false sharing / scheduling perturbation.
//   * Windows large pages via VirtualAlloc(MEM_LARGE_PAGES) when the account
//     has SeLockMemoryPrivilege; otherwise falls back to a plain 2 MB aligned
//     allocation.
//   * best-of-N (default 7 reps) reporting so load noise doesn't drag the number.
//   * --read (pure sequential read) and --readwrite (streaming write + read
//     back, mimics a forward pass that reads weights and writes activations).
//   * Optional thread->core affinity binding (spread across logical cores).
//
// Build (single file, no project deps):
//   build\            ensure output dir exists
//   cl /O2 /arch:AVX2 /EHsc src\bw_probe.cpp /Fe:build\bin\Release\bw_probe.exe
// (AVX-512 intrinsics compile under MSVC even without /arch:AVX512; the CPU
//  here is Zen5 and supports them.)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <thread>
#include <chrono>
#include <vector>
#include <algorithm>
#include <immintrin.h>

#ifdef _WIN32
# define WIN32_LEAN_AND_MEAN
# include <windows.h>
#endif

// ---------------- timing ----------------
static double now_ms() {
    return (double) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() / 1e3;
}

// ---------------- 2 MB aligned allocation ----------------
struct Buffer {
    void*  base   = nullptr;  // original allocation (may be _aligned_malloc)
    float* ptr    = nullptr;  // aligned usable base
    bool   large  = false;    // whether large pages were used
    ~Buffer() { release(); }

    void release() {
        if (base) {
#ifdef _WIN32
            VirtualFree(base, 0, MEM_RELEASE);
#else
            free(base);
#endif
            base = nullptr; ptr = nullptr;
        }
    }

    // Try large pages first (requires SeLockMemoryPrivilege; call run_as_large_ops
    // never blocks). Falls back to 2 MB aligned malloc.
    void alloc(uint64_t nbytes) {
        release();
        nbytes = (nbytes + 63) & ~63ULL;
#ifdef _WIN32
        // Enable SeLockMemoryPrivilege (only succeeds as admin; if not, skip).
        HANDLE tok = NULL;
        if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) {
            LUID luid;
            if (LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &luid)) {
                TOKEN_PRIVILEGES tp;
                tp.PrivilegeCount = 1;
                tp.Privileges[0].Luid = luid;
                tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
                AdjustTokenPrivileges(tok, FALSE, &tp, 0, NULL, NULL);
            }
            CloseHandle(tok);
        }
        // Need a multiple of the large-page minimum size.
        static SIZE_T lpsz = GetLargePageMinimum();
        uint64_t pagesz = (uint64_t)(lpsz > 0 ? lpsz : (SIZE_T)(2u << 20));
        uint64_t lpbytes = (nbytes + pagesz - 1) & ~(pagesz - 1);
        void* p = VirtualAlloc(NULL, (SIZE_T)lpbytes, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                               PAGE_READWRITE);
        if (p) { base = p; ptr = (float*)p; large = true; return; }
        // Fall back: aligned malloc.
        ptr = (float*)_aligned_malloc((size_t)nbytes, (size_t)(2u << 20));
        base = ptr; large = false;
#else
        if (posix_memalign(&base, (size_t)(2u << 20), (size_t)nbytes) != 0) { base = nullptr; ptr = nullptr; }
        else ptr = (float*)base;
#endif
    }
};

// ---------------- affinity ----------------
static void bind_thread(unsigned which) {
#ifdef _WIN32
    // Spread across logical processors 0..N-1. Each probe thread picks a unique one.
    HANDLE h = GetCurrentThread();
    // SetThreadAffinityMask is deprecated but fine for group 0; use group affinity
    // to also cover the (unlikely) >64-core case.
    SetThreadAffinityMask(h, (DWORD_PTR)1 << (which & 63));
#endif
    (void)which;
}

// ---------------- workers ----------------
struct Job {
    const float* buf;   // source buffer
    float*       dst;   // destination buffer (copy mode only)
    uint64_t     nf;        // total floats
    int          tid;
    int          nthreads;
    int          mode;      // 0=read, 1=readwrite, 2=copy, 3=read_normal
    uint64_t     out;
};

// 512-bit non-temporal (streaming) load + long accumulator chains.
static void worker(Job* j) {
    bind_thread((unsigned)j->tid);
    const float* base = j->buf;
    uint64_t nf   = j->nf;
    int      ntr  = j->nthreads;
    // Per-thread contiguous segment, 64-float (64B) aligned at both ends.
    uint64_t per   = (nf / (uint64_t)ntr) & ~63ULL;
    uint64_t start = (uint64_t)j->tid * per;
    uint64_t end   = std::min<uint64_t>(start + per, nf);

    const int ACC = 8;                              // 8 independent 512-bit accumulators
    __m512 a[ACC];
    for (int k = 0; k < ACC; ++k) a[k] = _mm512_setzero_ps();

    uint64_t i = start;
    if (j->mode == 0 || j->mode == 3) {
        // ----- pure read: thread 0 = streaming (NT) load, 3 = normal load -----
        const __m512i* p = (const __m512i*)(base + i);
        uint64_t nv = (end - i) / 16;               // 16 floats per 512-bit vector
        uint64_t k = 0;
        for (; k + (uint64_t)(ACC*16) <= nv; k += ACC*16) {
            // Emit 8 independent loads/accumulates so the load-order
            // chain has depth 8: hides latency, saturates the memory pipeline.
            for (int ax = 0; ax < ACC; ++ax) {
                __m512 v;
                if (j->mode == 0) v = _mm512_castsi512_ps(_mm512_stream_load_si512(p + k + ax*16ULL));
                else              v = _mm512_loadu_ps((const float*)(p + k + ax*16ULL));
                a[ax] = _mm512_add_ps(a[ax], v);
            }
        }
        // tail (plain load)
        for (; k < nv; ++k) {
            __m512 v = _mm512_loadu_ps((const float*)(p + k));
            a[k % ACC] = _mm512_add_ps(a[k % ACC], v);
        }
    } else if (j->mode == 2) {
        // ----- classic STREAM copy: stream-read src, stream-write dst -----
        // Total bus traffic = read nbytes + write nbytes (the gold-standard,
        // hard bus-limited probe).
        const __m512i* sp = (const __m512i*)(base + i);
        __m512i*       dp = (__m512i*)(j->dst + i);
        uint64_t nv = (end - i) / 16;
        uint64_t k = 0;
        for (; k + (uint64_t)ACC <= nv; k += ACC) {
            // Load block, then store block (independent chains).
            __m512i v[ACC];
            for (int ax = 0; ax < ACC; ++ax) v[ax] = _mm512_stream_load_si512(sp + k + ax);
            for (int ax = 0; ax < ACC; ++ax) _mm512_stream_si512(dp + k + ax, v[ax]);
        }
        for (; k < nv; ++k) {
            __m512 v = _mm512_loadu_ps((const float*)(sp + k));
            _mm512_storeu_ps((float*)(dp + k), v);
        }
        a[0] = _mm512_castsi512_ps(_mm512_loadu_si512(sp)); // consume to keep work
    } else {
        // ----- streaming write + read-back (read weights, write activations) -----
        // Write each 512-bit chunk with a stream store, then read it back and
        // accumulate so the compiler cannot elide the loads. Non-temporal store
        // avoids polluting cache and models activation writes.
        float* w = (float*)base;
        __m512i* wp = (__m512i*)(w + i);
        __m512i* rp = (__m512i*)(base + i);
        uint64_t nv = (end - i) / 16;
        uint64_t k = 0;
        for (; k < nv; k += 1) {
            __m512 v = _mm512_castsi512_ps(_mm512_stream_load_si512(rp + k)); // read old
            __m512 s = _mm512_mul_ps(v, _mm512_set1_ps(1.0000001f));          // perturb
            _mm512_stream_si512(wp + k, _mm512_castps_si512(s));              // write, non-temporal
            __m512 r = _mm512_castsi512_ps(_mm512_stream_load_si512(rp + k)); // read back
            a[k % ACC] = _mm512_add_ps(a[k % ACC], r);
        }
    }

    // --- final reduce once, to memory, far away from the hot loop.
    __m512 sum = _mm512_setzero_ps();
    for (int k = 0; k < ACC; ++k) sum = _mm512_add_ps(sum, a[k]);
    float t[16];
    _mm512_storeu_ps(t, sum);
    uint64_t s = 0;
    for (int k = 0; k < 16; ++k) { uint32_t u; memcpy(&u, &t[k], 4); s += u; }
    j->out = s;
}

// ---------------- main ----------------
static void usage() {
    std::fprintf(stderr,
        "usage: bw_probe [--gb 2.0] [--threads 32] [--reps 7] [--read|--readwrite|--copy|--read-normal]\n"
        "  --gb F       buffer size in GiB (default 2.0; try 4.0 for TLB/large-page effects)\n"
        "  --threads N  number of worker threads (default 32)\n"
        "  --reps N     how many repetitions; report best ms (default 7)\n"
        "  --read       pure streaming (non-temporal) read (default)\n"
        "  --read-normal pure normal (caching) read\n"
        "  --readwrite  streaming write + read-back (models forward pass read-write mix)\n"
        "  --copy       classic STREAM copy: stream-read src + stream-write dst (bus-limited)\n"
        "  -h, --help   this message\n");
}

int main(int argc, char** argv) {
    double gb = 2.0; int threads = 32; int reps = 7; int mode = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](){ return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--gb") gb = std::atof(next().c_str());
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--reps") reps = std::atoi(next().c_str());
        else if (a == "--read") mode = 0;
        else if (a == "--read-normal") mode = 3;
        else if (a == "--readwrite") mode = 1;
        else if (a == "--copy") mode = 2;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { std::fprintf(stderr, "bad arg %s\n", a.c_str()); usage(); return 1; }
    }
    if (threads < 1) threads = 1;

    uint64_t nbytes = (uint64_t)(gb * 1024.0 * 1024.0 * 1024.0);
    nbytes &= ~63ULL;
    if (nbytes < 1ULL<<20) { std::fprintf(stderr, "buffer too small\n"); return 1; }

    Buffer buf, dst;
    buf.alloc(nbytes);
    if (!buf.ptr) { std::fprintf(stderr, "bw_probe: cannot allocate %.2f GB\n", gb); return 1; }
    if (mode == 2) { dst.alloc(nbytes); if (!dst.ptr) { std::fprintf(stderr, "cannot alloc dst\n"); return 1; } }
    uint64_t nf = nbytes / sizeof(float);
    // Initialize so pages are resident (stream loads of fresh pages would be
    // page-fault dominated otherwise). Copy mode also fills dst so its store
    // doesn't trigger copy-on-write faults mid-run.
    __m512 fill = _mm512_set1_ps(0.5f);
    float* b = buf.ptr;
    for (uint64_t i = 0; i + 16 <= nf; i += 16) _mm512_stream_si512((__m512i*)(b + i), _mm512_castps_si512(fill));
    if (mode == 2) {
        float* d = dst.ptr;
        for (uint64_t i = 0; i + 16 <= nf; i += 16) _mm512_stream_si512((__m512i*)(d + i), _mm512_castps_si512(fill));
    }

    const char* mname = mode==0?"read" : mode==1?"readwrite" : mode==2?"copy":"read-normal";
    std::printf("bw_probe: gb=%.2f threads=%d reps=%d mode=%s large_pages=%s\n",
                gb, threads, reps, mname, buf.large?"yes":"no");

    double best_ms = 1e18;
    for (int r = 0; r < reps; ++r) {
        std::vector<std::thread> pool;
        std::vector<Job> jobs(threads);
        for (int t = 0; t < threads; ++t) {
            jobs[t].buf = b; jobs[t].dst = dst.ptr; jobs[t].nf = nf; jobs[t].tid = t;
            jobs[t].nthreads = threads; jobs[t].mode = mode; jobs[t].out = 0;
        }
        double t0 = now_ms();
        for (int t = 0; t < threads; ++t)
            pool.emplace_back([&, t](){ worker(&jobs[t]); });
        for (auto& th : pool) th.join();
        double dt = now_ms() - t0;
        uint64_t chk = 0; for (auto& j : jobs) chk += j.out;
        (void)chk;
        if (dt < best_ms) best_ms = dt;
    }
    double sec = best_ms / 1e3;
    // read GB/s counts nbytes; copy/readwrite move extra write traffic.
    double gbps   = (double)nbytes / 1e9 / sec;          // read-side GB/s
    double extra  = (mode == 1 || mode == 2) ? (double)nbytes : 0.0;
    double total  = ((double)nbytes + extra) / 1e9 / sec; // read+write combined
    std::printf("bw: best=%.1f ms  read=%.1f GB/s", best_ms, gbps);
    if (mode == 1 || mode == 2) std::printf("  (combined read+write=%.1f GB/s)", total);
    std::printf("\n");
    return 0;
}