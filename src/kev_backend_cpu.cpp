// kev_backend_cpu.cpp — CPU backend plugin implementing the portable C ABI.
//
// This module implements the kev_backend C ABI from kev_backend.h over the
// vendored llama.cpp / ggml. It is the ONLY module that is allowed to link
// llama statically when running as a plugin dll ("single-llama rule").
//
// Two build flavors are supported (see CMakeLists / docs):
//   * kev-backend-cpu.dll  — shared library loaded by kev::BackendFactory via
//                            LoadLibrary/dlopen (production / portable path). Only
//                            this dll links llama; the host executable must NOT.
//   * static in-process     — linked directly into main_server/bench targets used
//                            for bit-identical validation on a stable box today.
//                            main_server and this TU then share the SAME llama
//                            copy (single module), which is the current safe state.
//
// In multi-variant builds (GGML_CPU_ALL_VARIANTS + GGML_BACKEND_DL) the exe dir
// contains ggml-cpu-<tag>.dll variants. At init we enumerate them, read each
// one's runtime score (cpuid capability check), and probe candidates from the
// highest score down: each candidate runs a small known-answer compute (F32
// GEMM + Q8_0 x F32 GEMM fingerprint) in a probe subprocess, and the result is
// compared against the scalar x64 baseline variant. The first candidate whose
// fingerprint matches the baseline is pre-registered via ggml_backend_load()
// before llama_backend_init(), which then skips ggml_backend_load_all() (it
// only runs when the registry is empty), so a numerically broken variant is
// never loaded. KEV_CPU_VARIANT=off disables the auto selection,
// KEV_CPU_VARIANT=<tag> forces a variant (still KAT-gated). This gates the
// VNNI/dpbusd variants that are numerically WRONG on this toolchain/model
// (see docs/portable_backend_deployment.md) while keeping faster correct
// variants available.
//
// This file must compile bit-identically to the legacy single-module path:
// the ops below call the same llama_* functions kev_model.cpp used to call
// directly, so decode semantics are preserved exactly (same llama underneath).
#include "kev_backend.h"

#include "kev_backend_moe.h"
#include "kev_chat_data.h"

#include "llama.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

namespace {

struct KevCpuBackend {
    llama_model          * model = nullptr;
    llama_context        * ctx   = nullptr;
    llama_adapter_lora   * lora  = nullptr;
    const llama_vocab    * vocab = nullptr;
    int32_t               n_embd = 0;
    int32_t               n_seq_max = 0;
    bool                  ok = false;
    // MoE expert offload: tensor_buft_overrides 的生命期必须覆盖 model 生命周期。
    std::vector<llama_model_tensor_buft_override> moe_overrides_;
};

KevCpuBackend * as_cpu(kev_backend_t * b) {
    return reinterpret_cast<KevCpuBackend *>(b);
}

void write_err(char * err, size_t cap, const std::string & msg) {
    if (err && cap) std::snprintf(err, cap, "%s", msg.c_str());
}

// llama.cpp/ggml log bridge: only ERROR passes through to stderr (filters the
// INFO/WARN metadata/backend dumps that main_server used to suppress via
// llama_log_set before — now handled inside the llama-linking backend dll, since
// the host is llama-free).
static void kev_cpu_log_filter(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (level >= GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

// ---- runtime CPU variant selection (multi-variant builds, Windows) ----
// See the file header for the design. Everything below is Windows-only and
// no-ops when no ggml-cpu-*.dll variants exist next to the executable.

#ifdef _WIN32

struct KatMetrics {
    double c1_l2 = 0.0, c2_l2 = 0.0;
    double c1_max = 0.0, c2_max = 0.0;
    double c1_e[8] = {}, c2_e[8] = {};
    bool   valid = false;
};

static float kat_half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u;
    const uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    const uint32_t frac = (uint32_t)h & 0x3FFu;
    float v;
    if (exp == 0) {
        v = (float)frac * (1.0f / 16777216.0f);          // frac * 2^-24
    } else if (exp == 31) {
        v = frac ? NAN : INFINITY;
    } else {
        v = std::ldexp((float)(frac | 0x400u), (int)exp - 25);
    }
    return sign ? -v : v;
}

// Runs the KAT on the variant registered in THIS process (the probe
// subprocess) and emits machine-readable metrics on stdout. Correctness is
// judged by the parent against the x64 baseline probe, not here.
static int kat_run_and_report() {
    // Deterministic inputs (LCG), exercising both the plain F32 GEMM and the
    // Q8_0 x F32 mul_mat path used by Q8_0 model weights.
    // K must be a multiple of 256 (tiled GEMM path constraint, ne00 % 256).
    const int M = 128, K = 256, N = 128;
    std::vector<float> A((size_t)M * K), B((size_t)K * N);
    uint32_t seed = 0x12345678u;
    auto lcg = [&seed]() { seed = seed * 1103515245u + 12345u; return (float)((int)((seed >> 16) & 0x7FFF) - 16384) / 16384.0f; };
    for (auto & v : A) v = lcg();
    for (auto & v : B) v = lcg();

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) { std::fprintf(stdout, "KAT no-cpu-device\n"); std::fflush(stdout); return 1; }
    ggml_backend_t bk = ggml_backend_dev_init(dev, nullptr);
    if (!bk)    { std::fprintf(stdout, "KAT backend-init-failed\n"); std::fflush(stdout); return 1; }

    struct ggml_init_params ip = { 8u << 20, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  K, M);
    ggml_tensor * b  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  K, N);
    ggml_tensor * aq = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, M);
    ggml_tensor * c1 = ggml_mul_mat(ctx, a,  b);
    ggml_tensor * c2 = ggml_mul_mat(ctx, aq, b);
    ggml_cgraph * g  = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, c1);
    ggml_build_forward_expand(g, c2);

    ggml_backend_buffer * buf = ggml_backend_alloc_ctx_tensors(ctx, bk);
    if (!buf) { ggml_free(ctx); ggml_backend_free(bk); std::fprintf(stdout, "KAT alloc-failed\n"); std::fflush(stdout); return 1; }

    ggml_backend_tensor_set(a, A.data(), 0, A.size() * sizeof(float));
    ggml_backend_tensor_set(b, B.data(), 0, B.size() * sizeof(float));
    std::vector<uint8_t> qbuf(ggml_nbytes(aq));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, A.data(), qbuf.data(), 0, M, K, nullptr);
    ggml_backend_tensor_set(aq, qbuf.data(), 0, qbuf.size());

    if (ggml_backend_graph_compute(bk, g) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(bk);
        std::fprintf(stdout, "KAT compute-failed\n"); std::fflush(stdout); return 1;
    }

    std::vector<float> r1((size_t)N * M), r2((size_t)N * M);
    ggml_backend_tensor_get(c1, r1.data(), 0, r1.size() * sizeof(float));
    ggml_backend_tensor_get(c2, r2.data(), 0, r2.size() * sizeof(float));

    KatMetrics m;
    auto acc = [](const std::vector<float> & r, double & l2, double & mx, double e[8]) {
        l2 = 0.0; mx = 0.0;
        for (float v : r) { double d = (double)v; l2 += d * d; mx = std::max(mx, std::fabs(d)); }
        l2 = std::sqrt(l2);
        const size_t n = r.size();
        for (int i = 0; i < 8; ++i) e[i] = (double)r[(size_t)((n * (size_t)i) / 8 + 17) % n];
    };
    acc(r1, m.c1_l2, m.c1_max, m.c1_e);
    acc(r2, m.c2_l2, m.c2_max, m.c2_e);
    m.valid = std::isfinite(m.c1_l2) && std::isfinite(m.c2_l2);

    std::fprintf(stdout, "KAT_BEGIN\n");
    std::fprintf(stdout, "c1_l2 %.9f\nc2_l2 %.9f\nc1_max %.9f\nc2_max %.9f\n",
                 m.c1_l2, m.c2_l2, m.c1_max, m.c2_max);
    std::fprintf(stdout, "c1_e");
    for (int i = 0; i < 8; ++i) std::fprintf(stdout, " %.9f", m.c1_e[i]);
    std::fprintf(stdout, "\nc2_e");
    for (int i = 0; i < 8; ++i) std::fprintf(stdout, " %.9f", m.c2_e[i]);
    std::fprintf(stdout, "\nKAT_END\n");
    std::fflush(stdout);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(bk);
    return m.valid ? 0 : 1;
}

static std::wstring exe_path_w() {
    wchar_t path[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    return path;
}

static std::string wide_to_utf8(const std::wstring & w) {
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string out((size_t)n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), out.data(), n, nullptr, nullptr);
    return out;
}

struct VariantCandidate {
    std::wstring path;
    std::string  name;   // tag between "ggml-cpu-" and ".dll"
    int          score = 0;
};

// Reads the variant's runtime score (cpuid capability probe) without
// registering it. Score 0 => this CPU cannot run the variant.
static int variant_score(const std::wstring & dll_path) {
    HMODULE h = LoadLibraryW(dll_path.c_str());
    if (!h) return -1;
    using score_fn_t = int (*)();
    auto fn = (score_fn_t)GetProcAddress(h, "ggml_backend_score");
    int s = fn ? fn() : -1;
    FreeLibrary(h);
    return s;
}

// Spawns `<host-exe> --kev-probe-cpu-variant <dll>` (see main_server.cpp),
// captures its stdout and exit code. The probe loads the variant in a fresh
// process and prints the KAT metrics, so no unloading is ever needed.
static bool probe_variant_subprocess(const std::wstring & dll_path, int & exit_code,
                                     std::string & stdout_text) {
    std::wstring cmd = L"\"" + exe_path_w() + L"\" --kev-probe-cpu-variant \"" + dll_path + L"\"";
    SECURITY_ATTRIBUTES sa = { sizeof(sa), nullptr, TRUE };
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) return false;
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW si = {};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = wr;
    si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
    si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi = {};
    std::wstring mutable_cmd = cmd;
    BOOL ok = CreateProcessW(nullptr, mutable_cmd.data(), nullptr, nullptr, TRUE, 0,
                             nullptr, nullptr, &si, &pi);
    CloseHandle(wr);
    if (!ok) { CloseHandle(rd); return false; }

    char buf[4096];
    DWORD got = 0;
    while (ReadFile(rd, buf, sizeof(buf), &got, nullptr) && got > 0) {
        stdout_text.append(buf, buf + got);
        if (stdout_text.size() > (1u << 20)) break;
    }
    CloseHandle(rd);
    if (WaitForSingleObject(pi.hProcess, 120000) != WAIT_OBJECT_0) {
        TerminateProcess(pi.hProcess, 3);
        WaitForSingleObject(pi.hProcess, 5000);
    }
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    exit_code = (int)code;
    return true;
}

static bool parse_kat_metrics(const std::string & text, KatMetrics & m) {
    auto find_val = [&](const char * key, double & out) {
        std::string pat = std::string(key) + " ";
        size_t p = text.find(pat);
        if (p == std::string::npos) return false;
        out = std::atof(text.c_str() + p + pat.size());
        return std::isfinite(out);
    };
    bool ok = true;
    ok = ok && find_val("c1_l2", m.c1_l2);
    ok = ok && find_val("c2_l2", m.c2_l2);
    ok = ok && find_val("c1_max", m.c1_max);
    ok = ok && find_val("c2_max", m.c2_max);
    if (!ok) return false;
    size_t p = text.find("c1_e");
    if (p == std::string::npos) return false;
    p += 4;
    for (int i = 0; i < 8; ++i) { p = text.find(' ', p); if (p == std::string::npos) return false; m.c1_e[i] = std::atof(text.c_str() + p + 1); }
    p = text.find("c2_e");
    if (p == std::string::npos) return false;
    p += 4;
    for (int i = 0; i < 8; ++i) { p = text.find(' ', p); if (p == std::string::npos) return false; m.c2_e[i] = std::atof(text.c_str() + p + 1); }
    m.valid = true;
    return true;
}

static bool kat_metrics_match(const KatMetrics & ref, const KatMetrics & cand) {
    // Pass if within 1e-3 relative OR 1e-3 absolute (sampled elements can sit
    // near zero where pure relative comparison over-penalizes benign
    // reduction-order noise; the VNNI failure moves c2 by ~3x, far above both).
    auto close = [](double a, double b) {
        const double d = std::fabs(a - b), s = std::max(std::fabs(a), std::fabs(b));
        return d / std::max(s, 1e-9) <= 1e-3 || d <= 1e-3;
    };
    if (!ref.valid || !cand.valid) return false;
    if (!close(ref.c1_l2, cand.c1_l2)) return false;
    if (!close(ref.c2_l2, cand.c2_l2)) return false;
    if (!close(ref.c1_max, cand.c1_max)) return false;
    if (!close(ref.c2_max, cand.c2_max)) return false;
    for (int i = 0; i < 8; ++i) {
        if (!close(ref.c1_e[i], cand.c1_e[i])) return false;
        if (!close(ref.c2_e[i], cand.c2_e[i])) return false;
    }
    return true;
}

// One-shot per process: picks the best KAT-verified variant and pre-registers
// it so llama_backend_init() skips ggml_backend_load_all().
static bool select_cpu_variant(std::string & err_out) {
    static std::atomic<bool> done{false};
    static std::atomic<bool> ok{false};
    if (done.exchange(true)) { if (!ok) err_out = "cpu variant selection failed earlier"; return ok; }

    const char * forced = std::getenv("KEV_CPU_VARIANT");
    const std::string forced_s = forced ? forced : "";
    if (forced_s == "off") {
        std::fprintf(stderr, "[kev] cpu-variant: KEV_CPU_VARIANT=off, skipping auto selection\n");
        ok = true;
        return true;
    }

    const std::wstring exe = exe_path_w();
    const size_t slash = exe.find_last_of(L"/\\");
    const std::wstring dir = slash == std::wstring::npos ? L"." : exe.substr(0, slash);

    WIN32_FIND_DATAW fd = {};
    HANDLE h = FindFirstFileW((dir + L"\\ggml-cpu-*.dll").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) { ok = true; return true; }   // no variants: plain build
    std::vector<VariantCandidate> cands;
    do {
        VariantCandidate c;
        c.path = dir + L"\\" + fd.cFileName;
        std::wstring wname(fd.cFileName);
        const size_t pre = wname.find(L"ggml-cpu-");
        const size_t ext = wname.rfind(L".dll");
        if (pre == std::string::npos || ext == std::string::npos || ext <= pre + 9) continue;
        c.name = wide_to_utf8(wname.substr(pre + 9, ext - pre - 9));
        c.score = variant_score(c.path);
        if (c.score > 0) cands.push_back(std::move(c));
    } while (FindNextFileW(h, &fd));
    FindClose(h);

    if (cands.empty()) { ok = true; return true; }                // nothing usable here; let ggml default

    std::sort(cands.begin(), cands.end(), [](const VariantCandidate & x, const VariantCandidate & y) {
        if (x.score != y.score) return x.score > y.score;
        return x.name < y.name;
    });
    std::fprintf(stderr, "[kev] cpu-variant: candidates:");
    for (const auto & c : cands) std::fprintf(stderr, " %s(%d)", c.name.c_str(), c.score);
    std::fprintf(stderr, "\n");

    // Reference fingerprint: the scalar x64 baseline variant.
    KatMetrics ref;
    bool have_ref = false;
    auto it_x64 = std::find_if(cands.begin(), cands.end(),
                               [](const VariantCandidate & c) { return c.name == "x64"; });
    if (it_x64 != cands.end()) {
        int code = -1; std::string out;
        if (probe_variant_subprocess(it_x64->path, code, out) && code == 0 && parse_kat_metrics(out, ref)) {
            have_ref = true;
            std::fprintf(stderr, "[kev] cpu-variant: reference x64 KAT ok (c1_l2=%.6f c2_l2=%.6f)\n", ref.c1_l2, ref.c2_l2);
        }
    }
    if (!have_ref) {
        err_out = "cpu-variant auto requires a working 'x64' baseline variant for the reference fingerprint";
        std::fprintf(stderr, "[kev] cpu-variant: %s\n", err_out.c_str());
        return false;
    }

    auto probe_one = [&](const VariantCandidate & c, KatMetrics & m) {
        int code = -1; std::string out;
        if (!probe_variant_subprocess(c.path, code, out)) {
            std::fprintf(stderr, "[kev] cpu-variant: %s probe spawn failed\n", c.name.c_str());
            return false;
        }
        if (code != 0 || !parse_kat_metrics(out, m)) {
            std::fprintf(stderr, "[kev] cpu-variant: %s KAT RUN FAILED (exit=%d)\n", c.name.c_str(), code);
            return false;
        }
        if (!kat_metrics_match(ref, m)) {
            std::fprintf(stderr, "[kev] cpu-variant: %s KAT MISMATCH (c1_l2=%.6f c2_l2=%.6f vs ref %.6f/%.6f) - rejected\n",
                         c.name.c_str(), m.c1_l2, m.c2_l2, ref.c1_l2, ref.c2_l2);
            return false;
        }
        return true;
    };

    const VariantCandidate * chosen = nullptr;
    if (!forced_s.empty() && forced_s != "auto" && forced_s != "off") {
        auto it = std::find_if(cands.begin(), cands.end(),
                               [&](const VariantCandidate & c) { return c.name == forced_s; });
        if (it == cands.end()) {
            err_out = "KEV_CPU_VARIANT='" + forced_s + "': no such cpu variant with score>0";
            std::fprintf(stderr, "[kev] cpu-variant: %s\n", err_out.c_str());
            return false;
        }
        KatMetrics m;
        if (!probe_one(*it, m)) {
            err_out = "KEV_CPU_VARIANT='" + forced_s + "' failed the KAT against the x64 reference";
            return false;
        }
        chosen = &*it;
    } else {
        for (const auto & c : cands) {
            KatMetrics m;
            if (probe_one(c, m)) { chosen = &c; break; }
        }
        if (!chosen) {
            err_out = "all cpu variants failed the KAT against the x64 reference";
            std::fprintf(stderr, "[kev] cpu-variant: %s\n", err_out.c_str());
            return false;
        }
    }

    std::fprintf(stderr, "[kev] cpu-variant: selected '%s' (score=%d, KAT-verified vs x64)\n",
                 chosen->name.c_str(), chosen->score);
    ggml_backend_reg_t reg = ggml_backend_load(wide_to_utf8(chosen->path).c_str());
    if (!reg) {
        err_out = "ggml_backend_load failed for cpu variant '" + chosen->name + "'";
        std::fprintf(stderr, "[kev] cpu-variant: %s\n", err_out.c_str());
        return false;
    }
    ok = true;
    return true;
}

#endif // _WIN32

// ---- lifecycle ----

int op_create(kev_backend_t ** out) {
    if (!out) return 1;
    *out = reinterpret_cast<kev_backend_t *>(new KevCpuBackend());
    return 0;
}

void op_destroy(kev_backend_t * b) {
    KevCpuBackend * kb = as_cpu(b);
    if (!kb) return;
    if (kb->lora)  { llama_adapter_lora_free(kb->lora);  kb->lora  = nullptr; }
    if (kb->ctx)   { llama_free(kb->ctx);                kb->ctx   = nullptr; }
    if (kb->model) { llama_model_free(kb->model);        kb->model = nullptr; }
    if (kb->ctx || kb->model) llama_backend_free();
    kb->vocab = nullptr; kb->ok = false;
    delete kb;
}

int op_init(kev_backend_t * b, const kev_backend_params * p, char * err, size_t err_cap) {
    KevCpuBackend * kb = as_cpu(b);
    if (!kb) { write_err(err, err_cap, "null backend"); return 1; }
    if (kb->ctx || kb->model) { write_err(err, err_cap, "backend already initialized"); return 1; }
    if (!p || !p->model_path) { write_err(err, err_cap, "model_path required"); return 1; }

#ifdef _WIN32
    // Multi-variant builds: pick + pre-register a KAT-verified ggml-cpu variant
    // BEFORE llama_backend_init() so its load_all() (registry-empty shortcut)
    // never sees the broken variants. No-op in plain single-variant builds.
    {
        std::string sel_err;
        if (!select_cpu_variant(sel_err)) { write_err(err, err_cap, sel_err); return 1; }
    }
#endif

    llama_backend_init();

    // Host is llama-free; the log filter that used to live in main_server is now
    // installed here inside the single-llama module (must be set before any model
    // action, mirroring the old main_server llama_log_set placement).
    llama_log_set(kev_cpu_log_filter, nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = p->n_gpu_layers;   // 默认 0 = 全 CPU(legacy); 负数 = 全 GPU
    // MoE 专家 offload: moe_offload 且走 GPU 时, 把专家张量强制留 host RAM(省显存),
    // 其余按 n_gpu_layers 进设备。纯 CPU(n_gpu_layers=0)时为空, 行为与 legacy 完全一致。
    kb->moe_overrides_ = kev_backend_moe::build_moe_overrides(p->moe_offload, p->n_gpu_layers, p->model_path);
    if (!kb->moe_overrides_.empty()) {
        mparams.tensor_buft_overrides = kb->moe_overrides_.data();
    }
    kb->model = llama_model_load_from_file(p->model_path, mparams);
    if (!kb->model) { write_err(err, err_cap, "failed to load model: " + std::string(p->model_path)); return 1; }
    kb->vocab = llama_model_get_vocab(kb->model);
    kb->n_embd = llama_model_n_embd(kb->model);

    const int n_ctx   = p->n_ctx > 0 ? p->n_ctx : 2048;
    kb->n_seq_max     = p->n_seq_max > 0 ? p->n_seq_max : 4;
    const int threads = p->n_threads > 0 ? p->n_threads : (int) std::thread::hardware_concurrency();

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = (uint32_t) n_ctx;
    cparams.n_batch           = (uint32_t) (p->n_batch > 0 ? p->n_batch : 512);
    cparams.n_ubatch          = (uint32_t) (p->n_ubatch > 0 ? p->n_ubatch : 256);
    cparams.n_seq_max         = (uint32_t) kb->n_seq_max;
    cparams.n_outputs_max_per_seq = (uint32_t) (p->n_outputs_max_per_seq > 0 ? p->n_outputs_max_per_seq : 128);
    cparams.embeddings        = p->embeddings != 0;
    cparams.lm_head           = p->lm_head != 0;
    cparams.pooling_type      = LLAMA_POOLING_TYPE_NONE;
    cparams.n_threads         = (int32_t) threads;
    cparams.n_threads_batch   = (int32_t) threads;
    kb->ctx = llama_init_from_model(kb->model, cparams);
    if (!kb->ctx) { write_err(err, err_cap, "failed to init context"); return 1; }

    // Optional LoRA adapter (only when not already folded into the weights).
    if (p->lora_path && p->lora_path[0] != '\0' && !p->lora_premerged) {
        kb->lora = llama_adapter_lora_init(kb->model, p->lora_path);
        if (!kb->lora) { write_err(err, err_cap, "failed to load lora adapter: " + std::string(p->lora_path)); return 1; }
        float scale = p->lora_scale;
        if (llama_set_adapters_lora(kb->ctx, &kb->lora, 1, &scale) != 0) {
            write_err(err, err_cap, "failed to set lora adapters"); return 1;
        }
    }

    kb->ok = true;
    return 0;
}

// ---- decode ----

int op_decode_seq(kev_backend_t * b, const kev_backend_decode_seq * d, char * err, size_t err_cap) {
    KevCpuBackend * kb = as_cpu(b);
    if (!kb || !kb->ctx) { write_err(err, err_cap, "backend not initialized"); return 1; }
    if (!d || d->n_tokens <= 0) { write_err(err, err_cap, "empty sequence to decode"); return 1; }
    const int32_t n = d->n_tokens;

    llama_batch batch = llama_batch_init(n, /*embd=*/0, kb->n_seq_max);
    for (int32_t i = 0; i < n; ++i) {
        batch.token[i]     = (llama_token) d->tokens[i];
        batch.pos[i]       = (llama_pos)(d->start_pos + i);
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = d->seq;
        batch.logits[i]    = 0;
    }
    for (int32_t k = 0; k < d->n_logits_pos; ++k) {
        const int32_t pos = d->logits_pos[k];
        if (pos >= 0 && pos < n) batch.logits[pos] = 1;
    }
    batch.n_tokens = n;

    const int r = llama_decode(kb->ctx, batch);
    llama_batch_free(batch);
    if (r != 0) { write_err(err, err_cap, "decode failed ret=" + std::to_string(r)); return 1; }
    return 0;
}

int op_decode_batch(kev_backend_t * b, const kev_backend_decode_branch * rows, int32_t n_rows,
                    char * err, size_t err_cap) {
    KevCpuBackend * kb = as_cpu(b);
    if (!kb || !kb->ctx) { write_err(err, err_cap, "backend not initialized"); return 1; }
    if (!rows || n_rows <= 0) { write_err(err, err_cap, "empty batch"); return 1; }
    const int32_t L = rows[0].n_ids;
    if (L <= 0) { write_err(err, err_cap, "empty branch in batch"); return 1; }
    for (int32_t s = 0; s < n_rows; ++s) {
        if (rows[s].n_ids != L) { write_err(err, err_cap, "batch requires equal branch lengths"); return 1; }
    }

    const int32_t n = L * n_rows;
    llama_batch batch = llama_batch_init(n, /*embd=*/0, kb->n_seq_max);
    for (int32_t s = 0; s < n_rows; ++s) {
        const kev_backend_decode_branch & br = rows[s];
        const int32_t base = s * L;
        for (int32_t t = 0; t < L; ++t) {
            const int32_t i = base + t;
            batch.token[i]     = (llama_token) br.ids[t];
            batch.pos[i]       = (llama_pos)(br.start_pos + t);
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = br.seq;
            batch.logits[i]    = 0;
        }
        for (int32_t k = 0; k < br.n_opts; ++k) {
            const int32_t o = br.opts[k];
            if (o >= 0 && o < L) batch.logits[base + o] = 1;
        }
        if (br.decide >= 0 && br.decide < L) batch.logits[base + br.decide] = 1;
    }
    batch.n_tokens = n;

    const int r = llama_decode(kb->ctx, batch);
    llama_batch_free(batch);
    if (r != 0) { write_err(err, err_cap, "batch decode failed ret=" + std::to_string(r)); return 1; }
    return 0;
}

int op_read_hidden(kev_backend_t * b, int32_t out_idx, float * out, size_t out_cap,
                   char * err, size_t err_cap) {
    KevCpuBackend * kb = as_cpu(b);
    if (!kb || !kb->ctx) { write_err(err, err_cap, "backend not initialized"); return 1; }
    float * p = llama_get_embeddings_ith(kb->ctx, out_idx);
    if (!p) { write_err(err, err_cap, "no embedding at output idx " + std::to_string(out_idx)); return 1; }
    if (out_cap < (size_t) kb->n_embd) { write_err(err, err_cap, "output buffer too small"); return 1; }
    std::memcpy(out, p, (size_t) kb->n_embd * sizeof(float));
    return 0;
}

// B1: decode multiple independent sequences (each its own KV slot) in ONE
// llama_decode. seq-major token layout; each seq's positions are counted relative
// to its own start_pos (independent M-RoPE). For every requested output
// (out_seq[i], out_pos[i]) we mark llama_batch.logits[row]=1 and fill out_idx[i]
// with the ordinal of that marked row in llama_get_logits_ith order (batch-row
// major among all marked logits). This lets the host pass out_idx[] straight into
// read_logits().
int op_decode_multi_seq(kev_backend_t * b, const int32_t * tokens,
                        const int32_t * n_tokens, const int32_t * start_pos,
                        const int32_t * seq, int32_t n_seq,
                        const int32_t * out_seq, const int32_t * out_pos, int32_t n_out,
                        int32_t * out_idx, int32_t out_idx_cap,
                        char * err, size_t err_cap) {
    KevCpuBackend * kb = as_cpu(b);
    if (!kb || !kb->ctx) { write_err(err, err_cap, "backend not initialized"); return 1; }
    if (n_seq <= 0 || !tokens || !n_tokens || !start_pos || !seq) { write_err(err, err_cap, "invalid multi-seq request"); return 1; }
    if (n_out > 0 && (!out_seq || !out_pos)) { write_err(err, err_cap, "invalid output request"); return 1; }
    if (out_idx_cap < n_out) { write_err(err, err_cap, "out_idx buffer too small"); return 1; }

    // Validate seq ids are distinct and within n_seq_max.
    std::vector<int32_t> seen(kb->n_seq_max, 0);
    for (int32_t s = 0; s < n_seq; ++s) {
        if (seq[s] < 0 || seq[s] >= kb->n_seq_max) { write_err(err, err_cap, "seq id out of range"); return 1; }
        if (seen[seq[s]]) { write_err(err, err_cap, "duplicate seq id in multi-seq batch"); return 1; }
        seen[seq[s]] = 1;
    }

    // Total tokens; must not exceed n_batch (llama asserts n_tokens_all <= n_batch).
    int64_t total = 0;
    std::vector<int32_t> base(n_seq);
    for (int32_t s = 0; s < n_seq; ++s) {
        base[s] = (int32_t) total;
        if (n_tokens[s] <= 0) { write_err(err, err_cap, "empty sequence in multi-seq batch"); return 1; }
        total += n_tokens[s];
    }
    const int32_t nb = (int32_t) llama_n_batch(kb->ctx);
    if (total > nb) { write_err(err, err_cap, "multi-seq batch too large: " + std::to_string(total) + " > n_batch " + std::to_string(nb)); return 1; }

    llama_batch batch = llama_batch_init((int32_t) total, /*embd=*/0, kb->n_seq_max);
    for (int32_t s = 0; s < n_seq; ++s) {
        const int32_t n = n_tokens[s];
        for (int32_t t = 0; t < n; ++t) {
            const int32_t row = base[s] + t;
            batch.token[row]     = (llama_token) tokens[row];
            batch.pos[row]       = (llama_pos)(start_pos[s] + t);
            batch.n_seq_id[row]  = 1;
            batch.seq_id[row][0] = (llama_seq_id) seq[s];
            batch.logits[row]    = 0;
        }
    }
    for (int32_t i = 0; i < n_out; ++i) {
        const int32_t s = out_seq[i];
        const int32_t p = out_pos[i];
        if (s < 0 || s >= n_seq || p < 0 || p >= n_tokens[s]) { write_err(err, err_cap, "output position out of range"); llama_batch_free(batch); return 1; }
        const int32_t row = base[s] + p;
        batch.logits[row] = 1;
    }
    batch.n_tokens = (int32_t) total;

    const int r = llama_decode(kb->ctx, batch);
    if (r != 0) { write_err(err, err_cap, "multi-seq decode failed ret=" + std::to_string(r)); llama_batch_free(batch); return 1; }

    // The host reads each requested output via read_logits(out_idx, token, &lt), whose
    // out_idx is passed straight to llama_get_logits_ith(ctx, out_idx). get_logits_ith
    // treats its argument as a BATCH ROW index and translates it to the contiguous logits
    // row via ctx->output_ids[row] (= ordinal among all llama_batch.logits!=0 rows, in
    // batch order). So we must hand back the batch row base[s]+pos for each output.
    // (The existing single-seq path does the same: it hands back the local position,
    // which equals the batch row because that decode starts at batch row 0.)
    for (int32_t i = 0; i < n_out; ++i) {
        const int32_t s = out_seq[i];
        out_idx[i] = base[s] + out_pos[i];
    }
    llama_batch_free(batch);
    return 0;
}

int op_read_logits(kev_backend_t * b, int32_t out_idx, int32_t token_id, float * out) {
    KevCpuBackend * kb = as_cpu(b);
    if (!kb || !kb->ctx || !out || !kb->vocab) return 1;
    // Requires lm_head enabled (kev_backend_params.lm_head) so the logits buffer
    // is allocated; otherwise llama_get_logits_ith returns nullptr (Release).
    float * lt = llama_get_logits_ith(kb->ctx, out_idx);
    if (!lt) return 1;   // position not marked for output / logits disabled
    const int nv = llama_vocab_n_tokens(kb->vocab);
    if (token_id < 0 || token_id >= nv) return 1;
    *out = lt[token_id];
    return 0;
}

int32_t op_vocab_n_tokens(kev_backend_t * b) {
    KevCpuBackend * kb = as_cpu(b);
    return (kb && kb->vocab) ? (int32_t) llama_vocab_n_tokens(kb->vocab) : 0;
}

int op_seq_clear(kev_backend_t * b, int32_t seq) {
    KevCpuBackend * kb = as_cpu(b);
    if (!kb || !kb->ctx) return 1;
    llama_memory_t mem = llama_get_memory(kb->ctx);
    llama_memory_seq_rm(mem, seq, 0, -1);
    return 0;
}

int op_seq_copy(kev_backend_t * b, int32_t src, int32_t dst) {
    KevCpuBackend * kb = as_cpu(b);
    if (!kb || !kb->ctx) return 1;
    llama_memory_t mem = llama_get_memory(kb->ctx);
    llama_memory_seq_cp(mem, src, dst, 0, -1);
    return 0;
}

// ---- sizing ----

int32_t op_n_ctx_seq_capacity(kev_backend_t * b) {
    KevCpuBackend * kb = as_cpu(b);
    return (kb && kb->ctx) ? (int32_t) llama_n_ctx_seq(kb->ctx) : 0;
}

// Logical max tokens per llama_decode call (cparams.n_batch). Used by KevModel to
// cap batch groups and chunk long sequences so no single llama_decode exceeds it
// (avoids llama.cpp's n_tokens_all <= n_batch GGML_ASSERT crash).
int32_t op_n_batch(kev_backend_t * b) {
    KevCpuBackend * kb = as_cpu(b);
    return (kb && kb->ctx) ? (int32_t) llama_n_batch(kb->ctx) : 0;
}

// Plain llama tokenizer count (no kev special-token rewrite).
// NOTE: KevModel today counts billing tokens in-process via kev::tokenize(vocab_)
// (which applies the kev <|x|> -> <¦x¦> rewrite). This op is the backend-side
// utility for the fully-plugged (single-llama) state; the current in-process
// static build keeps KevModel's count_tokens() path untouched so behaviour is
// bit-identical to the baseline. See docs/portable_backend_deployment.md.
int32_t op_count_tokens(kev_backend_t * b, const char * text) {
    KevCpuBackend * kb = as_cpu(b);
    if (!kb || !kb->vocab || !text) return 0;
    std::vector<llama_token> tokens(4096);
    const int32_t n = llama_tokenize(kb->vocab, text, (int32_t) std::strlen(text),
                                     tokens.data(), (int32_t) tokens.size(),
                                     /*add_special=*/false, /*parse_special=*/false);
    return n < 0 ? 0 : n;
}

int32_t op_model_n_embd(kev_backend_t * b) {
    KevCpuBackend * kb = as_cpu(b);
    return kb ? kb->n_embd : 0;
}

int32_t op_threads(kev_backend_t * b) {
    KevCpuBackend * kb = as_cpu(b);
    return (kb && kb->ctx) ? llama_n_threads(kb->ctx) : 0;
}

int32_t op_threads_batch(kev_backend_t * b) {
    KevCpuBackend * kb = as_cpu(b);
    return (kb && kb->ctx) ? llama_n_threads_batch(kb->ctx) : 0;
}

kev_backend_vocab_t * op_vocab_opaque(kev_backend_t * b) {
    KevCpuBackend * kb = as_cpu(b);
    return kb ? reinterpret_cast<kev_backend_vocab_t *>(const_cast<llama_vocab *>(kb->vocab)) : nullptr;
}

// ---- delegated tokenizer (B-1/B-2) ----
// The host is llama-free (only this dll links llama), so tokenize / special-token
// lookup must be served here. These reproduce the exact llama_tokenize semantics
// kev_encode.cpp used to invoke in-process, so encode is bit-identical across the
// in-process (bench/m3/m5/...) and dll-delegated (main_server) paths.

int32_t op_tokenize(kev_backend_t * b, const char * text, int32_t text_len,
                    bool add_special, bool parse_special,
                    int32_t * out_ids, int32_t cap, char * err, size_t err_cap) {
    KevCpuBackend * kb = as_cpu(b);
    if (!kb || !kb->vocab || !text) { write_err(err, err_cap, "tokenize: no vocab"); return -1; }
    if (text_len < 0) text_len = (int32_t) std::strlen(text);
    if (!out_ids || cap <= 0) return -1;
    std::vector<llama_token> buf((size_t) cap);
    const int32_t n = llama_tokenize(kb->vocab, text, text_len,
                                     buf.data(), (int32_t) buf.size(),
                                     add_special, parse_special);
    if (n < 0) return n;   // llama returns -required-capacity (negative)
    for (int32_t i = 0; i < n; ++i) out_ids[i] = (int32_t) buf[i];
    return n;
}

int32_t op_special_token_id(kev_backend_t * b, const char * special) {
    KevCpuBackend * kb = as_cpu(b);
    if (!kb || !kb->vocab || !special) return -1;
    const int len = (int) std::strlen(special);
    std::vector<llama_token> buf((size_t) len * 4 + 16);
    int32_t n = llama_tokenize(kb->vocab, special, len,
                               buf.data(), (int32_t) buf.size(),
                               /*add_special=*/false, /*parse_special=*/true);
    if (n < 0) {
        buf.resize((size_t) -n);
        n = llama_tokenize(kb->vocab, special, len, buf.data(), (int32_t) buf.size(),
                           /*add_special=*/false, /*parse_special=*/true);
    }
    if (n <= 0) return -1;
    return (int32_t) buf[n - 1];   // <|name|> special token is the last id
}

// Plain-token count helper used by kev_backend_get_ops-less code paths is not
// needed; count goes through the vocab_opaque + kev::tokenize path in-process.

kev_backend_ops g_ops = {};

} // namespace

// ---- exported factory ----

extern "C" {

KEV_BACKEND_API int kev_backend_get_chat_data(kev_backend_t * b, int kind, char * out,
                                              size_t cap, size_t * needed, char * err, size_t err_cap) {
    const auto * kb = as_cpu(b);
    return kev_get_chat_data(kb ? kb->model : nullptr, kind, out, cap, needed, err, err_cap);
}

KEV_BACKEND_API kev_backend_meta kev_backend_get_meta(void) {
    kev_backend_meta m{};
    m.abi_version  = KEV_BACKEND_ABI_VERSION;
    m.name         = "cpu";
    m.device       = "cpu";
    m.version      = "kev-backend-cpu-0.1-static-single-llama";
    m.capabilities = KEV_BACKEND_CAP_BATCH_DECODE | KEV_BACKEND_CAP_SEQ_MEMORY | KEV_BACKEND_CAP_HIDDEN_READ
                     | KEV_BACKEND_CAP_LOGITS_READ;
    m.struct_size  = sizeof(kev_backend_meta);
    return m;
}

KEV_BACKEND_API kev_backend_ops kev_backend_get_ops(void) {
    if (g_ops.struct_size == 0) {
        g_ops.struct_size     = sizeof(kev_backend_ops);
        g_ops.create          = op_create;
        g_ops.destroy         = op_destroy;
        g_ops.init            = op_init;
        g_ops.decode_seq      = op_decode_seq;
        g_ops.decode_batch    = op_decode_batch;
        g_ops.read_hidden     = op_read_hidden;
        g_ops.seq_clear       = op_seq_clear;
        g_ops.seq_copy        = op_seq_copy;
        g_ops.n_ctx_seq_capacity = op_n_ctx_seq_capacity;
        g_ops.count_tokens    = op_count_tokens;
        g_ops.model_n_embd    = op_model_n_embd;
        g_ops.threads         = op_threads;
        g_ops.threads_batch   = op_threads_batch;
        g_ops.vocab_opaque    = op_vocab_opaque;
        g_ops.tokenize        = op_tokenize;
        g_ops.special_token_id = op_special_token_id;
        g_ops.read_logits     = op_read_logits;
        g_ops.vocab_n_tokens  = op_vocab_n_tokens;
        g_ops.n_batch         = op_n_batch;
        g_ops.decode_multi_seq = op_decode_multi_seq;
    }
    return g_ops;
}

KEV_BACKEND_API kev_backend_t * kev_backend_create(void) {
    kev_backend_t * out = nullptr;
    (void) op_create(&out);
    return out;
}

KEV_BACKEND_API void kev_backend_destroy(kev_backend_t * b) {
    op_destroy(b);
}

#ifdef _WIN32
// Internal: called by the probe subprocess (main_server --kev-probe-cpu-variant).
// Registers the given ggml-cpu variant in THIS process, runs the KAT (metrics
// printed to stdout as KAT_BEGIN..KAT_END), and returns the KAT status. The
// short human-readable status also lands in `report`. The parent process
// compares the captured metrics against the x64 reference.
KEV_BACKEND_API int kev_probe_cpu_variant(const char * variant_dll, char * report, size_t report_cap) {
    auto wr = [&](const char * s) { if (report && report_cap) std::snprintf(report, report_cap, "%s", s); };
    if (!variant_dll || !variant_dll[0]) { wr("probe: empty variant path"); return 2; }
    ggml_backend_reg_t reg = ggml_backend_load(variant_dll);
    if (!reg) { wr("probe: ggml_backend_load failed"); return 1; }
    const int rc = kat_run_and_report();
    wr(rc == 0 ? "KAT_OK" : "KAT_FAILED");
    return rc;
}
#endif

} // extern "C"
