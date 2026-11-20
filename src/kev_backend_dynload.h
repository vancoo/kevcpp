// kev_backend_dynload.h — ensure the dynamic ggml GPU backend library
// (ggml-vulkan / ggml-cuda) is registered even when the backend registry is
// already non-empty when the GPU plugin initializes.
//
// Background (v0.7.0 Windows package regression): on Windows multi-variant
// builds the CPU KAT probe pre-registers a ggml-cpu-* variant via
// ggml_backend_load() BEFORE the GPU backend plugin initializes. Because
// llama_backend_init() only calls ggml_backend_load_all() when the registry is
// EMPTY (third_party/llama.cpp/src/llama.cpp), the ggml-vulkan / ggml-cuda
// dynamic library shipped next to the executable is then never discovered and
// the GPU backend silently falls back to the CPU (verified: zero GPU engine
// activity, no vulkan-1.dll loaded in-process). Linux v0.7.0 builds were
// unaffected (the KAT pre-registration is _WIN32-only) but this helper keeps
// both platforms and any future Windows CUDA build safe.
//
// ggml_backend_load() cannot be used here: in this build layout the GPU
// libraries are regular shared ggml components exporting their reg getter
// (ggml_backend_vk_reg / ggml_backend_cuda_reg), not GGML_BACKEND_DL dynamic
// backends exporting ggml_backend_init. So we load the library ourselves,
// resolve the reg getter and register the returned reg with the process-wide
// registry (ggml_backend_register), mirroring what ggml does for
// statically-linked backends.
//
// Behavior of ensure_gpu_backend():
//   * no-op when a GPU device is already registered (statically linked
//     backends, ggml-vulkan/cuda already linked into the binary, or load_all
//     already ran on an empty registry);
//   * otherwise loads the library, first by bare name (resolves against the
//     executable directory on Windows), then by absolute path next to this
//     plugin's shared object;
//   * prints a loud warning when no GPU device can be registered, instead of
//     silently degrading to CPU.
#pragma once

#include "ggml-backend.h"

#include <cstdio>
#include <string>
#include <vector>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace kev_dynload {

inline bool has_gpu_device() {
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
           ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU) != nullptr;
}

inline std::string gpu_lib_name(const char * backend_name) {
#ifdef _WIN32
    return std::string("ggml-") + backend_name + ".dll";
#else
    return std::string("libggml-") + backend_name + ".so";
#endif
}

// Directory of the shared object containing this inline function (the backend
// plugin); the ggml GPU library ships next to it.
inline std::string self_module_dir() {
#ifdef _WIN32
    HMODULE h = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&self_module_dir), &h)) {
        return {};
    }
    WCHAR w[MAX_PATH] = {};
    const DWORD n = GetModuleFileNameW(h, w, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return {};
    std::string p(n * 4, '\0');
    const int m = WideCharToMultiByte(CP_UTF8, 0, w, (int) n, &p[0], (int) p.size(), nullptr, nullptr);
    if (m <= 0) return {};
    p.resize((size_t) m);
    const auto pos = p.find_last_of("/\\");
    return pos == std::string::npos ? std::string() : p.substr(0, pos);
#else
    Dl_info info = {};
    if (dladdr(reinterpret_cast<void *>(&self_module_dir), &info) && info.dli_fname && info.dli_fname[0]) {
        const std::string p = info.dli_fname;
        const auto pos = p.find_last_of("/\\");
        return pos == std::string::npos ? std::string() : p.substr(0, pos);
    }
    return {};
#endif
}

// Load 'path' and keep it loaded for the process lifetime (mirrors ggml's own
// backend loading: the registry keeps a dl handle forever).
inline void * load_library_keep(const std::string & path) {
#ifdef _WIN32
    return reinterpret_cast<void *>(LoadLibraryA(path.c_str()));
#else
    return dlopen(path.c_str(), RTLD_NOW);
#endif
}

inline void * resolve_symbol(void * handle, const char * symbol) {
#ifdef _WIN32
    return reinterpret_cast<void *>(GetProcAddress(reinterpret_cast<HMODULE>(handle), symbol));
#else
    return dlsym(handle, symbol);
#endif
}

// Explicitly load the dynamic ggml GPU backend library for 'backend_name'
// ("vulkan" / "cuda") and register its backend via 'reg_symbol'
// ("ggml_backend_vk_reg" / "ggml_backend_cuda_reg") when no GPU device is
// registered yet. Returns true when a GPU device is (or became) available;
// warns loudly otherwise.
inline bool ensure_gpu_backend(const char * backend_name, const char * reg_symbol) {
    if (has_gpu_device()) return true;

    const std::string lib = gpu_lib_name(backend_name);

    // Candidate paths: bare name first (Windows resolves against the
    // executable directory / Linux against LD_LIBRARY_PATH), then the
    // directory of this plugin's shared object.
    std::vector<std::string> candidates{lib};
    const std::string dir = self_module_dir();
    if (!dir.empty()) {
        candidates.push_back(dir +
#ifdef _WIN32
            "\\"
#else
            "/"
#endif
            + lib);
    }

    for (const auto & path : candidates) {
        void * handle = load_library_keep(path);
        if (!handle) continue;

        using reg_fn_t = ggml_backend_reg_t (*)(void);
        auto reg_fn = reinterpret_cast<reg_fn_t>(resolve_symbol(handle, reg_symbol));
        if (reg_fn) {
            ggml_backend_reg_t reg = reg_fn();
            if (reg) {
                ggml_backend_register(reg);
                if (has_gpu_device()) return true;
            }
        }
    }

    std::fprintf(stderr,
                 "[kev] WARN: no GPU device registered for backend '%s': failed to load '%s' and register "
                 "'%s'. The model will run on the CPU. Check that %s is present next to kevcpp and that a "
                 "usable GPU driver/runtime is installed.\n",
                 backend_name, lib.c_str(), reg_symbol, lib.c_str());
    return false;
}

} // namespace kev_dynload
