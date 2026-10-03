cmake_minimum_required(VERSION 3.21)
project(kevcpp LANGUAGES C CXX)

if (NOT MSVC)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#
# llama.cpp is vendored as a full copy under third_party/llama.cpp.
# Using add_subdirectory() makes LLAMA_STANDALONE=OFF inside, so the copy's
# root CMakeLists only builds the static `llama` + `ggml` targets and does NOT
# pull in common/tools/examples/tests (those are OFF by default when not
# standalone). We never modify E:\van\projects\llama.cpp upstream.
#
set(GGML_NATIVE ON CACHE BOOL "ggml: enable CPU runtime detection (best on target machine)" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "llama: use curl" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "llama: use openssl" FORCE)
# Conservative CPU baseline: disable the experimental runtime weight repack and the
# LLAMAFILE sgemm (they are optional perf kernels, disabled here for robustness and
# a simpler reproducible build).
set(GGML_CPU_REPACK OFF CACHE BOOL "ggml: runtime weight conversion Q4_0->Q4_X_X" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "ggml: use LLAMAFILE sgemm" FORCE)

add_subdirectory(third_party/llama.cpp)

#
# kev executable: minimal loader/prefill/hidden-state verification (M0).
# Later milestones (M1-M4) add their own sources here.
#
add_executable(kev
    src/main_cli.cpp
    src/kev_head.cpp
)

target_link_libraries(kev PRIVATE llama)
target_compile_features(kev PRIVATE cxx_std_17)

# M1 对拍工具: headbench(纯 PointerHead 数学对拍) + readout(端到端 hidden->head 概率)
add_executable(headbench
    src/headbench.cpp
    src/kev_head.cpp
    src/kev_encode.cpp
)
target_link_libraries(headbench PRIVATE llama)
target_include_directories(headbench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor)
target_compile_features(headbench PRIVATE cxx_std_17)

# M3 状态前缀缓存基准 (同 state 二次请求命中 + 概率不变 + 耗时对比)
add_executable(m3_bench
    src/m3_bench.cpp
    src/kev_head.cpp
    src/kev_encode.cpp
    src/kev_prefix.cpp
    src/kev_model.cpp
    src/kev_merge.cpp
)
target_link_libraries(m3_bench PRIVATE llama)
target_include_directories(m3_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor)
target_compile_features(m3_bench PRIVATE cxx_std_17)

# M5 真实 LoRA 对拍 (base vs base+LoRA hidden/概率落盘)
add_executable(m5_bench
    src/m5_bench.cpp
    src/kev_head.cpp
    src/kev_encode.cpp
    src/kev_prefix.cpp
    src/kev_model.cpp
    src/kev_merge.cpp
)
target_link_libraries(m5_bench PRIVATE llama)
target_include_directories(m5_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor)
target_compile_features(m5_bench PRIVATE cxx_std_17)

# M7 CPU 慢定位 (TypeSafe systemone 分段计时)
add_executable(m7_bench
    src/m7_bench.cpp
    src/kev_api.cpp
    src/kev_head.cpp
    src/kev_encode.cpp
    src/kev_prefix.cpp
    src/kev_model.cpp
    src/kev_merge.cpp
)
target_link_libraries(m7_bench PRIVATE llama)
target_include_directories(m7_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor)
target_compile_features(m7_bench PRIVATE cxx_std_17)

# M4 HTTP 服务 (TypeSafe /v1/systemone)
add_executable(main_server
    src/main_server.cpp
    src/kev_api.cpp
    src/kev_model.cpp
    src/kev_prefix.cpp
    src/kev_encode.cpp
    src/kev_head.cpp
    src/kev_merge.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor/cpp-httplib/httplib.cpp
)
target_link_libraries(main_server PRIVATE llama)
target_include_directories(main_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor)
target_compile_features(main_server PRIVATE cxx_std_17)
# cpp-httplib (vendored; 本版为 h + httplib.cpp 拆分, 需编译链接 httplib.cpp)
target_include_directories(main_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor/cpp-httplib)

if (MSVC)
    # source files contain non-ASCII (Chinese comments); keep them as UTF-8
    target_compile_options(kev PRIVATE /utf-8)
    target_compile_options(headbench PRIVATE /utf-8)
    target_compile_options(m3_bench PRIVATE /utf-8)
    target_compile_options(m5_bench PRIVATE /utf-8)
    target_compile_options(m7_bench PRIVATE /utf-8)
    target_compile_options(main_server PRIVATE /utf-8)
endif()

# ---- 仅追加(不修改上方任何已存在段) ----
# P.4 对照 bench: 等长多问"批量 decode" vs "逐问 decode" 逐位一致性 + 耗时。
add_executable(m8_bench
    src/m8_bench.cpp
    src/kev_api.cpp
    src/kev_head.cpp
    src/kev_encode.cpp
    src/kev_prefix.cpp
    src/kev_model.cpp
    src/kev_merge.cpp
)
target_link_libraries(m8_bench PRIVATE llama)
target_include_directories(m8_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor)
target_compile_features(m8_bench PRIVATE cxx_std_17)
if (MSVC)
    target_compile_options(m8_bench PRIVATE /utf-8)
endif()

# ---- 仅追加(不修改上方任何已存在段) ----
# cold_profile: 受控性能计时探针, 隔离 cold 请求的 图构建/调度固定开销 vs 前向带宽。
# 独立 bench, 只用 llama C API, 不经过 KevModel。新增 src/cold_profile.cpp。
add_executable(cold_profile
    src/cold_profile.cpp
)
target_link_libraries(cold_profile PRIVATE llama)
target_include_directories(cold_profile PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor)
target_compile_features(cold_profile PRIVATE cxx_std_17)
if (MSVC)
    target_compile_options(cold_profile PRIVATE /utf-8)
endif()

# ---- KEV perf-gemv: argmax A/B 对拍工具 (临时 target; 不改 build.ps1 清单, 单独 msbuild) ----
add_executable(kab_bench
    src/kab_bench.cpp
    src/kev_api.cpp
    src/kev_head.cpp
    src/kev_encode.cpp
    src/kev_prefix.cpp
    src/kev_model.cpp
    src/kev_merge.cpp
)
target_link_libraries(kab_bench PRIVATE llama)
target_include_directories(kab_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor)
target_compile_features(kab_bench PRIVATE cxx_std_17)
if (MSVC)
    target_compile_options(kab_bench PRIVATE /utf-8)
endif()

# ---- merged-LoRA: 生成把 LoRA fold 进基础权重的 merged GGUF (只写 src/, 不动 vendored) ----
add_executable(merge_lora
    src/merge_lora_tool.cpp
    src/kev_merge.cpp
)
target_link_libraries(merge_lora PRIVATE)
target_compile_features(merge_lora PRIVATE cxx_std_17)
if (MSVC)
    target_compile_options(merge_lora PRIVATE /utf-8)
endif()

# ---- merged-vs-separate LoRA 对拍 / 性能 A/B ----
add_executable(lora_ab
    src/lora_ab.cpp
    src/kev_api.cpp
    src/kev_model.cpp
    src/kev_prefix.cpp
    src/kev_encode.cpp
    src/kev_head.cpp
    src/kev_merge.cpp
)
target_link_libraries(lora_ab PRIVATE llama)
target_include_directories(lora_ab PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor)
target_compile_features(lora_ab PRIVATE cxx_std_17)
if (MSVC)
    target_compile_options(lora_ab PRIVATE /utf-8)
endif()