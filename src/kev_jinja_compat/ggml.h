#pragma once

#include <cstdio>
#include <cstdlib>

// common/json.cpp uses ggml only for JSON_ASSERT. Keep the host llama-free.
#define GGML_ASSERT(condition) do { if (!(condition)) { std::fputs("Jinja JSON assertion failed: " #condition "\n", stderr); std::abort(); } } while (0)
