// merge_lora_tool.cpp — 命令行: 把 LoRA fold 进基础模型生成 merged GGUF。
//   merge_lora_tool --model base.gguf --lora lora.gguf --out merged.gguf [--scale 1.0]
#include "kev_merge.h"
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char ** argv) {
    std::string base, lora, out;
    float scale = 1.0f;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--model") base = next();
        else if (a == "--lora") lora = next();
        else if (a == "--out") out = next();
        else if (a == "--scale") scale = (float)std::atof(next().c_str());
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (base.empty() || lora.empty() || out.empty()) {
        std::fprintf(stderr, "usage: merge_lora_tool --model base.gguf --lora lora.gguf --out merged.gguf [--scale 1.0]\n");
        return 2;
    }
    std::string err;
    std::fprintf(stderr, "merging LoRA into base model...\n");
    if (!kev::merge_lora_into_model(base, lora, out, scale, &err)) {
        std::fprintf(stderr, "merge failed: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "merged GGUF written to %s\n", out.c_str());
    return 0;
}