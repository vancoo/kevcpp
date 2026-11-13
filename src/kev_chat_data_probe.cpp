#include "kev_backend_factory.h"
#include "kev_chat_template.h"

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
int missing_metadata(kev_backend_t *, int, char *, size_t, size_t *, char * err, size_t cap) {
    if (err && cap) std::snprintf(err, cap, "model has no default GGUF chat template");
    return 1;
}

int malformed_metadata(kev_backend_t *, int kind, char * out, size_t cap, size_t * needed,
                       char *, size_t) {
    const char * value = kind == KEV_BACKEND_CHAT_TEMPLATE ? "{% if %}" : "";
    *needed = std::char_traits<char>::length(value) + 1;
    if (!out || cap < *needed) return 2;
    std::char_traits<char>::copy(out, value, *needed);
    return 0;
}

bool check_error(const char * label, kev::Backend & backend) {
    std::string out = "unchanged", err;
    if (backend.render_chat("s", "u", &out, &err) || err.empty() || out != "unchanged") {
        std::fprintf(stderr, "%s did not preserve output and return an error\n", label);
        return false;
    }
    return true;
}
}

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: kev_chat_data_probe model.gguf [cpu|vulkan|gpu]\n");
        return 2;
    }
    std::string err, source, bos, eos, output;
    auto old_plugin = kev::Backend_wrap({}, {}, reinterpret_cast<kev_backend_t *>(1));
    auto missing = kev::Backend_wrap({}, {}, reinterpret_cast<kev_backend_t *>(1), missing_metadata);
    auto malformed = kev::Backend_wrap({}, {}, reinterpret_cast<kev_backend_t *>(1), malformed_metadata);
    if (!check_error("old plugin", *old_plugin) || !check_error("missing template", *missing) ||
        !check_error("malformed template", *malformed)) return 1;
    try {
        const auto values = kev::render_gguf_chat_template(
            "{{ bos_token }}|{{ eos_token }}|{{ enable_thinking }}|{{ add_generation_prompt }}|"
            "{{ messages[0].content }}|{{ messages[1].content }}",
            "<bos>", "<eos>", "S", "U");
        if (values != "<bos>|<eos>|False|True|S|U") throw std::runtime_error("wrong Jinja input variables");
        try {
            kev::render_gguf_chat_template("{{ raise_exception('expected failure') }}", "", "", "S", "U");
            throw std::runtime_error("runtime error was not raised");
        } catch (const std::exception & e) {
            if (std::string(e.what()).find("expected failure") == std::string::npos) throw;
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "synthetic template test failed: %s\n", e.what());
        return 1;
    }
    auto backend = kev::BackendFactory_load(argc == 3 ? argv[2] : "cpu", &err);
    if (!backend) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    if (backend->chat_template_data(&source, &bos, &eos, &err) || err.empty()) {
        std::fprintf(stderr, "uninitialized model did not fail explicitly\n");
        return 1;
    }
    if (!check_error("uninitialized model", *backend)) return 1;
    kev::BackendOptions opt;
    opt.model_path = argv[1];
    opt.n_ctx = 256;
    opt.n_seq_max = 1;
    opt.n_threads = 1;
    if (!backend->init(opt, &err)) {
        std::fprintf(stderr, "init failed: %s\n", err.c_str());
        return 1;
    }
    kev::Backend moved(std::move(*backend));
    if (!moved.chat_template_data(&source, &bos, &eos, &err)) {
        std::fprintf(stderr, "metadata failed: %s\n", err.c_str());
        return 1;
    }
    if (!moved.render_chat("  System instruction \n", "  Hello, world! \n", &output, &err)) {
        std::fprintf(stderr, "render failed: %s\n", err.c_str());
        return 1;
    }
    const std::string expected = "<|im_start|>system\nSystem instruction<|im_end|>\n"
        "<|im_start|>user\nHello, world!<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    if (output != expected) {
        std::fprintf(stderr, "Qwen 2B rendered bytes differ from expected output\n");
        return 1;
    }
    if (moved.render_chat("s", "u", nullptr, &err) || err.empty()) {
        std::fprintf(stderr, "null output did not fail explicitly\n");
        return 1;
    }
    std::string aliased = "unchanged";
    if (moved.render_chat("s", "u", &aliased, &aliased) || aliased != "unchanged") {
        std::fprintf(stderr, "aliased output and error were not rejected unchanged\n");
        return 1;
    }
    std::printf("PASS template_bytes=%zu bos=%s eos=%s output_bytes=%zu\n", source.size(), bos.c_str(), eos.c_str(), output.size());
    return 0;
}
