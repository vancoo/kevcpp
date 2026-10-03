// main_cli.cpp — kevcpp M0 最小加载验证
//
// 目标：加载 GGUF 基座 -> llama_decode 一次 prefill -> 用 llama_get_embeddings_ith
// 取出某 token 的 n_embd 维 hidden state，打印模型描述 / n_embd / 该向量 L2 范数。
//
// 纯 CPU：n_gpu_layers=0（默认），embeddings=true，pooling_type=NONE。

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <thread>
#include <algorithm>

int main(int argc, char ** argv) {
    // unbuffer stdout so RESULT lines are visible even if a later step faults
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::fprintf(stderr, "usage: kev <model.gguf> [<text>] [n_ctx]\n");
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string text       = argc > 2 ? argv[2] : "The capital of France is";
    const int32_t     n_ctx      = argc > 3 ? std::atoi(argv[3]) : 2048;

    // ---- init backend / model ----
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU only

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "error: failed to load model '%s'\n", model_path.c_str());
        llama_backend_free();
        return 1;
    }

    const int32_t n_embd = llama_model_n_embd(model);
    char desc[512];
    llama_model_desc(model, desc, sizeof(desc));

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // ---- context params ----
    const bool use_logits = std::getenv("KEV_LOGITS") != nullptr; // isolation: pure logits decode
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = (uint32_t) n_ctx;
    cparams.n_batch     = 512;
    cparams.n_ubatch    = 512;
    // For pooling NONE + embeddings, ALL input tokens of a sequence are output
    // embeddings. The default n_outputs_max_per_seq=1 would size the output buffer
    // too small for that, so allow up to a full ubatch of outputs per sequence.
    cparams.n_outputs_max_per_seq = 512;
    cparams.embeddings    = !use_logits;      // extract hidden states
    cparams.pooling_type  = LLAMA_POOLING_TYPE_NONE;
    cparams.n_threads     = 0;                // auto
    cparams.n_threads_batch = 0;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "error: failed to init context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // ---- tokenize (no special tokens for plain text) ----
    std::vector<llama_token> tokens(n_ctx * 2);
    const int32_t n_tok = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
                                         tokens.data(), (int32_t) tokens.size(),
                                         /*add_special=*/false, /*parse_special=*/true);
    if (n_tok < 0) {
        std::fprintf(stderr, "error: tokenization overflow\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    tokens.resize(n_tok);

    std::fprintf(stderr, "model path : %s\n", model_path.c_str());
    std::fprintf(stderr, "model desc : %s\n", desc);
    std::fprintf(stderr, "n_embd     : %d\n", (int) n_embd);
    std::fprintf(stderr, "n_tokens   : %d (%s)\n", (int) n_tok, text.c_str());

    // ---- prefill decode (single batch, all tokens output for embeddings) ----
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tok);

    if (llama_decode(ctx, batch)) {
        std::fprintf(stderr, "error: llama_decode failed\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // ---- read last token's hidden state (or logits) ----
    float * embd;
    if (use_logits) {
        embd = llama_get_logits_ith(ctx, -1);
        std::fprintf(stderr, "mode: LOGITS (%d classes)\n", llama_vocab_n_tokens(vocab));
    } else {
        embd = llama_get_embeddings_ith(ctx, -1);
        std::fprintf(stderr, "mode: EMBEDDINGS\n");
    }
    if (!embd) {
        std::fprintf(stderr, "error: output not available\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    const int32_t dim = use_logits ? llama_vocab_n_tokens(vocab) : n_embd;

    double sum_sq = 0.0, sum = 0.0, sum_abs = 0.0;
    float  max_abs = 0.0f;
    bool   has_nan = false;
    for (int i = 0; i < dim; ++i) {
        const float v = embd[i];
        if (std::isnan(v) || std::isinf(v)) has_nan = true;
        sum_sq  += (double) v * (double) v;
        sum     += (double) v;
        sum_abs += std::fabs((double) v);
        max_abs  = std::max(max_abs, std::fabs(v));
    }
    const double l2 = std::sqrt(sum_sq);

    std::printf("%s(n_dim=%d) last-token:\n", use_logits ? "logits" : "hidden_state", (int) dim);
    std::printf("  L2 norm          = %f\n", l2);
    std::printf("  L1 norm          = %f\n", sum_abs);
    std::printf("  mean(abs)        = %f\n", dim ? sum_abs / dim : 0.0);
    std::printf("  sum              = %f\n", sum);
    std::printf("  max_abs          = %f\n", (double) max_abs);
    std::printf("  has_nan_or_inf   = %s\n", has_nan ? "YES (BAD)" : "no (OK)");
    std::printf("  first 8 values   =");
    for (int i = 0; i < std::min(dim, 8); ++i) std::printf(" %+.4f", embd[i]);
    std::printf("\n");

    const bool ok = !has_nan && l2 > 1e-9 && dim == (use_logits ? llama_vocab_n_tokens(vocab) : n_embd);
    std::printf("RESULT: %s\n", ok ? "PASS" : "FAIL");

    // ---- cleanup ----
    // NOTE: batch was produced by llama_batch_get_one (NOT llama_batch_init), which
    // does NOT heap-allocate its members — its `token` points at our own buffer. It
    // must NOT be passed to llama_batch_free (that would free() a non-malloc pointer
    // and corrupt the heap -> 0xC0000374). Nothing to free here.
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return ok ? 0 : 2;
}