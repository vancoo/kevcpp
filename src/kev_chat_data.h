#pragma once

#include "kev_backend.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>

// Shared by the llama-owning backends; no llama symbols enter the host.
inline int kev_get_chat_data(const llama_model * model, int kind, char * out,
                             size_t cap, size_t * needed, char * err, size_t err_cap) noexcept {
    if (needed) *needed = 0;
    if (err && err_cap) err[0] = '\0';
    try {
        if (!model || !needed) throw std::runtime_error("model not initialized or needed is null");
        std::string value;
        if (kind == KEV_BACKEND_CHAT_TEMPLATE) {
            int n = llama_model_meta_val_str(model, "tokenizer.chat_template", nullptr, 0);
            if (n <= 0 || n > 16 * 1024 * 1024) {
                throw std::runtime_error("model has no default GGUF chat template");
            }
            std::string buf(static_cast<size_t>(n) + 1, '\0');
            int actual = llama_model_meta_val_str(model, "tokenizer.chat_template", &buf[0], buf.size());
            if (actual != n) throw std::runtime_error("failed to read GGUF chat template");
            value.assign(buf.data(), static_cast<size_t>(n));
        } else if (kind == KEV_BACKEND_CHAT_BOS || kind == KEV_BACKEND_CHAT_EOS) {
            const llama_vocab * vocab = llama_model_get_vocab(model);
            if (!vocab) throw std::runtime_error("model vocabulary unavailable");
            const llama_token token = kind == KEV_BACKEND_CHAT_BOS ? llama_vocab_bos(vocab) : llama_vocab_eos(vocab);
            if (token != LLAMA_TOKEN_NULL) {
                char buf[128];
                int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
                if (n < 0) {
                    if (n == std::numeric_limits<int32_t>::min()) throw std::runtime_error("special token is too large");
                    value.resize(static_cast<size_t>(-n));
                    int actual = llama_token_to_piece(vocab, token, &value[0], -n, 0, true);
                    if (actual != -n) throw std::runtime_error("failed to decode special token");
                } else {
                    value.assign(buf, static_cast<size_t>(n));
                }
            }
        } else {
            throw std::runtime_error("unknown chat data kind");
        }
        *needed = value.size() + 1;
        if (!out || cap < *needed) return 2;
        std::memcpy(out, value.c_str(), *needed);
        return 0;
    } catch (const std::exception & e) {
        if (err && err_cap) std::snprintf(err, err_cap, "%s", e.what());
    } catch (...) {
        if (err && err_cap) std::snprintf(err, err_cap, "unknown chat data error");
    }
    return 1;
}
