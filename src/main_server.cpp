// main_server.cpp — M4: TypeSafe 兼容 HTTP 服务 (POST /v1/systemone, GET /v1/models)
//
// 用法:
//   main_server --model <gguf> --head head.bin [--host 127.0.0.1] [--port 8008] [--lru 2]
//   可选 auth: 设环境变量 KEV_API_KEY 则要求请求头 Authorization: Bearer <key>.
// 响应头: x-typesafe-request-id(回显或生成), server-timing.
#include "httplib.h"
#include "kev_api.h"
#include "kev_model.h"
#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <fstream>
#include <string>

using nlohmann::json;

// 接管 llama.cpp/ggml 的日志: 只透传 ERROR 级别, 过滤 INFO/WARN/DEBUG
// (消灭 llama_graph_n_input_tensors / sched_reserve / metadata dump 等刷屏)。
static void llm_log_filter(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (level >= GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

// 每个请求一行核心日志
static void log_req(const char * method, const char * path, int status, double ms,
                    int in_tok, int out_tok) {
    std::time_t now = std::time(nullptr);
    std::tm tmv; 
#if defined(_WIN32)
    localtime_s(&tmv, &now);
#else
    localtime_r(&now, &tmv);
#endif
    char ts[32];
    std::strftime(ts, sizeof ts, "%H:%M:%S", &tmv);
    std::fprintf(stderr, "[%s] %s %s %d %.0fms input=%d output=%d\n",
                 ts, method, path, status, ms, in_tok, out_tok);
}

static std::string request_id(const httplib::Request & req) {
    if (req.has_header("x-typesafe-request-id")) return req.get_header_value("x-typesafe-request-id");
    // generate an id (not cryptographically strong; enough for a request tag)
    std::time_t now = std::time(nullptr);
    static long c = 0;
    char b[64];
    std::snprintf(b, sizeof b, "%08lx%08lx%04lx", (unsigned long) now, (unsigned long)(now >> 32),
                  (unsigned long)(++c));
    return b;
}

static json model_card(const kev::KevModel & m, const std::string & base) {
    json card;
    card["description"] = "Kev pointer head on " + base + " (kevcpp C++ CPU serving)";
    card["release_date"] = "unknown";
    card["run"] = "kevcpp";
    card["base"] = base;
    card["backend"] = "llama.cpp";
    card["dtype"] = "fp32";
    card["temperature"] = m.temperature();
    card["prefix_cache"] = {{"size", m.lru().capacity()}, {"cached_states", (int) m.lru().contents().size()}};
    return card;
}

int main(int argc, char ** argv) {
    std::string model_path, head, lora_path, host = "127.0.0.1";
    std::string prewarm_file;   // 可选: 启动预热 state 文本文件(--prewarm <file>)
    int port = 8008, lru = 2, threads = 0, ctx = -1; float lora_scale = 1.0f;
    bool quiet = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--model") model_path = next();
        else if (a == "--head") head = next();
        else if (a == "--host") host = next();
        else if (a == "--port") port = std::atoi(next().c_str());
        else if (a == "--lru") lru = std::atoi(next().c_str());
        else if (a == "--lora") lora_path = next();
        else if (a == "--lora-scale") lora_scale = (float) std::atof(next().c_str());
        else if (a == "--threads" || a == "-t") threads = std::atoi(next().c_str());
        else if (a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--prewarm") prewarm_file = next();
        else if (a == "--quiet") quiet = true;
        else if (a == "--help" || a == "-h") {
            std::fprintf(stderr,
              "usage: main_server --model gguf [options]\n"
              "  --head <bin>        pointer head .bin\n"
              "  --lora <gguf>       LoRA adapter (optional); --lora-scale f\n"
              "  --host ip  --port p listen address/port (default 127.0.0.1:8008)\n"
              "  --threads N / -t N  compute threads (0 or unset = auto detect cores, cap 32)\n"
              "  --ctx N             context size (default 8192; 2048 typical for short states)\n"
              "  --lru N             state cache slots (default 2)\n"
              "  --prewarm <file>    startup-prewarm a state: read raw state text, prefill into LRU (optional)\n"
              "  --quiet             suppress per-request log lines\n");
            return 0;
        }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty()) {
        std::fprintf(stderr, "usage: main_server --model model.gguf [--head head.bin] [--host ip] [--port p] [--lru n] [--lora lora.gguf] [--lora-scale f] [--threads N] [--ctx N] [--prewarm file] [--quiet]\n");
        return 2;
    }

    // 在任何模型动作之前就接管 llama.cpp 日志 -> 加载阶段的 metadata dump 也被过滤
    llama_log_set(llm_log_filter, nullptr);

    if (!quiet) std::fprintf(stderr, "kevcpp server loading model: %s\n", model_path.c_str());
    kev::KevModelOptions opt;
    opt.model_path = model_path; opt.head_path = head; opt.lru_capacity = lru;
    opt.n_ctx = ctx > 0 ? ctx : 8192; opt.n_threads = threads;
    if (!lora_path.empty()) { opt.lora_path = lora_path; opt.lora_scale = lora_scale; }
    kev::KevModel model;
    std::string err;
    if (!model.init(opt, &err)) { std::fprintf(stderr, "kevcpp error: init failed: %s\n", err.c_str()); return 1; }
    if (!quiet) {
        std::fprintf(stderr, "kevcpp server model ready (n_embd=%d temp=%.3f head=%s lora=%s)\n",
                 model.n_seq_max() ? model.dim_in() : 0, model.temperature(),
                 head.empty() ? "-" : head.c_str(), lora_path.empty() ? "-" : lora_path.c_str());
        std::fprintf(stderr, "kevcpp using %d threads %s (n_threads_batch=%d)\n",
                 model.threads_actual(), threads == 0 ? "(auto)" : "(set)", model.threads_batch_actual());
    }

    // 可选启动预热(P0a): state 文本 encode 后 prefill 进状态缓存。默认不指定 => 完全不影响现状。
    if (!prewarm_file.empty()) {
        std::ifstream pf(prewarm_file, std::ios::binary);
        if (!pf) {
            std::fprintf(stderr, "kevcpp error: cannot open prewarm file: %s\n", prewarm_file.c_str());
            return 1;
        }
        std::string state_text((std::istreambuf_iterator<char>(pf)), std::istreambuf_iterator<char>());
        if (!state_text.empty() && state_text.back() == '\n') state_text.pop_back(); // 去尾换行, 与 body 的 state 一致
        // 仅取 state 部分 ids: 空 questions => enc 全为 state 段。
        kev::KevEnc e = model.encode(kev::KevRequest{state_text, {}, {}}, nullptr);
        auto rows = model.rows_of(e, nullptr);
        if (!rows.state_ids.empty()) {
            if (model.prewarm_state(rows.state_ids)) {
                if (!quiet) std::fprintf(stderr, "kevcpp prewarmed state (%zu tokens) into LRU\n", rows.state_ids.size());
            } else if (!quiet) {
                std::fprintf(stderr, "kevcpp warning: prewarm_state failed for %s\n", prewarm_file.c_str());
            }
        } else if (!quiet) {
            std::fprintf(stderr, "kevcpp warning: prewarm state produced no tokens: %s\n", prewarm_file.c_str());
        }
    }

    // 启动预热建图(P0a): 在监听前做一次最小 dummy decode, 把 llama.cpp 进程一次性的
    // 图构建 + compute buffer 预留移到服务起跑线之前。state(38) 与 branch(readout) 两形状
    // 都预建, 使冷启动首个请求的 state 与 branch 都不再付一次性建图(~290ms)。
    // 不触碰 LRU / 缓存 state / 业务输出, 数值逐位一致。默认开启; KEV_NO_WARM=1 关闭。
    if (std::getenv("KEV_NO_WARM") == nullptr) {
        std::string werr;
        if (model.warm_graph(/*warm_branch=*/true, /*state_tokens=*/38, /*branch_tokens=*/151, &werr)) {
            if (!quiet) std::fprintf(stderr, "kevcpp warmed decode graph (state+branch) before listen\n");
        } else {
            std::fprintf(stderr, "kevcpp warning: warm_graph failed: %s\n", werr.empty() ? "unknown" : werr.c_str());
        }
    }

    const char * api_key = std::getenv("KEV_API_KEY");

    httplib::Server svr;
    svr.set_payload_max_length(16ull * 1024 * 1024);

    auto maybe_auth = [&](const httplib::Request & req, httplib::Response & res) -> bool {
        if (!api_key) return true;
        std::string want = std::string("Bearer ") + api_key;
        bool ok = req.has_header("authorization") && req.get_header_value("authorization") == want;
        if (!ok) {
            res.status = 401;
            res.set_header("www-authenticate", "Bearer");
            res.set_content(R"({"detail":"missing or invalid API key; send Authorization: Bearer <KEV_API_KEY>"})", "application/json");
        }
        return ok;
    };

    svr.Post("/v1/systemone", [&](const httplib::Request & req, httplib::Response & res) {
        auto t0 = std::chrono::steady_clock::now();
        res.set_header("x-typesafe-request-id", request_id(req));
        auto elapsed_ms = [&]() {
            return (double) std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0).count() / 1e3;
        };
        if (!maybe_auth(req, res)) { if (!quiet) log_req("POST", "/v1/systemone", 401, elapsed_ms(), 0, 0); return; }
        const bool TR = (std::getenv("KEV_TRACE") != nullptr && std::string(std::getenv("KEV_TRACE")) == "1");
        double tw = 0.0, t_parse = 0.0, t_sys = 0.0, t_dump = 0.0, t_http = 0.0;
        if (TR) tw = elapsed_ms();
        std::string err;
        json body;
        try { body = json::parse(req.body); }
        catch (...) { res.status = 400; res.set_content(R"({"detail":"invalid JSON body"})", "application/json");
                      if (!quiet) log_req("POST", "/v1/systemone", 400, elapsed_ms(), 0, 0); return; }
        if (TR) { t_parse = elapsed_ms() - tw; tw = elapsed_ms(); }
        double model_ms = 0.0;
        json out = kev::systemone_json(model, body, &model_ms, &err);
        if (TR) { t_sys = elapsed_ms() - tw; tw = elapsed_ms(); }
        if (out.is_null()) {
            res.status = 422;
            json e = {{"detail", err.empty() ? std::string("could not answer request") : err}};
            res.set_content(e.dump(), "application/json");
            if (!quiet) log_req("POST", "/v1/systemone", 422, elapsed_ms(), 0, 0);
            return;
        }
        res.set_content(out.dump(), "application/json");
        if (TR) { t_dump = elapsed_ms() - tw; tw = elapsed_ms(); }
        char tb[64];
        std::snprintf(tb, sizeof tb, "app;dur=%.1f", elapsed_ms());
        res.set_header("server-timing", tb);
        int in_tok = 0, out_tok = 0;
        if (out.contains("usage")) {
            in_tok  = out["usage"].value("input_tokens", 0);
            out_tok = out["usage"].value("output_tokens", 0);
        }
        if (TR) t_http = elapsed_ms() - tw;
        if (TR) std::fprintf(stderr, "[TRACE] handler json_parse=%.2fms systemone_json=%.2fms out_dump=%.2fms "
                             "httphead+log_req=%.2fms TOTAL=%.2fms\n", t_parse, t_sys, t_dump, t_http, elapsed_ms());
        if (!quiet) log_req("POST", "/v1/systemone", 200, elapsed_ms(), in_tok, out_tok);
    });

    svr.Get("/v1/models", [&](const httplib::Request & req, httplib::Response & res) {
        auto t0 = std::chrono::steady_clock::now();
        res.set_header("x-typesafe-request-id", request_id(req));
        auto elapsed_ms = [&]() {
            return (double) std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0).count() / 1e3;
        };
        if (!maybe_auth(req, res)) { if (!quiet) log_req("GET", "/v1/models", 401, elapsed_ms(), 0, 0); return; }
        auto card = [&](const std::string & name) {
            json c;
            c["name"] = name;
            c["description"] = "Kev pointer head on Qwen3.5-0.8B (kevcpp C++ CPU serving)";
            c["release_date"] = "unknown";
            c["base"] = "qwen3.5-0.8b";
            c["backend"] = "llama.cpp";
            c["dtype"] = "fp32";
            c["temperature"] = model.temperature();
            c["prefix_cache"] = {{"size", model.lru().capacity()}};
            return c;
        };
        json out = {{"models", json::array({card("kev-latest"), card("jev-latest")})}};
        res.set_content(out.dump(), "application/json");
        char tb[64]; std::snprintf(tb, sizeof tb, "app;dur=%.1f", elapsed_ms());
        res.set_header("server-timing", tb);
        if (!quiet) log_req("GET", "/v1/models", 200, elapsed_ms(), 0, 0);
    });

    std::fprintf(stderr, "kevcpp server listening on %s:%d  (POST /v1/systemone, GET /v1/models; auth=%s)\n",
                 host.c_str(), port, api_key ? "on" : "off");
    svr.listen(host.c_str(), port);
    model.close();
    return 0;
}