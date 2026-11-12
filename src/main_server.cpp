// main_server.cpp — M4: TypeSafe 兼容 HTTP 服务 (POST /v1/systemone, GET /v1/models)
//
// 用法:
//   main_server --model <gguf> [--host 127.0.0.1] [--port 8008]
//   可选 auth: 设环境变量 KEV_API_KEY 则要求请求头 Authorization: Bearer <key>.
// 响应头: x-typesafe-request-id(回显或生成), server-timing.
#include "httplib.h"
#include "kev_api.h"
#include "kev_model.h"
// NOTE: llama-free host. Log filtering that used to live here (llama_log_set)
// is now installed inside the backend dll (the only module linking llama).

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <string>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

using nlohmann::json;

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
    card["description"] = "kevcpp logits-primary decision model on " + base + " (model-agnostic logits discriminant, no KEV/head/LoRA-fold)";
    card["release_date"] = "unknown";
    card["run"] = "kevcpp";
    card["base"] = base;
    card["backend"] = "llama.cpp";
    card["backend_name"] = m.backend_name();
    card["backend_device"] = m.backend_device();
    card["backend_version"] = m.backend_version();
    card["dtype"] = "fp32";
    card["temperature"] = m.temperature();
    const auto & dc = m.decision_config();
    card["decision_markers"] = {{"state", dc.state_marker}, {"question", dc.question_marker},
                                {"option_start", dc.option_start}, {"option_end", dc.option_end},
                                {"decide", dc.decide_marker}};
    card["prefix_cache"] = {{"size", m.prefix_cache().capacity()},
                            {"cached_prefixes", (int) m.prefix_cache().contents().size()}};
    return card;
}

int main(int argc, char ** argv) {
    std::string model_path, lora_path, host = "127.0.0.1";
    std::string backend = "cpu";  // pluggable backend (default "cpu")
    std::string decision_config_file;   // 可选: 决策配置 JSON (模型无关 marker + 温度)
    std::string opt_end_override;       // 可选: 覆盖 option_end marker (--opt-end)
    std::string decide_token_override;  // 可选: 覆盖 decide marker (--decide-token)
    int port = 8008, threads = 0, ctx = -1, max_concurrency = 8, prefix_capacity = 4;
    float lora_scale = 1.0f, temperature = 0.0f;
    int n_batch = 0, n_ubatch = 0;   // 0 = use backend defaults (512/256)
    bool quiet = false;
    // MoE 专家 offload / GPU 层数控制。--gpu-layers N: offload 到 GPU 的层数(负=全上设备,
    // 0=全 CPU); --moe-mode vram|ram: vram=专家全量进显存(跟随 --gpu-layers), ram=专家强制
    // 留系统内存、按需转显存(降低显存占用, 默认节省显存)。仅对 MoE 模型 + 非 cpu 后端有意义。
    int gpu_layers = 0;
    bool gpu_layers_set = false;    // 是否显式指定 --gpu-layers
    std::string moe_mode = "ram";   // 默认省显存: 专家留内存, 其余按 --gpu-layers
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--model") model_path = next();
        else if (a == "--host") host = next();
        else if (a == "--port") port = std::atoi(next().c_str());
        else if (a == "--lora") lora_path = next();
        else if (a == "--lora-scale") lora_scale = (float) std::atof(next().c_str());
        else if (a == "--threads" || a == "-t") threads = std::atoi(next().c_str());
        else if (a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--max-concurrency") max_concurrency = std::atoi(next().c_str());
        else if (a == "--prefix-capacity") prefix_capacity = std::atoi(next().c_str());
        else if (a == "--n-batch") n_batch = std::atoi(next().c_str());
        else if (a == "--n-ubatch") n_ubatch = std::atoi(next().c_str());
        else if (a == "--backend") backend = next();
        else if (a == "--gpu-layers") { gpu_layers = std::atoi(next().c_str()); gpu_layers_set = true; }
        else if (a == "--moe-mode") moe_mode = next();
        else if (a == "--decide-config") decision_config_file = next();
        else if (a == "--temperature") temperature = (float) std::atof(next().c_str());
        else if (a == "--opt-end") opt_end_override = next();
        else if (a == "--decide-token") decide_token_override = next();
        else if (a == "--quiet") quiet = true;
        else if (a == "--help" || a == "-h") {
            std::fprintf(stderr,
              "usage: main_server --model gguf [options]\n"
              "  --lora <gguf>       LoRA adapter (optional); --lora-scale f\n"
              "  --decide-config <json> 决策配置(prompt_template/聚合/温度等), 模型无关; 见 config/ (optional)\n"
              "  --temperature f     logits softmax 温度 (default 1.0; 覆盖 config)\n"
              "  --opt-end <token>   覆盖 option_end marker (如 <|box_end|>; 覆盖 config)\n"
              "  --decide-token <token> 覆盖 decide marker (question 收尾; 覆盖 config)\n"
              "  --host ip  --port p listen address/port (default 127.0.0.1:8008)\n"
              "  --threads N / -t N  compute threads (0 or unset = auto detect cores, cap 32)\n"
               "  --ctx N             context size (default 8192; 2048 typical for short states)\n"
               "  --max-concurrency N concurrent sequence slots (default 8; lower values increase per-sequence context)\n"
               "  --prefix-capacity N prefix cache slots (default 4)\n"
              "  --n-batch N         llama n_batch (single llama_decode logical max, default 512)\n"
              "  --n-ubatch N        llama n_ubatch (actual prefill block, default 256; must be <= n_batch)\n"
              "  --backend <name>    inference backend: 'cpu' (default). 'vulkan'(需对应 dll 构建).\n"
              "  --gpu-layers N      layers offloaded to GPU (neg=all, 0=none; default: neg for gpu backends).\n"
              "  --moe-mode <mode>   MoE expert placement: 'ram' (default, experts stay in host RAM to save\n"
              "                       VRAM, loaded on demand) or 'vram' (experts fully in VRAM). MoE only.\n"
              "  --quiet             suppress per-request log lines\n");
            return 0;
        }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty()) {
        std::fprintf(stderr, "usage: main_server --model model.gguf [--decide-config cfg.json] [--host ip] [--port p] [--lora lora.gguf] [--lora-scale f] [--temperature f] [--opt-end tok] [--threads N] [--ctx N] [--quiet]\n");
        return 2;
    }

    // Single-exe CPU non-HIP gate (fallback, dll portability not required): this
    // binary is now llama-free, but the env var is harmless and kept as a belt-and-
    // suspenders guard for any backend dll that registers ggml backends eagerly.
    // It must be set BEFORE the backend dll is loaded / first llama/ggml call.
    if (backend == "cpu") {
#if defined(_WIN32)
        SetEnvironmentVariableA("GGML_BACKEND_DISABLE_GPU", "1");
#else
        setenv("GGML_BACKEND_DISABLE_GPU", "1", 1);
#endif
    }

    // (llama.cpp log filtering is installed inside the backend dll at init; the
    //  host is llama-free and cannot call llama_log_set.)

    if (!quiet) std::fprintf(stderr, "kevcpp server loading model: %s\n", model_path.c_str());
    kev::KevModelOptions opt;
    opt.model_path = model_path;
    opt.n_ctx = ctx > 0 ? ctx : 8192; opt.n_threads = threads; opt.backend = backend;
    opt.max_concurrency = max_concurrency > 0 ? max_concurrency : 1;
    opt.prefix_capacity = prefix_capacity > 0 ? prefix_capacity : 1;
    opt.n_batch = n_batch; opt.n_ubatch = n_ubatch;
    // MoE 专家 offload / GPU 层数:
    //  - 后端为 CPU 时 n_gpu_layers 恒 0(全 CPU, legacy 行为; MoE offload 无意义)。
    //  - 后端为 GPU/Vulkan 时, 未显式给 --gpu-layers 则默认全层上设备(n_gpu_layers=-1);
    //    显式给了则用给定值。
    //  - --moe-mode ram  => moe_offload=1, 专家经 tensor_buft_overrides 强制留 host RAM;
    //    --moe-mode vram => moe_offload=0, 专家跟随 n_gpu_layers 全量进显存。
    if (backend == "cpu") {
        opt.n_gpu_layers = 0;
    } else {
        opt.n_gpu_layers = gpu_layers_set ? gpu_layers : -1;
    }
    opt.moe_offload  = (moe_mode == "ram") ? 1 : 0;
    opt.decision_config = decision_config_file;
    opt.temperature_override = temperature > 0.0f ? temperature : 0.0f;
    opt.option_end_override  = opt_end_override;
    opt.decide_marker_override = decide_token_override;
    if (!lora_path.empty()) { opt.lora_path = lora_path; opt.lora_scale = lora_scale; }
    // B2: 并发批处理开关。默认开; KEV_NO_BATCH=1 退回原 compute_mutex_ 全程串行路径
    // (作 A/B 基准 / 可回退)。
    if (std::getenv("KEV_NO_BATCH")) { opt.enable_batching = false; }
    // B3: continuous-batching 窗口可调(微秒)。KEV_BATCH_AGG_US=0 即完全事件驱动(到达即解码)。
    if (const char * agg = std::getenv("KEV_BATCH_AGG_US")) { opt.batch_aggregation_us = std::atol(agg); }
    kev::KevModel model;
    std::string err;
    if (!model.init(opt, &err)) { std::fprintf(stderr, "kevcpp error: init failed: %s\n", err.c_str()); return 1; }
    if (!quiet) {
        std::fprintf(stderr, "kevcpp server model ready (n_embd=%d temp=%.3f lora=%s "
                     "logits_channel=%s)\n",
                 model.n_embd(), model.temperature(),
                 lora_path.empty() ? "-" : lora_path.c_str(),
                 model.logits_channel_active() ? "ON" : "OFF");
        std::fprintf(stderr, "kevcpp using backend '%s' (device=%s, ver=%s)\n",
                 model.backend_name(), model.backend_device(), model.backend_version());
        std::fprintf(stderr, "kevcpp using %d threads %s (n_threads_batch=%d)\n",
                 model.threads_actual(), threads == 0 ? "(auto)" : "(set)", model.threads_batch_actual());
        std::fprintf(stderr, "kevcpp batch config: n_batch=%d n_ubatch=%d\n",
                 n_batch > 0 ? n_batch : 512, n_ubatch > 0 ? n_ubatch : 256);
    }

    // 启动预热建图(P0a): 在监听前做一次最小 dummy decode, 把 llama.cpp 进程一次性的
    // 图构建 + compute buffer 预留移到服务起跑线之前。state(38) 与 branch(readout) 两形状
    // 都预建, 使冷启动首个请求的 state 与 branch 都不再付一次性建图(~290ms)。
    // 不触碰前缀缓存 / 业务输出, 数值逐位一致。默认开启; KEV_NO_WARM=1 关闭。
    if (std::getenv("KEV_NO_WARM") == nullptr) {
        std::string werr;
        if (model.warm_graph(/*state_tokens=*/38, /*branch_tokens=*/151, &werr)) {
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
        nlohmann::ordered_json body;
        try { body = nlohmann::ordered_json::parse(req.body); }
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
            c["description"] = "kevcpp logits-primary decision model (model-agnostic logits discriminant)";
            c["release_date"] = "unknown";
            c["base"] = model_path;
            c["backend"] = "llama.cpp";
            c["backend_name"] = model.backend_name();
            c["backend_device"] = model.backend_device();
            c["backend_version"] = model.backend_version();
            c["dtype"] = "fp32";
            c["temperature"] = model.temperature();
            c["prefix_cache"] = {{"size", model.prefix_cache().capacity()},
                                 {"cached_prefixes", (int) model.prefix_cache().contents().size()}};
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
