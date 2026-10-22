// token_position_probe.cpp — 诊断: "模型从 Answer: 开始实际前 10 个 token 是什么,正确答案在第几位"
//
// 直接链接 vendored llama.cpp(有 llama.lib), 不用 llama-cli、不滚动到 kevcpp backend。
// 01 层次1: decode 完整 prompt, 读末位(Lp-1, 'Answer:' 后)全量 vocab logits, 打印 top-k token 文本,
//          并标出正确答案对应 token 是否在 top-k 内、top1 是不是答案。
// 02 层次2: 贪心滚动 —— 取末位 argmax 追加到序列再 decode, 循环; 打印前 10 个产生的 token, 标出
//          正确答案 token 出现于第几步。
//
// 用法: token_position_probe --model <gguf> [--claims] [--steps N]
// 默认 steps=10 (贪心滚动步数)。只诊断、零训练; 不做任何打分改进。
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace {

std::string render_prompt(const std::string & tpl, const std::string & state,
                          const std::string & instr, const std::vector<std::string> & opts,
                          const std::vector<std::string> & keys,
                          const std::vector<std::string> & corr_keys) {
    (void)corr_keys;
    auto rep = [](std::string & p, const std::string & ph, const std::string & val) {
        size_t at = 0;
        while ((at = p.find(ph, at)) != std::string::npos) { p.replace(at, ph.size(), val); at += val.size(); }
    };
    std::string p = tpl;
    std::string opt_str;
    for (size_t i = 0; i < opts.size(); ++i) { if (i) opt_str += ", "; opt_str += opts[i]; }
    std::string opt_lines;
    for (size_t i = 0; i < opts.size(); ++i) { if (i) opt_lines += "\n"; opt_lines += opts[i]; }
    rep(p, "{state}", state);
    rep(p, "{instructions}", instr);
    rep(p, "{options}", opt_str);
    rep(p, "{optlist}", opt_lines);
    return p;
}

std::string detok(const llama_vocab * vocab, llama_token id) {
    // llama_token_to_piece(vocab, token, buf, length, lstrip, special)
    std::vector<char> buf(64);
    int n = llama_token_to_piece(vocab, id, buf.data(), (int) buf.size(), 0, false);
    if (n < 0) { buf.resize((size_t)(-n)); n = llama_token_to_piece(vocab, id, buf.data(), (int) buf.size(), 0, false); }
    if (n < 0) n = 0;
    return std::string(buf.data(), (size_t) n);
}

struct Case {
    const char * name;      // 编号/标签
    const char * qtype;     // noul/choice/score
    const char * instr;     // instructions
    const char * state;     // 状态文本
    std::vector<std::string> keys;      // 候选 key(token 读数对象)
    std::vector<std::string> opts;      // 展示文本(key: desc)+ 模板里拼
    const char * corr;      // 正确答案 key
    int          corr_idx;  // 正确答案在 keys 里的下标, 或 -1
};

std::vector<Case> cases() {
    // 覆盖: 否定/反义 always-yes 重灾(noul), choice, score, 日期。
    std::vector<Case> base = {
        {"c1n_noul", "noul", "A公司是否参与B公司的本轮增资？",
         "A公司宣布将不参与B公司的本轮增资。",
         {"false","true"}, {"no: 否", "yes: 是"}, "false", 0},
        {"c2n_noul", "noul", "该项目是否已开工？",
         "该项目没有通过环评审批，因此未开工。",
         {"false","true"}, {"no: 否", "yes: 是"}, "false", 0},
        {"c3n_noul", "noul", "客户是否表示不满？",
         "客户没有表示不满，反而赞扬了我们的服务。",
         {"false","true"}, {"no: 否", "yes: 是"}, "false", 0},
        {"c4n_noul", "noul", "公司2024年是否存在重大内部控制缺陷？",
         "经审计，公司2024年不存在重大内部控制缺陷。",
         {"false","true"}, {"no: 否", "yes: 是"}, "false", 0},
        {"c5_choice", "choice", "哪家公司2023年营收更高？",
         "公司A 2023年营收5.2亿元，公司B 2023年营收3.7亿元。",
         {"A","B"}, {"A: 公司A", "B: 公司B"}, "A", 0},
        {"c6_score", "score", "客户有多生气？",
         "如果再不处理，我就要去消费者协会投诉你们了！",
         {"0","1","2"}, {"平静", "不满", "非常愤怒"}, "2", 2},
        {"c7_date", "noul", "事件A是否比事件B发生得晚？",
         "事件A发生于2024-02-29，事件B发生于2024-03-01。",
         {"false","true"}, {"no: 否", "yes: 是"}, "false", 0},
        {"c8_choice", "choice", "谁在1号位？",
         "原本A球在1号位，B球在2号位。A和B交换位置后。",
         {"A","B"}, {"A: A球", "B: B球"}, "B", 1},
    };

    // env KEV_BENCH_CASES=1: 把 tests/bench-60cases.jsonl 的 85 题追加为 case(经 llama 循环
    // 已有的贪心生成路径跑, 规避自写 decode 崩溃)。用于测"模型自由生成自答准确率"。
    if (const char * bc = std::getenv("KEV_BENCH_CASES")) {
        if (std::string(bc) == "1") {
            const char * bp = std::getenv("KEV_BENCH");
            std::string path = bp ? std::string(bp) : std::string("tests/bench-60cases.jsonl");
            std::ifstream ifs(path);
            std::string line;
            std::vector<Case> extra;
            static std::vector<std::string> holder;   // 保活所有动态字符串, 避免 const char* 悬垂
            auto keep = [&](const std::string & s) { holder.push_back(s); return holder.back().c_str(); };
            while (std::getline(ifs, line)) {
                if (line.empty()) continue;
                nlohmann::json rec; try { rec = nlohmann::json::parse(line); } catch (...) { continue; }
                std::string state0 = rec["state"].get<std::string>();
                auto & exp = rec["expected"];
                auto & qs = rec["questions"];
                for (auto it = qs.begin(); it != qs.end(); ++it) {
                    const auto & q = it.value();
                    std::string qtype = q["type"].get<std::string>();
                    std::string instr0 = q.value("instructions", "");
                    std::vector<std::string> keys, opts;
                    std::string corr;
                    nlohmann::json ev = exp[it.key()];
                    int corr_idx = -1;
                    if (qtype == "choice") {
                        for (auto cit = q["criteria"].begin(); cit != q["criteria"].end(); ++cit) { keys.push_back(cit.key()); opts.push_back(cit.key() + ": " + cit.value().get<std::string>()); }
                        corr = ev.get<std::string>();
                        for (size_t j = 0; j < keys.size(); ++j) if (keys[j] == corr) corr_idx = (int) j;
                    } else if (qtype == "noul") {
                        keys = {"false", "true"}; opts = {"否", "是"};
                        bool b = ev.get<bool>();
                        corr = b ? "true" : "false"; corr_idx = b ? 1 : 0;
                    } else {
                        int ki = 0; for (auto & x : q["criteria"]) { keys.push_back(std::to_string(ki++)); opts.push_back(x.get<std::string>()); }
                        int ei = ev.get<int>(); corr = std::to_string(ei); corr_idx = ei;
                    }
                    std::string nm = "b" + std::to_string(extra.size());
                    extra.push_back({keep(nm), keep(qtype), keep(instr0), keep(state0), keys, opts, keep(corr), corr_idx});
                }
            }
            return extra;
        }
    }
    return base;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path, freegen_path, tpl_arg;
    int steps = 10;
    int max_gen = 8;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--steps" && i + 1 < argc) steps = std::atoi(argv[++i]);
        else if (a == "--freegen" && i + 1 < argc) freegen_path = argv[++i];
        else if (a == "--template" && i + 1 < argc) tpl_arg = argv[++i];
        else if (a == "--max-gen" && i + 1 < argc) max_gen = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "usage: token_position_probe --model <gguf> [--steps N] [--freegen out.txt]\n"); return 2; }
    }
    if (model_path.empty()) { std::fprintf(stderr, "usage: token_position_probe --model <gguf> [--steps N]\n"); return 2; }
    const char * bench_env = std::getenv("KEV_BENCH");
    const std::string BENCH_PATH = bench_env ? std::string(bench_env) : std::string("tests/bench-60cases.jsonl");
    (void)tpl_arg;

    llama_backend_init();
    setvbuf(stdout, nullptr, _IONBF, 0); // 实时输出, 防异常退出时缓冲丢失

    llama_model_params mpar = llama_model_default_params();
    mpar.n_gpu_layers = 0; // CPU
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mpar);
    if (!model) { std::fprintf(stderr, "fail: load model %s\n", model_path.c_str()); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_ctx = 2048;

    llama_context_params cpar = llama_context_default_params();
    cpar.n_ctx = n_ctx;
    cpar.n_batch = 512;
    cpar.n_threads = 0; // auto
    llama_context * ctx = llama_init_from_model(model, cpar);
    if (!ctx) { std::fprintf(stderr, "fail: init ctx\n"); return 1; }

    // 模板变体数组(任务: prompt 次序/内容调优)。目标是"Lp-1 首 decode(argmax) == 候选标签 token"出现频率最高。
    // {optlist} = 选项逐行(A. xx\nB. yy); {options} = 逗号连接。{instructions}/{state} 占位。
    const std::vector<const char *> TEMPLATES = {
        // ---- 基线 / 收敛词 ----
        // V0 基线(现状)
        "{instructions}\n[ {options} ]\n{state}\nAnswer:",
        // V1 结论是 (已知 85 题 best template 之一)
        "{instructions}\n[ {options} ]\n{state}\n结论是：",
        // V2 少字强命令
        "{instructions}\n[ {options} ]\n{state}\n正确选项是：",
        // ---- 次序/结构 ----
        // V3 state 在选项前
        "{state}\n{instructions}\n[ {options} ]\n结论是：",
        // V4 state 在选项后(与默认同), 但收尾"答案是"
        "{instructions}\n[ {options} ]\n{state}\n答案是：",
        // V5 选项逐行 + 结论是
        "{instructions}\n{optlist}\n{state}\n结论是：",
        // V6 选项逐行 + 答案是
        "{instructions}\n{optlist}\n{state}\n答案是：",
        // V7 系统头 + 收敛
        "系统：你是一个严格按给定选项作答的判别器。\n{instructions}\n[ {options} ]\n{state}\n答案：",
        // ---- 内容/措辞 ----
        // V8 禁空格禁解释
        "{instructions}\n[ {options} ]\n{state}\n请只输出一个选项，不要空格、标点或解释。答案：",
        // V9 只输出标签不输出内容
        "{instructions}\n[ {options} ]\n{state}\n只输出选项答案，不要输出标点或空格。答案：",
        // V10 "请选出一个答案"
        "{instructions}\n[ {options} ]\n{state}\n请选出一个答案：",
        // V11 "你的选择是"
        "{instructions}\n[ {options} ]\n{state}\n你的选择是：",
        // ---- 少字钉子 ----
        // V12 单个"答"
        "{instructions}\n[ {options} ]\n{state}\n答：",
        // V13 换行交互式
        "{instructions}\n[ {options} ]\n{state}\n请回答：",
        // V14 结论是 + 选项逐行 + state先行
        "{state}\n{instructions}\n{optlist}\n结论是：",
        // V15 few-shot 前置格式样板 + 结论是
        "例：题干：某讯 → 答案：A\n{instructions}\n[ {options} ]\n{state}\n结论是：",
        // V16 最简"答案是"(无标点)
        "{instructions}\n[ {options} ]\n{state}\n答案是:",
        // V17 nothink: Qwen3.5 原生 ChatML + enable_thinking=false 的空思考块
        //   (<|im_start|>assistant\n thinking\n\n</think>\n\n) —— 关闭思考, 让模型直接作答。
        "<|im_start|>system\n你是严格按给定选项作答的判别器，只输出最终选项，不要任何思考或解释。<|im_end|>\n"
        "<|im_start|>user\n{instructions}\n[ {options} ]\n{state}<|im_end|>\n"
        "<|im_start|>assistant\n thinking\n\n</think>\n\n",
        // ---- ChatML 收尾消融(验证首位空格是否由 ' response\n\n' 换行造成) ----
        // N1: 少一个换行
        "<|im_start|>system\n你是严格按给定选项作答的判别器，只输出最终选项，不要任何思考或解释。<|im_end|>\n"
        "<|im_start|>user\n{instructions}\n[ {options} ]\n{state}<|im_end|>\n"
        "<|im_start|>assistant\n thinking\n\n</think>\n",
        // N2: 不进 think/response, 直接 assistant 起始(老 chatml 收尾)
        "<|im_start|>system\n你是严格按给定选项作答的判别器，只输出最终选项，不要任何思考或解释。<|im_end|>\n"
        "<|im_start|>user\n{instructions}\n[ {options} ]\n{state}<|im_end|>\n"
        "<|im_start|>assistant\n",
        // N3: assistant 后直接给"答案："
        "<|im_start|>system\n你是严格按给定选项作答的判别器，只输出最终选项，不要任何思考或解释。<|im_end|>\n"
        "<|im_start|>user\n{instructions}\n[ {options} ]\n{state}<|im_end|>\n"
        "<|im_start|>assistant\n答案：",
        // N4: assistant + 强指令(禁空格/解释) + "答案："
        "<|im_start|>system\n你是严格按给定选项作答的判别器。<|im_end|>\n"
        "<|im_start|>user\n{instructions}\n[ {options} ]\n{state}<|im_end|>\n"
        "<|im_start|>assistant\n只输出一个选项，不要空格、标点或解释。答案：",
        // N5: 空思考块 + 结尾不给换行, 直接跟"答案："钉子
        "<|im_start|>system\n你是严格按给定选项作答的判别器，只输出最终选项，不要任何思考或解释。<|im_end|>\n"
        "<|im_start|>user\n{instructions}\n[ {options} ]\n{state}<|im_end|>\n"
        "<|im_start|>assistant\n thinking\n\n</think>\n答案：",
        // ---- 方案B: 纯提示词引导"模型该输出哪个词"(对齐 false/true 比对对象) ----
        // 均基于 nothink2 修正收尾( response\n答案：, 首token=答案)。差异只在"要求用什么词作答"。
        // B1: 要求用 false/true 作答(英文布尔)
        "<|im_start|>system\n你是严格按给定选项作答的判别器。<|im_end|>\n"
        "<|im_start|>user\n{instructions}\n[ {options} ]\n{state}\n请只回答 true 或 false(不要输出其他任何内容):<|im_end|>\n"
        "<|im_start|>assistant\n thinking\n\n</think>\n答案：",
        // B2: 要求用 True/False 作答(首字母大写)
        "<|im_start|>system\n你是严格按给定选项作答的判别器。<|im_end|>\n"
        "<|im_start|>user\n{instructions}\n[ {options} ]\n{state}\n请从 True 或 False 中选择一个回答:<|im_end|>\n"
        "<|im_start|>assistant\n thinking\n\n</think>\n答案：",
        // B3: 要求用 是/否 作答(中文布尔)
        "<|im_start|>system\n你是严格按给定选项作答的判别器。<|im_end|>\n"
        "<|im_start|>user\n{instructions}\n[ {options} ]\n{state}\n请只回答 是 或 否(不要输出其他任何内容):<|im_end|>\n"
        "<|im_start|>assistant\n thinking\n\n</think>\n答案：",
        // B4: 要求用 Yes/No 作答
        "<|im_start|>system\n你是严格按给定选项作答的判别器。<|im_end|>\n"
        "<|im_start|>user\n{instructions}\n[ {options} ]\n{state}\n请只回答 Yes 或 No(不要输出其他任何内容):<|im_end|>\n"
        "<|im_start|>assistant\n thinking\n\n</think>\n答案：",
        // B5: 直接给出"唯一允许的输出集"(true/false) 并禁用解释
        "<|im_start|>system\n你只能输出 true 或 false 两个词之一，输出其他任何内容都是错误的。<|im_end|>\n"
        "<|im_start|>user\n{instructions}\n[ {options} ]\n{state}<|im_end|>\n"
        "<|im_start|>assistant\n thinking\n\n</think>\ntrue/false：",
        // B6: 用"判断为真/假"的中文表述引导输出 真/假
        "<|im_start|>system\n你是严格按给定选项作答的判别器。<|im_end|>\n"
        "<|im_start|>user\n{instructions}\n[ {options} ]\n{state}\n该陈述成立吗？请只回答 真 或 假:<|im_end|>\n"
        "<|im_start|>assistant\n thinking\n\n</think>\n答案：",
        // B7: 回显候选集: 明确"从 [false, true] 中选一个原样输出"
        "<|im_start|>system\n你是严格按给定选项作答的判别器，必须原样输出候选词之一。<|im_end|>\n"
        "<|im_start|>user\n{instructions}\n可选答案只有两个: false, true。\n{state}\n请原样输出其中一个(不要翻译、不要解释):<|im_end|>\n"
        "<|im_start|>assistant\n thinking\n\n</think>\n答案：",
    };
    // 含 ChatML 特殊 token 的模板必须 parse_special=true, 否则 <|im_start|> 会被当普通文本。
    auto is_chatml = [](const std::string & t) { return t.find("<|im_start|>") != std::string::npos; };

    std::fprintf(stderr, "model loaded: n_vocab=%d n_embd=%d\n\n", n_vocab, llama_model_n_embd(model));

    for (const char * trav : TEMPLATES) {
      const std::string tpl = trav;
      // 只看单个模板时(KEV_TPL_ONLY=<index>, 0-based)跳过其余 —— 供 85 题 benchmark 快速跑。
      if (const char * only = std::getenv("KEV_TPL_ONLY")) {
          int idx = std::atoi(only);
          static int tcount = 0;
          if (tcount++ != idx) continue;
      }
      int n_cand_argmax = 0, n_cases_run = 0, n_alias_correct = 0;
      std::fprintf(stdout, "\n###### TEMPLATE: %s\n", trav);
      for (const auto & c : cases()) {
        std::string prompt = render_prompt(tpl, c.state, c.instr, c.opts, c.keys, {});
        std::vector<llama_token> ids(n_ctx);
        ids.assign(n_ctx, (llama_token) 0); // 每 case 复原缓冲(容纳任意长度 prompt)
        int n_tok = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), ids.data(), (int) ids.size(), /*add_special=*/false, /*parse_special=*/is_chatml(tpl));
        if (n_tok < 0) { n_tok = -n_tok; } // 需要的容量
        if (n_tok <= 0) { std::fprintf(stderr, "[%s] tokenize empty\n", c.name); continue; }
        if ((int) ids.size() > n_tok) ids.resize(n_tok); // 只保留真实 token, 避免 attr 全零长度污染贪心序列

        // ---- 层次1: Lp-1 位全量 top-k ----
        llama_batch b = llama_batch_init(n_tok, /*embd=*/0, 1);
        for (int i = 0; i < n_tok; ++i) { b.token[i] = ids[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 0; }
        b.logits[n_tok - 1] = 1; // 标记末位
        b.n_tokens = n_tok;
        llama_memory_clear(llama_get_memory(ctx), true); // 每 decode 全量从 pos 0 开始, 需清缓存避免 M-RoPE X<Y 冲突
        int r = llama_decode(ctx, b);
        llama_batch_free(b);
        if (r != 0) { std::fprintf(stderr, "[%s] decode err %d\n", c.name, r); continue; }

        // 末位 logits 行 = 该位置模型的 next-token 分布
        float * lg = llama_get_logits_ith(ctx, n_tok - 1);

        // 答案 token id 集合(候选 key 的 token, 主要看 key)
        std::vector<std::pair<llama_token,int>> corr_toks;
        std::vector<int> key_first_tok(c.keys.size(), -1);
        for (size_t k = 0; k < c.keys.size(); ++k) {
            std::vector<llama_token> kt(16);
            int kn = llama_tokenize(vocab, c.keys[k].c_str(), (int) c.keys[k].size(), kt.data(), 16, false, false);
            std::vector<llama_token> kk;
            if (kn > 0) for (int z = 0; z < kn; ++z) kk.push_back(kt[z]);
            if (!kk.empty()) key_first_tok[k] = kk[0];
            if ((int)k == c.corr_idx && !kk.empty()) corr_toks.push_back({kk[0], 0});
        }

        // 找 top-k(这里 top=8)
        std::vector<std::pair<int,float>> scored; scored.reserve(n_vocab);
        for (int v = 0; v < n_vocab; ++v) scored.push_back({v, lg[v]});
        std::partial_sort(scored.begin(), scored.begin() + std::min(8,(int)scored.size()), scored.end(),
                          [](const auto & x, const auto & y){ return x.second > y.second; });
        int t8 = std::min(8, (int)scored.size());

        std::fprintf(stdout, "\n===== %s [%s] n_tok(Lp)=%d corr=%s =====\n", c.name, c.qtype, n_tok, c.corr);
        std::fprintf(stdout, "Lp-1('Answer:') top-%d tokens:\n", t8);
        bool top1_is_corr = false;
        for (int i = 0; i < t8; ++i) {
            llama_token id = (llama_token) scored[i].first;
            std::string txt = detok(vocab, id);
            bool iscorr = false;
            for (auto & ct : corr_toks) if (ct.first == id) iscorr = true;
            // 也标出它是否命中任一个候选 key 首token
            int hit_key = -1;
            for (size_t k = 0; k < key_first_tok.size(); ++k) if (key_first_tok[k] == (int)id) hit_key = (int)k;
            if (i == 0) top1_is_corr = iscorr;
            std::fprintf(stdout, "  [%d] id=%d logit=%7.2f '%s'  %s%s%s\n",
                         i, (int)id, scored[i].second, txt.c_str(),
                         iscorr ? "<<CORRECT>>" : "",
                         hit_key >= 0 ? " (==cand key)" : "",
                         hit_key >= 0 && hit_key != c.corr_idx ? " MISMATCH" : "");
            (void)top1_is_corr;
        }

        // ::::: compact prompt-template debug: 各候选标签 logit vs 空格 logit / argmax / 首decode是否=候选 :::::
        {
            // 空格 token ' ' 的 id
            std::vector<llama_token> sp(8);
            int sn = llama_tokenize(vocab, " ", 1, sp.data(), (int) sp.size(), false, false);
            llama_token space_tok = sn > 0 ? sp[0] : (llama_token) 0;
            std::string arg_txt = detok(vocab, (llama_token) scored[0].first);
            double space_lg = lg[space_tok];
            // 各候选 label 首 token 的 logit
            std::string cand_line = "";
            int cand_best_idx = -1; double cand_best_lg = -1e30;
            for (size_t k = 0; k < key_first_tok.size(); ++k) {
                double lv = key_first_tok[k] >= 0 ? (double) lg[key_first_tok[k]] : -1e30;
                cand_line += std::string(" ").c_str() + c.keys[k] + "=" + (lv < -1e29 ? "NA" : std::to_string(lv).substr(0,6));
                if (lv > cand_best_lg) { cand_best_lg = lv; cand_best_idx = (int) k; }
            }
            // argmax 是否为任一候选 key 首token
            llama_token argmax_tok = (llama_token) scored[0].first;
            bool arg_is_cand = false; int arg_hit = -1;
            for (size_t k = 0; k < key_first_tok.size(); ++k) if (key_first_tok[k] == (int) argmax_tok) { arg_is_cand = true; arg_hit = (int)k; break; }
            if (arg_is_cand) n_cand_argmax++;
            n_cases_run++;
            std::fprintf(stdout, "  [PTMPL] space='%s'(logit %.2f) argmax='%s' cand_best=%s(%.2f) | margin(cand_best-space)=%.2f | %s\n",
                         detok(vocab, space_tok).c_str(), space_lg, arg_txt.c_str(),
                         arg_is_cand ? c.keys[arg_hit].c_str() : "(none)", cand_best_lg, cand_best_lg - space_lg,
                         arg_is_cand ? "first-decode==CAND" : "first-decode!=cand");
            (void)cand_line;
        }

        // ::::: 不微调相似度/相关性: 多语言别名池 logit 聚合(路径2) :::::
        // 对每个选项构造多语言/同义别名池, 选项得分 = logsumexp(该池内各 token 首token 的 logit)。
        // 目的: 消除"模型说 否 但 key 是 false"的错位; 决策 argmax 是否=正确选项。
        {
            auto lse = [&](const std::vector<std::string> & pool) -> double {
                std::vector<double> vs;
                for (const auto & w : pool) {
                    std::vector<llama_token> wt(16);
                    int wn = llama_tokenize(vocab, w.c_str(), (int) w.size(), wt.data(), 16, false, false);
                    if (wn <= 0) continue;
                    vs.push_back((double) lg[wt[0]]);
                }
                if (vs.empty()) return -1e30;
                double mx = *std::max_element(vs.begin(), vs.end());
                double s = 0; for (double v : vs) s += std::exp(v - mx);
                return mx + std::log(s);
            };
            // 每题的别名池: 依 qtype/key 构造(通用: key + 常见多语言/同义别名)。
            const int Kpools = (int) c.keys.size();
            std::vector<std::vector<std::string>> pools(Kpools);
            if (std::string(c.qtype) == "noul") {
                // keys = {false, true}
                pools[0] = {"false", "False", "否", "no", "No", "假", "不"};
                if (Kpools > 1) pools[1] = {"true", "True", "是", "yes", "Yes", "真"};
            } else if (std::string(c.qtype) == "score") {
                for (int j = 0; j < Kpools; ++j) pools[j] = { std::to_string(j), std::to_string(j + 1) };
            } else {
                for (int j = 0; j < Kpools; ++j) pools[j] = { c.keys[j] };
            }
            std::vector<double> z(Kpools, 0.0);
            for (int j = 0; j < Kpools; ++j) z[j] = lse(pools[j]);
            int best = (int)(std::max_element(z.begin(), z.end()) - z.begin());
            bool alias_hit = (best == c.corr_idx);
            if (alias_hit) n_alias_correct++;
            std::fprintf(stdout, "  [ALIAS] qtype=%s alias-argmax=%s corr=%s -> %s\n",
                         c.qtype, c.keys[best].c_str(), c.corr,
                         alias_hit ? "ALIAS-CORRECT" : "alias-wrong");
        }

        // ---- 层次2: 贪心滚动前 N 步 ----
        std::fprintf(stdout, "greedy rollout (top1 each step, steps param=%d):\n", steps);
        std::vector<llama_token> seq = ids;
        int found_step = -1;
        for (int st = 0; st < steps && (int) seq.size() < n_ctx; ++st) {
            std::fprintf(stdout, "  [iter %d] seq_len=%zu\n", st, seq.size()); // DEBUG
            int N = (int) seq.size();
            llama_batch bg = llama_batch_init(N, 0, 1);
            for (int i = 0; i < N; ++i) { bg.token[i] = seq[i]; bg.pos[i] = i; bg.n_seq_id[i] = 1; bg.seq_id[i][0] = 0; bg.logits[i] = 0; }
            bg.logits[N - 1] = 1;
            bg.n_tokens = N;
            llama_memory_clear(llama_get_memory(ctx), true);
            int rg = llama_decode(ctx, bg);
            llama_batch_free(bg);
            if (rg != 0) { std::fprintf(stdout, "  step %d decode err %d\n", st, rg); break; }
            float * lgg = llama_get_logits_ith(ctx, N - 1);
            llama_token argmax = 0; float best = lgg[0];
            for (int v = 1; v < n_vocab; ++v) if (lgg[v] > best) { best = lgg[v]; argmax = v; }
            std::string txt = detok(vocab, argmax);
            bool iscorr = false;
            for (auto & ct : corr_toks) if (ct.first == argmax) iscorr = true;
            if (found_step < 0 && iscorr) found_step = st;
            std::fprintf(stdout, "  step %d: id=%d '%s'%s\n", st, (int)argmax, txt.c_str(), iscorr ? "  <<CORRECT TOKEN HERE>>" : "");
            seq.push_back(argmax);
        }
        std::fprintf(stdout, "  corr token first appears at rolling step: %d  (0=the very first generated token; -1=not in first %d)\n",
                     found_step, steps);

        // 汇总
        std::vector<int> cand_first = key_first_tok;
        int top1 = (int) scored[0].first;
        bool top1_hits_any_cand = false; int top1_hit = -1;
        for (size_t k = 0; k < cand_first.size(); ++k) if (cand_first[k] == top1) { top1_hits_any_cand = true; top1_hit = (int)k; }
        if (top1_hits_any_cand) {
            std::fprintf(stdout, "-> Lp-1 top1 IS a candidate key[%d] (=%s); corr key idx=%d -> %s\n",
                         top1_hit, c.keys[top1_hit].c_str(), c.corr_idx,
                         (top1_hit == c.corr_idx) ? "CORRECT" : "WRONG-choice");
        } else {
            std::fprintf(stdout, "-> Lp-1 top1 is NOT any candidate key (it's a filler/other token): '%s'\n",
                         detok(vocab, (llama_token) top1).c_str());
        }
      } // for (cases())

      // ---- TEMPLATE 级汇总: 首decode==候选标签 频率 ----
      std::fprintf(stdout, "--- [TEMPLATE-AGG] first-decode==CAND: %d/%d ---\n", n_cand_argmax, n_cases_run);
      std::fprintf(stdout, "--- [TEMPLATE-AGG] alias-pool-argmax==CORRECT: %d/%d ---\n", n_alias_correct, n_cases_run);
      (void)(n_cand_argmax);
      std::fprintf(stdout, "--- [template-summary] rank of candidate-label among top, space-margin per case ---\n");
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}