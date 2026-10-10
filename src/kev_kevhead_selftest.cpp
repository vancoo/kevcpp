// kev_kevhead_selftest.cpp — 无权重 KEV 打分器的数值自检 (最小测试, 无框架)。
//
// 手工构造已知输入, 断言:
//   1) KEV 半边点积 + 归一正确 (n_embd 偶数)。
//   2) 温度 softmax 定义正确 (数值稳定)。
//   (不改代码时可通过; 失败则 exit 非 0。)
#include "kev_kevhead.h"
#include "kev_config.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
    else std::printf("ok  : %s\n", what);
}

} // namespace

int main() {
    // ---- 已知输入: n_embd=4, m=2 ----
    // h_decide = [1, 2, 0, 0]  (只有前 m=2 维参与 KEV 点积)
    // opt0     = [0, 0, 3, 4]  -> dot = 1*3 + 2*4 = 11,    s0 = 11/sqrt(2)
    // opt1     = [0, 0, 1, 1]  -> dot = 1*1 + 2*1 = 3,     s1 = 3/sqrt(2)
    // opt2     = [0, 0, 0, 0]  -> dot = 0,                 s2 = 0
    const int n_embd = 4, m = n_embd / 2;
    std::vector<float> h_decide = {1.0f, 2.0f, 0.0f, 0.0f};
    std::vector<float> h0 = {0.0f, 0.0f, 3.0f, 4.0f};
    std::vector<float> h1 = {0.0f, 0.0f, 1.0f, 1.0f};
    std::vector<float> h2 = {0.0f, 0.0f, 0.0f, 0.0f};
    const float * opts[3] = {h0.data(), h1.data(), h2.data()};

    const float inv_sqrt_m = 1.0f / std::sqrt((float) m);
    const float s0 = 11.0f * inv_sqrt_m, s1 = 3.0f * inv_sqrt_m, s2 = 0.0f;
    const float mx = s0;
    const float e0 = std::exp(s0 - mx), e1 = std::exp(s1 - mx), e2 = std::exp(s2 - mx);
    const double sum = e0 + e1 + e2;
    const float p0 = (float) (e0 / sum), p1 = (float) (e1 / sum), p2 = (float) (e2 / sum);

    // 显式 T=1.0
    {
        float p[3] = {0, 0, 0};
        kev::KevKevHead::probs(h_decide.data(), opts, 3, n_embd, 1.0f, p);
        check(std::fabs(p[0] - p0) < 1e-6f && std::fabs(p[1] - p1) < 1e-6f && std::fabs(p[2] - p2) < 1e-6f,
              "KEV dot + T=1.0 softmax matches hand-computed expectations");
        const float sump = p[0] + p[1] + p[2];
        check(std::fabs(sump - 1.0f) < 1e-6f && p[0] > p[1] && p[1] > p[2], "probs normalized & ordered");
    }

    // 温度 T=2.0 应"软化"分布: logits 除以 2, 概率更平均, 但保持单调。
    {
        float p[3] = {0, 0, 0};
        kev::KevKevHead::probs(h_decide.data(), opts, 3, n_embd, 2.0f, p);
        const float invT2 = 0.5f;
        const float e0_2 = std::exp((s0 - mx) * invT2), e1_2 = std::exp((s1 - mx) * invT2), e2_2 = std::exp((s2 - mx) * invT2);
        const double sum2 = e0_2 + e1_2 + e2_2;
        check(std::fabs(p[0] - (float)(e0_2 / sum2)) < 1e-6f &&
              std::fabs(p[1] - (float)(e1_2 / sum2)) < 1e-6f &&
              std::fabs(p[2] - (float)(e2_2 / sum2)) < 1e-6f,
              "T=2.0 softmax matches hand-computed (softer) expectations");
        check(p[0] > p[1] && p[1] > p[2] && p[0] < p0, "T=2.0 keeps order but top prob is lower (softer)");
    }

    // 奇数 n_embd: 截断 m=n_embd/2(与 llama.cpp 一致), 不崩溃且归一。
    {
        const int odd = 5, mo = odd / 2;
        std::vector<float> d(odd, 1.0f), o(odd, 0.0f);
        std::vector<float> o1(odd, 0.0f);
        // 第一个 option 后半(mo..)全 1, 第二个全 0 => dot>0 与 0。
        for (int i = mo; i < odd; ++i) { o[i] = 1.0f; o1[i] = 0.0f; }
        const float * oo[2] = {o.data(), o1.data()};
        float p[2] = {0, 0};
        kev::KevKevHead::probs(d.data(), oo, 2, odd, 1.0f, p);
        const float mof = (float) mo;
        const float sd0 = (float) mo / std::sqrt(mof);     // dot = m, / sqrt(m)
        const float sd1 = 0.0f;
        const float Ex = std::exp(sd0 - sd0), Ez = std::exp(sd1 - sd0);
        const double S = Ex + Ez;
        check(std::fabs(p[0] - (float)(Ex / S)) < 1e-6f && std::fabs(p[1] - (float)(Ez / S)) < 1e-6f
              && std::fabs(p[0] + p[1] - 1.0f) < 1e-6f,
              "odd n_embd truncates m=floor(n_embd/2), still normalized");
    }

    // 防御: 空 opts(0 个)不写、不崩; n_opts<=0 返回。
    {
        float p[1] = {123.0f};
        kev::KevKevHead::probs(h_decide.data(), opts, 0, n_embd, 1.0f, p);
        check(p[0] == 123.0f, "n_opts=0 writes nothing");
    }

    // KevKevHead 对象 dim/temperature setter/getter。
    {
        kev::KevKevHead head;
        head.set_dim(2048); head.set_temperature(0.9f);
        check(head.dim() == 2048 && std::fabs(head.temperature() - 0.9f) < 1e-6f,
              "KevKevHead::dim()/temperature() accessors");
        check(kev::default_decision_config().option_end == "<|box_end|>", "default config keeps Qwen3.5 option_end");
    }

    // 决策配置 JSON 加载: 写一份临时配置, 断言 marker/temperature 解析正确, 且缺省字段不覆盖默认。
    {
        const char * tmp = "kev_kevhead_selftest_cfg.json";
        {
            std::ofstream f(tmp, std::ios::binary);
            f << "{\n \"markers\": {\"state\":\"<s>\",\"question\":\"<q>\","
                 "\"option_start\":\"<o>\",\"option_end\":\"<oe>\",\"decide\":\"<d>\"},"
              "\n \"temperature\": 0.75\n}";
        }
        kev::KevDecisionConfig c;
        std::string err;
        bool ok = kev::load_decision_config(tmp, c, &err);
        check(ok && err.empty(), "load_decision_config parses a valid JSON file");
        check(ok && c.state_marker == "<s>" && c.question_marker == "<q>" && c.option_start == "<o>"
              && c.option_end == "<oe>" && c.decide_marker == "<d>", "config markers parsed");
        check(ok && std::fabs(c.temperature - 0.75f) < 1e-6f, "config temperature parsed");
        // 缺省字段应保留默认, 不被空文件覆盖。
        kev::KevDecisionConfig c2;
        std::string err2;
        bool ok2 = kev::load_decision_config(tmp, c2, &err2);
        check(ok2 && c2.temperature == 0.75f && c2.state_marker == "<s>", "same file re-loads consistently");
        std::remove(tmp);
        // 打不开的文件应优雅失败并给错误信息。
        kev::KevDecisionConfig c3;
        std::string err3;
        bool ok3 = kev::load_decision_config("no_such_cfg_anywhere.json", c3, &err3);
        check(!ok3 && !err3.empty(), "missing config file fails gracefully with error message");
    }

    std::printf(failures == 0 ? "RESULT: PASS\n" : "RESULT: FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}