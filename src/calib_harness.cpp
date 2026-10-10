// calib_harness.cpp — 方案3: logits 通道「决策映射层」校准/评测 harness (无训练权重, 模型无关)
//
// 用途: 用「少量已知正确答案」的标签样例, 离线统计当前 logits 现象与校准效果:
//   1. 从模型收集每个候选 key 的原始 logits z(不应用校准), 只 decode 一次。
//   2. 统计校准前: choice 准确率 / 置信-准确率校准(ECE) / 分布形态 / 位置顺序偏置。
//   3. 在【校准集】上网格搜温度 T / 拟合位置偏置 / 低置信阈值 —— 全部在已收集的 z 上做(零 re-decode)。
//   4. 在【固定测试集】(与校准集不重叠)上做「校准前 vs 校准后」回归对比。
//   5. 打印拟合后的校准参数(可落盘供运行期决策配置加载)。
//
// 用法:
//   calib_harness --model model.gguf [--decide-config cfg.json]
//                 --calib calib.json --test test.json
//                 [--out params.txt] [--threads N] [--ctx N]
// 数据文件格式(每条一个 sample, 单问或多问; 每问 criteria + 一个 "answer"=ground-truth key):
//   { "samples":[ { "state": {...}, "questions": {
//         "q0": {"type":"choice","instructions":"...",
//                "criteria": {"A":"...","B":"...","C":"..."},
//                "answer":"B"} } } ] }
#include "kev_api.h"
#include "kev_model.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

using nlohmann::json;

namespace {

struct DP {                    // 一个数据点 = 一个 question 的候选 logits + 标签
    int K = 0;
    std::vector<float> z;      // 候选 key 原始 logits(softmax 前)
    int true_idx = -1;
};

// 聚合指标
struct Agg {
    int n = 0;                 // 数据点数
    double acc = 0;            // choice 准确率
    double mean_conf = 0;      // 全部样本平均归一化置信
    double mean_conf_correct = 0, mean_conf_wrong = 0;
    int n_correct = 0, n_wrong = 0;
    double ece = 0;            // 10 桶 ECE(argmax 置信 vs 正确率)
    std::vector<double> pos_acc;   // 每位置准确率(候选顺序位置)
};

static std::vector<float> map_probs(const std::vector<float> & z,
                                    const kev::LogitsCalibrator & cal,
                                    float base_temp) {
    // 校准关闭 = 原始行为 softmax(z / base_temp)
    if (!cal.enabled) { std::vector<float> o; kev::LogitsCalibrator::softmax_scaled(z, base_temp, o); return o; }
    std::vector<float> o;
    cal.apply(z, o);
    return o;
}

Agg evaluate(const std::vector<DP> & dps, const kev::LogitsCalibrator & cal, float base_temp,
             float drop = 0.0f, int * n_dropped = nullptr) {
    Agg a; a.n = (int) dps.size();
    const int NB = 10;
    std::vector<double> bin_conf(NB, 0), bin_acc(NB, 0); std::vector<int> bin_n(NB, 0);
    int maxpos = 0; for (auto & d : dps) maxpos = std::max(maxpos, d.K);
    a.pos_acc.assign(maxpos + 1, 0); std::vector<int> pos_n(maxpos + 1, 0);
    if (n_dropped) *n_dropped = 0;
    for (const auto & d : dps) {
        auto p = map_probs(d.z, cal, base_temp);
        int am = (int) (std::max_element(p.begin(), p.end()) - p.begin());
        float conf = kev::LogitsCalibrator::normalized_confidence(p);
        bool drop_it = drop > 0.0f && conf < drop;
        if (drop_it) { if (n_dropped) ++*n_dropped; continue; }  // 拒绝硬选
        a.mean_conf += conf;
        bool correct = (am == d.true_idx);
        int b = (int) std::min(NB - 1, (int)(conf * NB));
        bin_n[b]++; bin_conf[b] += conf; bin_acc[b] += correct ? 1 : 0;
        if (correct) { a.n_correct++; a.mean_conf_correct += conf; }
        else { a.n_wrong++; a.mean_conf_wrong += conf; }
        if (d.true_idx >= 0 && d.true_idx < maxpos + 1) { pos_n[d.true_idx]++; a.pos_acc[d.true_idx] += correct; }
    }
    int total = a.n_correct + a.n_wrong;
    a.acc = total > 0 ? (double) a.n_correct / total : 0.0;
    a.mean_conf /= (double) total;
    a.mean_conf_correct /= (double) (a.n_correct > 0 ? a.n_correct : 1);
    a.mean_conf_wrong   /= (double) (a.n_wrong > 0 ? a.n_wrong : 1);
    double ece_num = 0; int used = 0;
    for (int b = 0; b < NB; ++b) if (bin_n[b]) { used += bin_n[b]; ece_num += std::fabs((double) bin_acc[b] / bin_n[b] - bin_conf[b] / bin_n[b]) * bin_n[b]; }
    if (used > 0) a.ece = ece_num / used;
    for (int j = 0; j <= maxpos; ++j) a.pos_acc[j] = pos_n[j] ? a.pos_acc[j] / pos_n[j] : 0;
    return a;
}

const char * bool_str(bool v) { return v ? "true" : "false"; }

} // namespace

int main(int argc, char ** argv) {
    std::string model, decide_config, calib_file, test_file, out_file;
    int threads = 0, ctx = 2048;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--model") model = next();
        else if (a == "--decide-config") decide_config = next();
        else if (a == "--calib") calib_file = next();
        else if (a == "--test") test_file = next();
        else if (a == "--out") out_file = next();
        else if (a == "--threads" || a == "-t") threads = std::atoi(next().c_str());
        else if (a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--help" || a == "-h") {
            std::printf("usage: calib_harness --model gguf [--decide-config cfg.json] --calib c.json --test t.json [--out params.txt] [--threads N] [--ctx N]\n");
            return 0;
        }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (model.empty() || calib_file.empty() || test_file.empty()) {
        std::fprintf(stderr, "need --model, --calib, --test\n"); return 2;
    }

#if defined(_WIN32)
    SetEnvironmentVariableA("GGML_BACKEND_DISABLE_GPU", "1");
#endif

    kev::KevModelOptions opt;
    opt.model_path = model; opt.head_path.clear();
    opt.n_ctx = ctx > 0 ? ctx : 2048; opt.n_threads = threads; opt.backend = "cpu";
    opt.decision_config = decide_config;
    kev::KevModel m;
    std::string err;
    if (!m.init(opt, &err)) { std::fprintf(stderr, "init failed: %s\n", err.c_str()); return 1; }
    std::fprintf(stderr, "model ready: n_embd=%d temp=%.3f backend=%s\n",
                 m.dim_in(), m.temperature(), m.backend_name());
    const float base_temp = m.decision_config().temperature > 0.0f ? m.decision_config().temperature : 1.0f;

    // 精确加载(含 ground truth label): 每条 sample 逐问收集候选 logits + 定位 true_idx。
    std::vector<DP> calib_dps, test_dps;
    auto load_exact = [&](const std::string & path, std::vector<DP> & out, std::string * e) -> bool {
        std::ifstream f(path, std::ios::binary);
        if (!f) { if (e) *e = "cannot open " + path; return false; }
        json roots;
        try { roots = json::parse(f); } catch (const std::exception & ex) { if (e) *e = ex.what(); return false; }
        if (!roots.contains("samples") || !roots["samples"].is_array()) { if (e) *e = "no samples[]"; return false; }
        int ns = 0;
        for (const auto & s : roots["samples"]) {
            kev::ParsedSystemOne ps;
            std::string pe;
            if (!kev::parse_systemone(s, ps, &pe)) { std::fprintf(stderr, "  [%s] sample %d parse: %s\n", path.c_str(), ns, pe.c_str()); ++ns; continue; }
            std::vector<std::vector<float>> z;
            std::string le;
            if (!m.raw_logits(ps.rec, z, &le)) { std::fprintf(stderr, "  [%s] sample %d logits: %s\n", path.c_str(), ns, le.c_str()); ++ns; continue; }
            const json & qs = s["questions"];
            size_t qi = 0;
            for (auto it = qs.begin(); it != qs.end(); ++it, ++qi) {
                if (qi >= z.size() || qi >= ps.meta.size()) break;
                DP d; d.K = (int) z[qi].size(); d.z = z[qi]; d.true_idx = -1;
                std::string ans;
                if (it.value().is_object() && it.value().contains("answer")) ans = it.value()["answer"].get<std::string>();
                const auto & M = ps.meta[qi];
                for (size_t k = 0; k < M.keys.size(); ++k) if (M.keys[k] == ans) { d.true_idx = (int) k; break; }
                if (d.true_idx < 0) { ++ns; continue; } // 无标签则跳过
                out.push_back(d);
            }
            ++ns;
        }
        return true;
    };

    auto report = [&](const char * name, const std::vector<DP> & dps, const kev::LogitsCalibrator & cal,
                      float base_t, float drop, const char * cfg_label) {
        int dropped = 0;
        Agg a = evaluate(dps, cal, base_t, drop, &dropped);
        std::printf("  %-34s  n=%3d  acc=%.3f  conf=%.3f  confCorr=%.3f  confWr=%.3f  ECE=%.3f  drop=%d\n",
                    cfg_label, a.n, a.acc, a.mean_conf, a.mean_conf_correct, a.mean_conf_wrong, a.ece, dropped);
    };

    std::string e1, e2;
    if (!load_exact(calib_file, calib_dps, &e1)) { std::fprintf(stderr, "calib load failed: %s\n", e1.c_str()); return 1; }
    if (!load_exact(test_file, test_dps, &e2)) { std::fprintf(stderr, "test load failed: %s\n", e2.c_str()); return 1; }
    std::fprintf(stderr, "collected calib=%zu dps, test=%zu dps\n", calib_dps.size(), test_dps.size());
    if (calib_dps.empty() || test_dps.empty()) {
        std::fprintf(stderr, "error: need >=1 labeled data point in BOTH calib and test sets (labels via per-question 'answer')\n");
        return 1;
    }

    // ================= 校准集基线(未校准) =================
    std::printf("=== 校准集基线 (raw logits, 校准关闭 = softmax(z/%.3f)) ===\n", base_temp);
    kev::LogitsCalibrator OFF; OFF.enabled = false;
    report("calib baseline", calib_dps, OFF, base_temp, 0.0f, "baseline(z/T)");

    // ================= 位置/顺序偏置检测 =================
    // 对每条做中心化(减该问候选 logits 均值), 再按候选顺序位置求均值 = 经验位置偏置。
    int maxpos = 0; for (auto & d : calib_dps) maxpos = std::max(maxpos, d.K);
    std::vector<float> pos_bias(maxpos + 1, 0.0f); std::vector<int> pos_n(maxpos + 1, 0);
    for (auto & d : calib_dps) {
        double mu = 0; for (float x : d.z) mu += x; mu /= (double) d.K;
        for (int j = 0; j < d.K; ++j) { pos_bias[j] += (float)(d.z[j] - mu); pos_n[j]++; }
    }
    for (int j = 0; j <= maxpos; ++j) if (pos_n[j]) pos_bias[j] /= (float) pos_n[j];
    std::printf("\n=== 位置顺序偏置检测 (校准集, 中心化候选 logits 按位置均值) ===\n");
    {
        double sumabs = 0; for (int j = 0; j <= maxpos; ++j) sumabs += std::fabs((double) pos_bias[j]);
        std::printf("  maxpos=%d  mean|bias|=%.4f\n", maxpos, maxpos > 0 ? (sumabs / (maxpos + 1)) : 0);
        for (int j = 0; j <= maxpos; ++j) std::printf("    pos[%d] bias=%+.4f (n=%d)\n", j, pos_bias[j], pos_n[j]);
    }

    // ================= 温度网格搜索(校准集) =================
    const float grid[] = {0.3f, 0.5f, 0.7f, 1.0f, 1.5f, 2.0f, 3.0f};
    std::printf("\n=== 温度网格搜索 (校准集; same T 单档 = 全局, +debias = 去位置偏置) ===\n");
    float best_acc_single = -1, best_acc_singleD = -1;
    float best_ece_single = 1e9, best_ece_singleD = 1e9;
    float best_ece_singleT = 1e9;   // 最低 ECE 的单档(不去偏, 置信更诚实)方案
    kev::LogitsCalibrator best_single, best_singleD, best_singleT_cal;
    for (size_t i = 0; i < sizeof(grid) / sizeof(grid[0]); ++i) {
        float T = grid[i];
        kev::LogitsCalibrator cS; cS.enabled = true; cS.temperature_mode = "single"; cS.single_T = T;
        kev::LogitsCalibrator cP = cS; cP.position_bias = pos_bias;
        Agg aS = evaluate(calib_dps, cS, base_temp, 0.0f);
        Agg aD = evaluate(calib_dps, cP, base_temp, 0.0f);
        std::printf("  T=%4.2f  single: acc=%.3f ECE=%.3f conf=%.3f | +debias: acc=%.3f ECE=%.3f conf=%.3f\n",
                    T, aS.acc, aS.ece, aS.mean_conf, aD.acc, aD.ece, aD.mean_conf);
        if (aS.acc > best_acc_single || (aS.acc == best_acc_single && aS.ece < best_ece_single)) { best_acc_single = (float) aS.acc; best_ece_single = (float) aS.ece; best_single = cS; }
        if (aD.acc > best_acc_singleD || (aD.acc == best_acc_singleD && aD.ece < best_ece_singleD)) { best_acc_singleD = (float) aD.acc; best_ece_singleD = (float) aD.ece; best_singleD = cP; }
        if (aS.ece < best_ece_singleT) { best_ece_singleT = (float) aS.ece; best_singleT_cal = cS; }
    }
    // by_n 分桶: 按候选数 K 分别网格选 T
    std::vector<int> ks; for (auto & d : calib_dps) if (std::find(ks.begin(), ks.end(), d.K) == ks.end()) ks.push_back(d.K);
    std::sort(ks.begin(), ks.end());
    kev::LogitsCalibrator byN; byN.enabled = true; byN.temperature_mode = "by_n";
    byN.by_n_T.assign(maxpos + 1, 0.0f);
    std::printf("  by_n 分桶(按候选数 K 逐档网格选 T; 下行为各 K 档最佳):\n");
    for (int K : ks) {
        std::vector<DP> sub;
        for (auto & d : calib_dps) if (d.K == K) sub.push_back(d);
        if (sub.empty()) continue;
        float bAcc = -1, bT = base_temp, bEce = 1e9;
        for (size_t i = 0; i < sizeof(grid) / sizeof(grid[0]); ++i) {
            kev::LogitsCalibrator c; c.enabled = true; c.single_T = grid[i];
            Agg a = evaluate(sub, c, base_temp, 0.0f);
            if (a.acc > bAcc || (a.acc == bAcc && a.ece < bEce)) { bAcc = (float) a.acc; bT = grid[i]; bEce = (float) a.ece; }
        }
        byN.by_n_T[K] = bT;
        std::printf("    K=%d  best T=%4.2f  (acc=%.3f ece=%.3f, n=%zu)\n", K, bT, bAcc, (float) bEce,
                    (size_t) sub.size());
    }
    Agg aByN = evaluate(calib_dps, byN, base_temp, 0.0f);
    kev::LogitsCalibrator byNpos = byN; byNpos.position_bias = pos_bias;
    Agg aByNpos = evaluate(calib_dps, byNpos, base_temp, 0.0f);
    std::printf("  by_n(全考校准集): acc=%.3f ECE=%.3f | by_n+debias: acc=%.3f ECE=%.3f\n",
                aByN.acc, aByN.ece, aByNpos.acc, aByNpos.ece);

    // ============ 在校准集上选"最合理"方案(主指标=acc, 伴随 ECE) ============
    // 候选: single / single+debias / by_n / by_n+debias; 取 acc 最高, 平手取 ECE 更低。
    std::string chosen_mode; kev::LogitsCalibrator chosen; float chosen_acc, chosen_ece;
    auto consider = [&](const kev::LogitsCalibrator & c, const std::string & mode, float acc, float ece) {
        if (chosen_mode.empty() || (acc > chosen_acc + 1e-9) || (std::fabs(acc - chosen_acc) <= 1e-9 && ece < chosen_ece)) {
            chosen = c; chosen_mode = mode; chosen_acc = acc; chosen_ece = ece;
        }
    };
    consider(best_single, "single", best_acc_single, best_ece_single);
    consider(best_singleD, "single+debias", best_acc_singleD, best_ece_singleD);
    consider(byN, "by_n", (float) aByN.acc, (float) aByN.ece);
    consider(byNpos, "by_n+debias", (float) aByNpos.acc, (float) aByNpos.ece);
    std::printf("  校准集选定方案: %s  (acc=%.3f ECE=%.3f)\n", chosen_mode.c_str(), chosen_acc, chosen_ece);
    float fitted_drop = 0.0f;

    // ================= 低置信阈值搜索(校准集, 用选定方案) =================
    {
        std::printf("\n=== 低置信拒绝阈值搜索 (校准集, 方案=%s) ===\n", chosen_mode.c_str());
        const float thr[] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        float best_thr_v = 0.0f, best_thr_acc = -1, best_thr_ece = 1e9; int best_kept = (int) calib_dps.size();
        for (size_t i = 0; i < sizeof(thr) / sizeof(thr[0]); ++i) {
            int dropped = 0;
            Agg a = evaluate(calib_dps, chosen, base_temp, thr[i], &dropped);
            int kept = a.n_correct + a.n_wrong;
            std::printf("  thr=%.2f  kept=%d  drop=%d  acc(kept)=%.3f  mean_conf(kept)=%.3f  ECE=%.3f\n",
                        thr[i], kept, dropped, a.acc, a.mean_conf, a.ece);
            // 选"不降 acc 前提下最低阈值"(默认 0 = 不拒绝), 优先让不能保证赢的别硬猜只用于报告。
            if (thr[i] == 0.0f) { best_thr_v = 0.0f; best_thr_acc = (float) a.acc; best_thr_ece = (float) a.ece; best_kept = kept; }
            else if (a.acc >= best_thr_acc - 1e-9 && a.ece < best_thr_ece) { best_thr_v = thr[i]; best_thr_acc = (float) a.acc; best_thr_ece = (float) a.ece; }
        }
        std::printf("  选定阈值: drop_low_conf=%.2f  (kept=%d/calib)\n", best_thr_v, best_kept);
        fitted_drop = best_thr_v; chosen_ece = best_thr_ece;
    }

    // ================= 固定测试集回归 (校准前 vs 校准后) =================
    std::printf("\n=== 固定测试集回归 (与校准集不重叠) ===\n");
    std::printf("  --- 校准前 (未校准, softmax(z/%.3f)) ---\n", base_temp);
    report("test baseline", test_dps, OFF, base_temp, 0.0f, "before: softmax(z/T)");
    {
        int mv_ = 0; for (auto & d : test_dps) mv_ = std::max(mv_, d.K);
        Agg a = evaluate(test_dps, OFF, base_temp, 0.0f);
        std::printf("  校准前逐位置准确率:");
        for (int j = 0; j <= mv_; ++j) std::printf(" pos[%d]=%.2f", j, a.pos_acc[j]);
        std::printf("\n");
    }
    std::printf("  --- 校准后 (方案=%s, drop=%.2f) ---\n", chosen_mode.c_str(), fitted_drop);
    report("test after(temp)", test_dps, chosen, base_temp, 0.0f, "after_temp");
    if (fitted_drop > 0.0f)
        report("test after(+drop)", test_dps, chosen, base_temp, fitted_drop, "after_temp+drop");
    // 推荐的'置信更诚实'方案(隐含): 最低 ECE 单档 T(不去偏; 温度不改 argmax, 故 acc 不变)
    {
        std::printf("  --- 校准后'置信更诚实'方案 (单档 T=%.2f, 不位置去偏; 置信更贴真实正确率) ---\n",
                    best_singleT_cal.single_T);
        report("test honest-conf", test_dps, best_singleT_cal, base_temp, 0.0f, "after_honestT");
        report("test honest-conf+drop", test_dps, best_singleT_cal, base_temp, 0.30f, "after_honestT+drop");
    }

    // ================= 输出拟合参数 =================
    kev::LogitsCalibrator fitted = chosen; fitted.drop_low_conf = fitted_drop;
    if (!out_file.empty()) {
        std::string perr;
        if (!fitted.save(out_file, &perr)) std::fprintf(stderr, "write params failed: %s\n", perr.c_str());
        else std::fprintf(stderr, "fitted params -> %s\n", out_file.c_str());
    }
    std::printf("\n=== 推荐配置片段 (决策 config JSON; params_file=%s) ===\n",
                out_file.empty() ? "(未写盘)" : out_file.c_str());
    std::printf("{\n  \"calib\": {\n    \"enable\": true,\n    \"temperature_mode\": \"%s\",\n",
                chosen.temperature_mode.c_str());
    if (chosen.temperature_mode == "single")
        std::printf("    \"temperature\": %.3f,\n", chosen.single_T);
    std::printf("    \"position_debias\": %s,\n    \"drop_low_conf\": %.2f,\n    \"params_file\": \"%s\"\n  }\n}\n",
                bool_str(!chosen.position_bias.empty()), fitted_drop, out_file.empty() ? "" : out_file.c_str());

    std::printf("\n=== 汇总 ===\n");
    std::printf("校准集: baseline acc=%.3f | 选定方案=%s acc(校准集)=%.3f\n",
                (float) evaluate(calib_dps, OFF, base_temp, 0.0f).acc, chosen_mode.c_str(), chosen_acc);
    std::printf("固定测试集: before acc=%.3f ; after(%s) acc=%.3f  (见上表)\n",
                (float) evaluate(test_dps, OFF, base_temp, 0.0f).acc, chosen_mode.c_str(),
                (float) evaluate(test_dps, chosen, base_temp, fitted_drop).acc);
    return 0;
}