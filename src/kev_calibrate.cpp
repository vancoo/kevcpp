// kev_calibrate.cpp — 方案3: logits 决策映射层轻量校准 (无训练权重, 模型无关)
#include "kev_calibrate.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace kev {

namespace {
// 与 probs_logits 内 softmax_scaled 完全一致的数值稳定 softmax。
void softmax_impl(const std::vector<float> & z, float T, std::vector<float> & out) {
    out.resize(z.size());
    if (z.empty()) return;
    float t = T > 0.0f ? T : 1.0f;
    float mx = z[0]; for (float x : z) mx = std::max(mx, x);
    std::vector<double> e(z.size()); double s = 0;
    for (size_t i = 0; i < z.size(); ++i) { e[i] = std::exp((double)((z[i] - mx) / t)); s += e[i]; }
    if (s <= 0) s = 1.0;
    for (size_t i = 0; i < z.size(); ++i) out[i] = (float)(e[i] / s);
}
} // namespace

float LogitsCalibrator::effective_temperature(int n) const {
    if (temperature_mode == "by_n" && n >= 0 && n < (int) by_n_T.size() && by_n_T[n] > 0.0f)
        return by_n_T[n];
    if (enabled && temperature_mode == "by_n" && !by_n_T.empty())
        return by_n_T[std::min(n, (int) by_n_T.size() - 1)];
    return single_T > 0.0f ? single_T : 1.0f;
}

void LogitsCalibrator::apply(const std::vector<float> & z, std::vector<float> & probs_out) const {
    const int K = (int) z.size();
    std::vector<float> w = z;
    if (position_bias.size() >= (size_t) K) {
        for (int j = 0; j < K; ++j) w[j] -= position_bias[j];   // 去位置/顺序偏置
    }
    softmax_impl(w, effective_temperature(K), probs_out);
}

int LogitsCalibrator::argmax(const std::vector<float> & v) {
    if (v.empty()) return -1;
    return (int) (std::max_element(v.begin(), v.end()) - v.begin());
}

void LogitsCalibrator::softmax_scaled(const std::vector<float> & z, float temperature,
                                      std::vector<float> & out) {
    softmax_impl(z, temperature, out);
}

float LogitsCalibrator::normalized_confidence(const std::vector<float> & probs) {
    const int K = (int) probs.size();
    if (K <= 1) return K == 1 ? 1.0f : 0.0f;
    double s = 0; for (float x : probs) s += x;
    if (s <= 0) return 0.0f;
    double mx = 0;
    for (float x : probs) mx = std::max((double) x, mx);
    double km = 1.0 / K;
    double conf = (mx / s - km) / (1.0 - km);
    return (float) std::max(0.0, std::min(1.0, conf));
}

bool LogitsCalibrator::load(const std::string & path, std::string * err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { if (err) *err = "cannot open calib params: " + path; return false; }
    enabled = false; by_n_T.clear(); position_bias.clear(); single_T = 1.0f; drop_low_conf = 0.0f;
    std::string line;
    while (std::getline(f, line)) {
        // 去首尾空白
        size_t b = line.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) continue;
        size_t e = line.find_last_not_of(" \t\r\n");
        line = line.substr(b, e - b + 1);
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string key;
        if (!(iss >> key)) continue;
        if (key == "enabled") { int v = 0; iss >> v; enabled = v != 0; }
        else if (key == "temperature_mode") { iss >> temperature_mode; }
        else if (key == "single_T") { float v = 1.0f; iss >> v; if (v > 0) single_T = v; }
        else if (key == "drop_low_conf") { float v = 0.0f; iss >> v; if (v >= 0) drop_low_conf = v; }
        else if (key == "by_n") {
            std::string rest; std::getline(iss, rest);
            size_t eq = rest.find('=');
            if (eq != std::string::npos) {
                int n = std::atoi(rest.substr(0, eq).c_str());
                float v = (float) std::atof(rest.c_str() + eq + 1);
                if (n >= 0 && v > 0) { if ((int) by_n_T.size() <= n) by_n_T.resize((size_t) n + 1, 0.0f); by_n_T[(size_t) n] = v; }
            }
        }
        else if (key == "position_bias") {
            std::string rest; std::getline(iss, rest);
            size_t eq = rest.find('=');
            if (eq != std::string::npos) {
                int j = std::atoi(rest.substr(0, eq).c_str());
                float v = (float) std::atof(rest.c_str() + eq + 1);
                if (j >= 0) { if ((int) position_bias.size() <= j) position_bias.resize((size_t) j + 1, 0.0f); position_bias[(size_t) j] = v; }
            }
        }
    }
    return true;
}

bool LogitsCalibrator::save(const std::string & path, std::string * err) const {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) { if (err) *err = "cannot write calib params: " + path; return false; }
    f << "# kevcpp logits decision mapping calibration params (model-agnostic)\n";
    f << "enabled " << (enabled ? 1 : 0) << "\n";
    f << "temperature_mode " << temperature_mode << "\n";
    f << "single_T " << single_T << "\n";
    f << "drop_low_conf " << drop_low_conf << "\n";
    for (size_t n = 0; n < by_n_T.size(); ++n)
        if (by_n_T[n] > 0.0f) f << "by_n " << n << "=" << by_n_T[n] << "\n";
    for (size_t j = 0; j < position_bias.size(); ++j)
        f << "position_bias " << j << "=" << position_bias[j] << "\n";
    return true;
}

} // namespace kev