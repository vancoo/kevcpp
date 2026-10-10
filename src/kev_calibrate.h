// kev_calibrate.h — 方案3: logits 通道「决策映射层」轻量校准 (无训练权重, 模型无关)
//
// 只调"候选 logits -> 概率"的映射, 不触碰模型权重 / 不改 llama 前向。校准对象是
//   对每问: 候选 key 首 token 的原始 logits z[K] -> softmax 概率。
//   现状(校准关闭): probs = softmax(z / decision.temperature)。
//   可选校准:
//     * 温度桶     temperature_mode "single"(全局一档) / "by_n"(按候选数 n 分桶 T[n])
//     * 位置/顺序 去偏  position_debias: probs 用 z[j] - position_bias[j] 代替 z[j], 矫
//                  正弱模型下"候选在前/在后"的顺序脏偏置(position_bias 由校准集拟合)
//     * 低置信拒绝 drop_low_conf: argmax 归一化置信不足时拒绝硬选(决策层, 不改 probs)
//   所有校准参数默认=关闭/原始行为(可回退), 且与某具体模型词表/标签无关。
#pragma once

#include <string>
#include <vector>

namespace kev {

// 候选 logits 的映射层校准器(默认空 = 恒等)。模型无关。
class LogitsCalibrator {
public:
    bool enabled = false;
    std::string temperature_mode = "single";  // "single" | "by_n"
    float single_T = 1.0f;                    // single 模式下有效温度
    std::vector<float> by_n_T;                // [n_options] 分桶温度(索引=候选数); 缺省回退 single_T
    std::vector<float> position_bias;         // [P] 每位置经验偏置(索引=候选顺序位置)
    float drop_low_conf = 0.0f;               // 0 = 关闭低置信拒绝

    // 该问候选数为 n 时实际生效的 softmax 温度(>0; 防御回退 1.0)。
    float effective_temperature(int n) const;

    // 应用校准映射: 传入原始候选 logits z[K] 与候选数 K, 输出校准后概率 out[K]。
    // position_debias 由外层是否提供 position_bias 决定; 若 position_bias 非空且 >=K,
    // 会对 z[j] 减 position_bias[j]。除以 effective_temperature(K) 后做数值稳定 softmax。
    // 当 disabled(或非模式默认)时等价于当前行为: softmax(z / single_T)。
    void apply(const std::vector<float> & z, std::vector<float> & probs_out) const;

    // 归一化 argmax 置信(max-margin 于均匀): 与 api.choice_confidence 同一量纲 [0,1]。
    static float normalized_confidence(const std::vector<float> & probs);
    static int argmax(const std::vector<float> & v);

    // 参数文件读写(模型无关的轻量文本; 供校准集拟合后落盘, 运行期加载)。
    // 格式: 每行 "key value"; 支持:
    //   enabled 0|1 | temperature_mode single|by_n | single_T f | drop_low_conf f
    //   by_n T <n>=<val> (每行一个桶) | position_bias <j>=<val> (每行一个)
    bool load(const std::string & path, std::string * err = nullptr);
    bool save(const std::string & path, std::string * err = nullptr) const;

    // 数值稳定 softmax(除 T): 与 probs_logits 的 softmax_scaled 语义一致, 供汇总/评测复用。
    static void softmax_scaled(const std::vector<float> & z, float temperature,
                               std::vector<float> & out);
};

} // namespace kev