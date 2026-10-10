// kev_config.h — 决策配置 (模型无关 marker + 温度) 与轻量 JSON 加载
//
// 路线 A 的核心: 不再把 Qwen3.5 的 <|box_start|>/<|box_end|> 等 marker 硬编码进代码路径,
// 而是放到一份"决策配置"(KevDecisionConfig)里, 对任意开源 GGUF 可按模型指定 marker/temperature。
// 原版 Qwen3.5 使用默认配置即可; 换模型改配置即可。可选的 JSON 配置文件(documented 示例见
// config/decision.qwen3.5.json / config/decision.example.json)。
#pragma once

#include <string>

namespace kev {

// 5 个决策 marker 的默认值(原版 Qwen3.5), 仅供默认; 必须可被配置覆盖, 不得在编码路径写死。
// 语义(与 llama.cpp decision 模型一致):
//   state_marker    状态段起始 token (SPECIAL[0]  <|fim_prefix|>)
//   question_marker 每个 question 起始 token (SPECIAL[1] <|fim_middle|>)
//   option_start    每个 option 起始 token (SPECIAL[2] <|box_start|>)
//   option_end      "收尾/结束" marker: option 结束位置即 KEV 打分的 marker 位置 (SPECIAL[3] <|box_end|>)
//   decide_marker   decide 收尾 marker: question 的 <decide> 位置隐层即为 KEV 的 pointer (SPECIAL[4] <|fim_suffix|>)
struct KevDecisionConfig {
    // ---- 旧 marker / 温度 (KEV 回退通道) ----
    std::string state_marker       = "<|fim_prefix|>";
    std::string question_marker    = "<|fim_middle|>";
    std::string option_start       = "<|box_start|>";
    std::string option_end         = "<|box_end|>";
    std::string decide_marker      = "<|fim_suffix|>";
    float       temperature        = 1.0f;

    // ---- 生成式 logits 判别主通道 (路线 D / 已实测验证) ----
    // enable_logits: true = 主决策通道用「填空式 prompt + 候选答案 token 的 logits + softmax」;
    //                false = 回退到旧 KEV 半边点积打分(兼容旧行为)。
    bool enable_logits = true;
    // 填空式 prompt 模板(模型无关, 每问拼装)。可选占位符:
    //   {state}        用户 state 文本(原始上下文)
    //   {instructions} 该 question 的 instructions
    //   {options}      该 question 各候选 option 文本, 以 ", " 连接(choice/noul: "key: desc"; score: 等级文本)
    // 默认与验证探针一致: 先给上下文/指令, 再列选项, 末尾 "Answer:" 提示续写候选答案 token。
    // **不要用"列出选项再让模型选 A/B"式 prompt**——实测会引入位置偏差。
    std::string prompt_template = "{state}\n{instructions}\n[ {options} ]\nAnswer:";
    // 多 token 候选的聚合策略:
    //   "first" = 只取该候选 key 的第一个 token 的 logit(与验证结果一致, 天然无位置偏置)
    //   "sum"   = 对候选 key 的全部 token 的 logit 求和
    //   "mean"  = 对候选 key 的全部 token 的 logit 求平均
    std::string aggregation = "first";
    // 每问判别的候选来源("logits_type" / 每题覆盖):
    //   ""      = 自动(choice -> 各 criteria key; noul -> false/true; score -> 0..L-1)
    //   "choice"|"noul"|"score" 显式覆盖(模型无关; 通常留 "" 即可)
    std::string logits_type = "";
};

// 默认决策配置(原版 Qwen3.5)。
inline KevDecisionConfig default_decision_config() { return KevDecisionConfig{}; }

// 从 JSON 配置文件加载决策配置。schema 见 config/decision.example.json:
//   {
//     "markers": { "state":"...", "question":"...", "option_start":"...",
//                  "option_end":"...", "decide":"..." },
//     "temperature": 1.0,
//     "logits": {
//        "enable": true,
//        "prompt_template": "...",
//        "aggregation": "first",
//        "logits_type": ""
//     }
//   }
// 缺省字段沿用默认值。返回 false 且 *err 说明原因(文件打不开 / JSON 语法错 / 某字段类型错)。
// 轻量、自包含的极小 JSON 对象解析器(无第三方依赖), 仅供决策配置使用。
bool load_decision_config(const std::string & file, KevDecisionConfig & cfg, std::string * err = nullptr);

} // namespace kev