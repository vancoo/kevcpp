// kev_api.h — TypeSafe 兼容的 /v1/systemone 请求/响应映射层 (M4)
//
// 复刻 kev/api.py 的 render/option_text/question_keys/to_record/to_answers/confidence 语义,
// 并用 kev_model 组装: 解析 SystemOneRequest JSON -> KevRequest(进 kev_encode) -> model.probs
// -> to_answers -> 响应 JSON {model, answers, usage, latency_ms}.
#pragma once

#include "kev_model.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace kev {

struct SystemOneMeta {
    std::string id;
    std::string type;                 // "noul" | "choice" | "score"
    std::vector<std::string> keys;    // question_keys: noul=["false","true"]; choice=criteria names; score=[str(i)]
    std::vector<std::string> legend;  // score 才有: 每级文本
};

struct ParsedSystemOne {
    KevRequest rec;                     // 给 kev_encode
    std::vector<SystemOneMeta> meta;    // 每问映射回上文
};

// 解析 POST body(SystemOneRequest) -> (KevRequest, meta)。失败返回 false 并置 err。
bool parse_systemone(const nlohmann::json & body, ParsedSystemOne & out, std::string * err);

// to_answers: probs[Q][K] + meta -> 每问答案 dict(见 api.py)。
nlohmann::json to_answers(const std::vector<std::vector<float>> & probs,
                          const std::vector<SystemOneMeta> & meta);

// 一次 /v1/systemone 服务: 解析->probs->answers->响应 JSON。
// t_model_ms 由调用方传入(记录模型耗时)。err 非空表示 422。
nlohmann::json systemone_json(KevModel & model, const nlohmann::json & body,
                              double * t_model_ms, std::string * err);

// 仅供诊断/测试暴露的 render(flatten JSON 到文本)
std::string render(const nlohmann::json & v);

} // namespace kev