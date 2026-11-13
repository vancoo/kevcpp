#include "kev_chat_template.h"

#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/value.h"

#include <stdexcept>

namespace kev {

std::string render_gguf_chat_template(const std::string & source,
                                      const std::string & bos, const std::string & eos,
                                      const std::string & system, const std::string & user) {
    if (source.empty()) throw std::runtime_error("model has no default GGUF chat template");
    jinja::lexer lexer;
    auto tokens = lexer.tokenize(source);
    auto program = jinja::parse_from_tokens(tokens);
    jinja::context ctx(tokens.source);
    auto messages = jinja::mk_val<jinja::value_array>();
    auto add_message = [&](const char * role, const std::string & content) {
        auto message = jinja::mk_val<jinja::value_object>();
        message->insert("role", jinja::mk_val<jinja::value_string>(std::string(role)));
        message->insert("content", jinja::mk_val<jinja::value_string>(content));
        messages->push_back(message);
    };
    add_message("system", system);
    add_message("user", user);
    ctx.set_val("messages", messages);
    ctx.set_val("bos_token", jinja::mk_val<jinja::value_string>(bos));
    ctx.set_val("eos_token", jinja::mk_val<jinja::value_string>(eos));
    ctx.set_val("enable_thinking", jinja::mk_val<jinja::value_bool>(false));
    ctx.set_val("add_generation_prompt", jinja::mk_val<jinja::value_bool>(true));
    jinja::runtime runtime(ctx);
    auto result = runtime.execute(program);
    return jinja::runtime::gather_string_parts(result)->as_string().str();
}

}
