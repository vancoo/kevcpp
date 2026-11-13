#pragma once

#include <string>

namespace kev {

// Throws on parse/render failure. Does not alter the source template.
std::string render_gguf_chat_template(const std::string & source,
                                      const std::string & bos, const std::string & eos,
                                      const std::string & system, const std::string & user);

}
