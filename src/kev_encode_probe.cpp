#include "kev_encode.h"

#include <cstdio>
#include <limits>

namespace {
struct State {
    int calls = 0;
    int mode = 0;
};

int32_t fake_tokenize(void * ud, const char *, int32_t, bool, bool, int32_t * out, int32_t cap) {
    auto & state = *static_cast<State *>(ud);
    ++state.calls;
    if (state.mode == 0) return -1;
    if (state.mode == 1) return std::numeric_limits<int32_t>::min();
    if (state.mode == 2) return cap + 1;
    if (state.mode == 4) return -(cap + 1);
    if (state.calls == 1) return -(cap + 1);
    out[0] = 42;
    return 1;
}
}

int main() {
    State state;
    kev::TokenFeeder feeder{};
    feeder.ud = &state;
    feeder.tokenize = fake_tokenize;
    for (int mode = 0; mode < 5; ++mode) {
        state = {0, mode};
        const auto tokens = kev::tokenize(feeder, "test", false, false);
        const bool valid = mode == 3 ? state.calls == 2 && tokens.size() == 1 && tokens[0] == 42
                           : state.calls == (mode == 4 ? 2 : 1) && tokens.empty();
        if (!valid) {
            std::fprintf(stderr, "tokenize regression in mode %d, calls=%d\n", mode, state.calls);
            return 1;
        }
    }
    std::puts("PASS tokenizer failure and capacity handling");
    return 0;
}
