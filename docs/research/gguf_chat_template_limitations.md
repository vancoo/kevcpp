# GGUF chat template limitations

The experimental binary scoring path renders the loaded model's default GGUF
`tokenizer.chat_template`. It is not a general-purpose, untrusted-input-safe
chat implementation. In particular:

- The rendered prompt is tokenized with special-token parsing enabled. Request
  content containing model control-token strings can therefore introduce token
  boundaries. Do not expose this path to untrusted input without an independent
  protective boundary. Natural-language prompt injection is also not prevented.
- The vendored Jinja implementation does not support every template, and source
  attribution is not preserved by every operation. Naively tokenizing its
  attributed fragments independently can also change token IDs at boundaries.
- `add_special` is disabled. Models that depend on automatic BOS insertion or
  use a different generation prefix require model-specific token-ID validation.
- Binary scoring requires distinct single-token `yes` and `no` labels. A model's
  template may reject the system role or may not honor `enable_thinking=false`;
  successful rendering alone does not establish valid answer-position scoring.

The Qwen3.5 2B and 4B template probes check a representative rendered string,
not independent full token-ID equivalence or security against malicious input.
Missing and malformed templates fail explicitly; this does not guarantee that
all unsupported templates will be detected.
