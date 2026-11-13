# Upstream provenance

These files are copied without modification from llama.cpp revision
`3d65c90d04d337e88f2b1f7f0061f40a5324e662`:

- `common/jinja/` -> `jinja/`
- `common/json.cpp`, `common/json.h`, `common/unicode.cpp`, `common/unicode.h`
- `LICENSE`

They are distributed under the MIT license in `LICENSE`. The build uses the
already-vendored nlohmann/json header and a small local assertion shim in
`src/kev_jinja_compat/ggml.h`; it does not link llama or ggml into the host.
