"""Benchmark the real Qwen3.5-0.8B bf16 trunk on this CPU (AMD Ryzen AI MAX+ PRO 395).
Measures (a) single-token decode ms (apples-to-apples with kevcpp ~2.9ms llama_decode),
(b) a 161-token forward total ms, and reports implied read bandwidth from the real
bf16 trunk byte count. New file, does not touch kev code.

Run: D:\\anaconda3\\envs\\proxy\\python.exe bench_bf16_cpu.py
"""
import os, time, torch
from transformers import AutoModelForCausalLM

MODEL = r"D:\share-models\qwen3.5-0.8b"   # real Qwen3.5-0.8B base (bf16 safetensors)
N_THREADS = int(os.environ.get("N_THREADS", "32"))
DECODE_TOKENS = 20      # single-token decodes averaged (best-of)
PREFIX_TOKENS = 1000    # warm KV cache context for the decode measure

torch.set_num_threads(N_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

def best_of(fn, n):
    times = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); times.append(time.perf_counter() - t0)
    return min(times), times

def main():
    print(f"torch {torch.__version__}  threads={torch.get_num_threads()} numcpus={os.cpu_count()}", flush=True)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager",
        device_map={"": "cpu"}).model.eval()
    cfg = m.config
    dtype = next(m.parameters()).dtype
    emb = sum(p.numel() * p.element_size() for name, p in m.named_parameters()
              if "embed" in name)
    total = sum(p.numel() * p.element_size() for p in m.parameters())
    other = total - emb
    print(f"hidden={cfg.hidden_size} head_dim={getattr(cfg,'head_dim',None)} "
          f"dtype={dtype}  params={sum(p.numel() for p in m.parameters())/1e6:.1f}M", flush=True)
    print(f"BF16 total bytes: {total/2**20:.1f} MiB | embeddings {emb/2**20:.1f} MiB | trunk(non-emb) {other/2**20:.1f} MiB", flush=True)

    ids = torch.randint(10, 1000, (1, 1))
    with torch.no_grad():
        _ = m(input_ids=ids)   # build graph / warm kernels

    # ---- (a) single-token decode with warm KV cache ----
    # hybrid backend needs a HybridCache; let the model build one (returns it on outputs).
    prefix = torch.randint(10, 1000, (1, PREFIX_TOKENS))
    def pref():
        with torch.no_grad():
            return m(input_ids=prefix, use_cache=True)
    out0 = pref(); out0 = pref()  # warm twice
    cache = out0.past_key_values
    def one_decode():
        nonlocal cache
        with torch.no_grad():
            cache = m(input_ids=ids, past_key_values=cache, use_cache=True).past_key_values
    best, all_times = best_of(one_decode, DECODE_TOKENS)
    ms_per_tok = best * 1e3
    print(f"\n[a] single-token decode (KV cached, prefix={PREFIX_TOKENS}):", flush=True)
    print(f"    best={ms_per_tok:.3f} ms/token   median={sorted(all_times)[len(all_times)//2]*1e3:.3f} ms/token", flush=True)

    # ---- (b) a 161-token forward (prefill-style, no cache) ----
    ids161 = torch.randint(10, 1000, (1, 161))
    def fwd161():
        with torch.no_grad():
            _ = m(input_ids=ids161)
    best161, _ = best_of(fwd161, 8)
    ms161 = best161 * 1e3
    print(f"[b] 161-token forward total: best={ms161:.3f} ms  ({(ms161/161):.3f} ms/token avg)", flush=True)

    # ---- implied bandwidth ----
    gbps_tot = total / (ms_per_tok * 1e-3) / 1e9
    gbps_trunk = other / (ms_per_tok * 1e-3) / 1e9
    gbps_161 = total / (ms161 * 1e-3) / 1e9
    print(f"    implied BW [total] per-token decode: {total/2**20:.1f} MiB/tok -> {gbps_tot:.1f} GB/s", flush=True)
    print(f"    implied BW [trunk] per-token decode: {other/2**20:.1f} MiB/tok -> {gbps_trunk:.1f} GB/s", flush=True)
    print(f"    implied BW [total] over 161-token fwd: {gbps_161:.1f} GB/s", flush=True)

    # ---- (c) batched-decode ms/token vs batch width (GEMM amortization probe) ----
    print("\n[c] single-forward decode, adding B tokens on-top of warm cache:", flush=True)
    row = []
    for B in (1, 2, 4, 8, 16):
        idsB = torch.randint(10, 1000, (1, B))
        c = out0.past_key_values
        def fwdB():
            nonlocal c
            with torch.no_grad():
                c = m(input_ids=idsB, past_key_values=c, use_cache=True).past_key_values
        bestB, _ = best_of(fwdB, 8)
        ms = bestB * 1e3
        row.append((B, ms, ms / B, 950.1 / (ms * 1e-3) / 1e9))
        print(f"    B={B:2d}: {ms:8.3f} ms total  {ms/B:6.3f} ms/token  implied_trunk_bw={950.1/(ms*1e-3)/1e9:6.1f} GB/s (trunk once)", flush=True)

    with open(r"E:\van\projects\kevcpp\docs\BF16_BENCH_RESULT.txt", "w") as f:
        f.write(f"threads={N_THREADS} torch={torch.__version__}\n")
        f.write(f"bf16_total_MiB={total/2**20:.1f} trunk_MiB={other/2**20:.1f}\n")
        f.write(f"single_token_decode_ms_best={ms_per_tok:.3f} median={sorted(all_times)[len(all_times)//2]*1e3:.3f}\n")
        f.write(f"fwd161_total_ms_best={ms161:.3f} fwd161_implied_BW={gbps_161:.1f}\n")
        f.write(f"single_tok_implied_BW_total={gbps_tot:.1f} trunk={gbps_trunk:.1f}\n")
        f.write("batch_decode_ms_total_per_token: " + " ".join(f"B{B}={ms:.1f}/{ms/B:.3f}" for B, ms, _, _ in row) + "\n")

if __name__ == "__main__":
    main()