#!/usr/bin/env python
"""Generate/export PointerHead weights to the kevcpp .bin format (magic 'KEVH').

The C++ side (src/kev_head.cpp) loads only this format, so we avoid libtorch.

.bin layout (little-endian):
    char  magic[4]      = b'KEVH'
    i32   dim_in        (d, 1024)
    i32   dim_proj      (dp, 256)
    f32   temperature
    f32   Wq[dp][dim_in]  row-major  (nn.Linear.weight)
    f32   bq[dp]
    f32   Wk[dp][dim_in]
    f32   bk[dp]

Usage:
    python gen_head_weights.py --out head.bin [--head head.pt] [--seed N]
  - with --head <!-- a torch-saved head.pt > --: read q.weight/q.bias/k.weight/k.bias/temperature
  - without: generate a random head (--seed) so the chain runs stand-alone
"""
import argparse
import os
import struct

def head_bytes(q_weight, q_bias, k_weight, k_bias, temperature):
    dp, dim = q_weight.shape
    assert q_bias.shape == (dp,) and k_weight.shape == (dp, dim) and k_bias.shape == (dp,)
    def flat(m):
        return m.reshape(-1).tolist()
    buf = bytearray()
    buf += b"KEVH"
    buf += struct.pack("<ii", dim, dp)
    buf += struct.pack("<f", float(temperature))
    buf += struct.pack("<%df" % (dp * dim), *flat(q_weight))
    buf += struct.pack("<%df" % dp, *flat(q_bias))
    buf += struct.pack("<%df" % (dp * dim), *flat(k_weight))
    buf += struct.pack("<%df" % dp, *flat(k_bias))
    return bytes(buf)

def write_head_bin(path, q_weight, q_bias, k_weight, k_bias, temperature=1.0):
    with open(path, "wb") as f:
        f.write(head_bytes(q_weight, q_bias, k_weight, k_bias, temperature))

def from_torch(meta):
    """meta = torch.load(head.pt) dict; head.pt layout defined in kev.checkpoint.Meta."""
    import torch
    head = meta["head"]            # PointerHead.state_dict(): q.weight, q.bias, k.weight, k.bias
    wq = head["q.weight"].to(torch.float32)
    bq = head["q.bias"].to(torch.float32)
    wk = head["k.weight"].to(torch.float32)
    bk = head["k.bias"].to(torch.float32)
    temperature = float(meta.get("temperature", 1.0))
    return wq, bq, wk, bk, temperature

def random_head(dp=256, dim=1024, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((dp, dim)).astype(np.float32),
            rng.standard_normal((dp,)).astype(np.float32),
            rng.standard_normal((dp, dim)).astype(np.float32),
            rng.standard_normal((dp,)).astype(np.float32))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--head", help="path to a torch head.pt to export (else random)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--dp", type=int, default=256)
    a = ap.parse_args()
    if a.head and os.path.isfile(a.head):
        import torch
        wq, bq, wk, bk, temp = from_torch(torch.load(a.head, map_location="cpu"))
    else:
        wq, bq, wk, bk = random_head(dp=a.dp, dim=a.dim, seed=a.seed)
        temp = 1.0
    write_head_bin(a.out, wq, bq, wk, bk, temp)
    print(f"wrote {a.out}: dim={wq.shape[1]} dp={wq.shape[0]} temp={temp} "
          f"(source={'head.pt' if (a.head and os.path.isfile(a.head)) else 'random seed=%d' % a.seed})")