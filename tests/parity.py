#!/usr/bin/env python
"""M1 对拍: C++ PointerHead <-> Python/numpy 参考。

工作流 (全部走文件, 不用 subprocess 管道捕获, 规避沙箱对命名管道的 EPERM):
  0. 生成随机 head -> head.bin (C++ 读) 并存 numpy 副本.
  1. headbench(纯数学对拍): 随机生成若干 (decide, opts) 隐藏向量用例,
     numpy 按 kev.model.PointerHead 公式算参考概率 -> cases.json;
     调 C++ headbench 用同一 head+向量算概率并与参考比较 -> 报 max|dp|/argmax.
  2. readout(端到端): 用 HF fast tokenizer 按 kev encode 布局构造一个真实小请求
     (state + 一个 question + K 个 options), 得到精确 ids 与 decide/opt 绝对索引 -> req.json;
     调 C++ headbench --mode readout: llama.cpp prefill -> 取 hidden -> 过同一 head 得
     probs.json, 并导出 hidden.json; 再用 numpy 用同一 head 在这些 hidden 上独立重算
     参考概率, 与 probs.json 比较 -> 报 max|dp|/argmax.

Python 参考公式 (与 model.py PointerHead 一致; nn.Linear: y = x @ W^T + b):
    kv = opts @ Wk.T + bk       # [K, dp]
    qv = decide @ Wq.T + bq     # [dp]
    z  = (kv @ qv) * scale      # [K], scale = 1/sqrt(dp)
    p  = softmax(z / T), T=1.0
"""
import argparse, json, os, subprocess, sys

import numpy as np
from gen_head_weights import write_head_bin

DPI = 0  # current directory marker (we chdir into tests/)

def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def head_ref(opts, decide, Wq, bq, Wk, bk, T=1.0):
    scale = 1.0 / np.sqrt(Wq.shape[0])
    kv = opts @ Wk.T + bk
    qv = decide @ Wq.T + bq
    z = (kv @ qv) / T * scale
    return softmax(z)

def run_cpp(argv):
    """Run a C++ binary, stdout/stderr to temp files (no pipe capture: sandbox EPERM)."""
    proj = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(proj)
    exe = os.path.join(root, "build", "bin", "Release", "headbench.exe")
    if not os.path.exists(exe):
        raise SystemExit(f"missing {exe} - build first (scripts/build.ps1)")
    stdout = os.path.join(proj, "golden", "_cpp_stdout.log")
    stderr = os.path.join(proj, "golden", "_cpp_stderr.log")
    os.makedirs(os.path.dirname(stdout), exist_ok=True)
    with open(stdout, "wb") as so, open(stderr, "wb") as se:
        r = subprocess.run([exe] + argv, stdout=so, stderr=se)
    return r.returncode, stdout, stderr

def argmax(a):
    return int(np.argmax(a))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=os.path.join(os.path.dirname(__file__), "golden"))
    ap.add_argument("--model", default=r"D:\share-models\Qwen3.5-0.8B-Q8_0.gguf")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.golden, exist_ok=True)
    g = lambda n: os.path.join(a.golden, n)

    DIM, DP = 1024, 256
    rng = np.random.default_rng(a.seed)
    Wq = rng.standard_normal((DP, DIM)).astype(np.float32)
    bq = rng.standard_normal((DP,)).astype(np.float32)
    Wk = rng.standard_normal((DP, DIM)).astype(np.float32)
    bk = rng.standard_normal((DP,)).astype(np.float32)
    T = 1.0

    # 0) head .bin (C++读)
    head_bin = g("head.bin")
    write_head_bin(head_bin, Wq, bq, Wk, bk, T)
    np.savez(g("head_weights.npz"), Wq=Wq, bq=bq, Wk=Wk, bk=bk, temperature=np.float32(T))

    # ---------- 1) headbench: 纯 PointerHead 数学对拍 ----------
    print("== 1) headbench (pure PointerHead math parity) ==")
    cases = []
    for c in range(8):
        decide = (rng.standard_normal(DIM) * 4).astype(np.float32)   # 模拟真实 scale 的 hidden
        K = int(rng.integers(2, 5))
        opts = (rng.standard_normal((K, DIM)) * 4).astype(np.float32)
        ref = head_ref(opts, decide, Wq, bq, Wk, bk, T)
        cases.append({"h_decide": decide.tolist(),
                      "h_opts": [o.tolist() for o in opts],
                      "ref_probs": ref.astype(np.float64).tolist()})
    with open(g("cases.json"), "w") as f:
        json.dump({"dim": DIM, "dp": DP, "cases": cases}, f)
    code, so, se = run_cpp(["--mode", "headbench", "--head", head_bin, "--cases", g("cases.json")])
    print(open(se, encoding="utf-8", errors="replace").read()[-4000:])
    print(open(so, encoding="utf-8", errors="replace").read()[-2000:])
    print(f"  [cpp exit={code}]")
    # 独立复核: 重新解析 cases.json 用 numpy 再算一次并 C++ 输出比较 (C++ 探针已打印), 这里自查一致性
    hb = json.load(open(g("cases.json")))
    ok = True
    for c_ in hb["cases"]:
        ref = np.array(c_["ref_probs"], dtype=np.float64)
        got = head_ref(np.array(c_["h_opts"], dtype=np.float32),
                       np.array(c_["h_decide"], dtype=np.float32), Wq, bq, Wk, bk, T)
        if np.argmax(ref) != np.argmax(got): ok = False
        if np.max(np.abs(ref - got)) > 1e-9: ok = False
    print(f"  [python ref self-check ok={ok}]")

    # ---------- 2) readout: 端到端 (llama.cpp hidden -> head) ----------
    print("== 2) readout (end-to-end) ==")
    req, tok = build_request(a)
    with open(g("req.json"), "w") as f:
        json.dump(req, f)
    code, so, se = run_cpp(["--mode", "readout", "--model", a.model, "--req", g("req.json"),
                            "--head", head_bin, "--probs-out", g("probs.json"),
                            "--hidden-out", g("hidden.json")])
    print(open(se, encoding="utf-8", errors="replace").read()[-3000:])
    print(open(so, encoding="utf-8", errors="replace").read()[-2000:])
    if code == 0 and os.path.exists(g("hidden.json")) and os.path.exists(g("probs.json")):
        hid = json.load(open(g("hidden.json")))
        cpp = json.load(open(g("probs.json")))
        max_dp, argmax_ok = 0.0, True
        for q, (d, opts) in enumerate(zip(hid["decide"], hid["opts"])):
            ref = head_ref(np.array(opts, dtype=np.float32), np.array(d, dtype=np.float32),
                           Wq, bq, Wk, bk, T)
            got = np.array(cpp["probs"][q], dtype=np.float64)
            max_dp = max(max_dp, float(np.max(np.abs(ref - got))))
            if argmax(got) != argmax(ref): argmax_ok = False
        print(f"  readout parity: max|dp|={max_dp:.3e}, argmax_all_match={argmax_ok}  "
              f"(C++ probs vs Python-head-on-C++-hidden)")
    print("done")

_SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]

def build_request(a, state="The key insight is causality.", instr="Pick the best follow-up.",
                  options=["It happened first.", "It explains the mechanism.", "Then we wait."]):
    """Build a small packed record's ids + decide/opt indices, mirroring kev.encode() layout,
    using the HF fast tokenizer (id source of truth). Returns (req_dict, tokenizer)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transformers import AutoTokenizer
    base_dir = os.path.join(os.path.dirname(os.path.abspath(a.model)), "qwen3.5-0.8b")
    tok = AutoTokenizer.from_pretrained(base_dir)
    usr = lambda t: tok(t, add_special_tokens=False).input_ids
    S = [tok.convert_tokens_to_ids(_SPECIAL[0])] + usr(state)
    qid, oid, cid, did = [tok.convert_tokens_to_ids(t) for t in _SPECIAL[1:]]
    ids = list(S)
    decide_idx, opt_idx = [], []
    questions = [(instr, options),
                 ("New best option for the same state.", ["Option alpha.", "Option beta.",
                                                          "Option gamma.", "Option delta."])]
    for k, (inst, opts) in enumerate(questions, start=1):
        instr_ids = [qid] + usr(inst)
        spans = [[oid] + usr(o) + [cid] for o in opts]
        br = instr_ids + [t for sp in spans for t in sp] + [did]
        base = len(ids)
        ends, cursor = [], len(instr_ids)
        for sp in spans:
            cursor += len(sp); ends.append(cursor - 1)
        ids += br
        decide_idx.append(base + len(br) - 1)
        opt_idx.append([base + e for e in ends])
    print(f"  built request: total_tokens={len(ids)}, n_questions={len(questions)}, "
          f"decide_idx={decide_idx}, opt_idx={opt_idx}")
    return {"ids": ids, "decide_idx": decide_idx, "opt_idx": opt_idx}, tok

if __name__ == "__main__":
    main()