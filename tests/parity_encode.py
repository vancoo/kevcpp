#!/usr/bin/env python
"""M2 对拍: C++ kev_encode <-> Python 参考(与 kev/model.py encode 逻辑一致, 用 HF fast tokenizer)。

只 import transformers 的 tokenizer(不经 kev.model, 规避 proxy 环境 transformers 5 缺
LinearAttentionCacheLayerMixin 无法 import 的问题)。参考实现逐行复刻 model.py 的
user_tokens/encode/rows_of, 与 C++ headbench --mode encode 的输出逐 token/逐索引比较。
走文件、不用 subprocess 管道捕获(规避沙箱 EPERM)。
"""
import json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "build", "bin", "Release", "headbench.exe")
GGUF = r"D:\share-models\Qwen3.5-0.8B-Q8_0.gguf"
BASE = r"D:\share-models\qwen3.5-0.8b"
GOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")

SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
OPT_NONE, OPT_DECIDE = -1, -2
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
MAX_STATE, MAX_BRANCH = 384, 1024

def user_tokens(tok, text):
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids

def encode(tok, rec):
    state_tokens = user_tokens(tok, rec["state"])
    S = [tok.convert_tokens_to_ids(SPECIAL[0])] + state_tokens[: MAX_STATE - 1]
    ids, seg, pos, opt = list(S), [0] * len(S), list(range(len(S))), [OPT_NONE] * len(S)
    q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(t) for t in SPECIAL[1:])
    decide_idx, opt_idx = [], []
    for k, q in enumerate(rec["questions"], start=1):
        instr = [q_id] + user_tokens(tok, q["instr"])
        spans = [[o_id] + user_tokens(tok, o) + [c_id] for o in q["options"]]
        br = instr + [t for sp in spans for t in sp] + [d_id]
        base = len(ids); p0 = len(S)
        br_opt = [OPT_NONE] * len(instr) + [j for j, sp in enumerate(spans) for _ in sp] + [OPT_DECIDE]
        br_pos = list(range(p0, p0 + len(br)))
        ends, cursor = [], len(instr)
        for sp in spans:
            cursor += len(sp); ends.append(cursor - 1)
        ids += br; seg += [k] * len(br); pos += br_pos; opt += br_opt
        decide_idx.append(base + len(br) - 1); opt_idx.append([base + e for e in ends])
    return {"ids": ids, "seg": seg, "pos": pos, "opt": opt,
            "decide_idx": decide_idx, "opt_idx": opt_idx}

def rows_of(enc):
    seg = enc["seg"]; Ls = seg.count(0)
    rows, start = [], Ls
    for k, (d, oi) in enumerate(zip(enc["decide_idx"], enc["opt_idx"]), start=1):
        end = d + 1
        rows.append({"ids": enc["ids"][start:end], "pos": enc["pos"][start:end],
                     "decide": d - start, "opts": [o - start for o in oi]})
        start = end
    return enc["ids"][:Ls], enc["pos"][:Ls], rows

def call_cpp(req):
    rf = os.path.join(GOLD, "_req.json")
    with open(rf, "w") as f: json.dump(req, f)
    sy = os.path.join(GOLD, "_enc_out.json"); se = os.path.join(GOLD, "_enc_err.log")
    with open(sy, "wb") as so, open(se, "wb") as ee:
        subprocess.run([EXE, "--mode", "encode", "--model", GGUF, "--req", rf], stdout=so, stderr=ee)
    txt = open(sy, encoding="utf-8", errors="replace").read().strip()
    if not txt.startswith("{"):
        return None, open(se, encoding="utf-8", errors="replace").read()[-300:]
    return json.loads(txt), None

def cmp(name, a, b):
    return a == b, f"{name}:\n  ref={a}\n  cpp={b}"

def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)

    requests = [
        {"state": "The key insight is about causality and <|eff|> contagion.",
         "questions": [
             {"instr": "Pick the best follow-up.", "options": ["It happened first.",
                                                               "It explains the mechanism.<|x|>",
                                                               "Then we wait."]},
             {"instr": "What is the main reason?", "options": ["A: numbers 1, 2, 3.",
                                                               "B", "C: second cause", "D: decided"]},
         ]},
        {"state": "dtype=f32, shape=[2,4], max=2.5e-3",
         "questions": [{"instr": "Check", "options": ["<|a|>yes", "no", "maybe|not|"]}]},
    ]

    allok = True
    for ri, req in enumerate(requests):
        ref = encode(tok, req)
        Ls_ref, _, rows_ref = rows_of(ref)
        cpp, err = call_cpp(req)
        print(f"== request {ri} ==")
        if cpp is None:
            print("  C++ err:", (err or "")[:300]); allok = False; continue
        checks = []
        checks.append(cmp("ids", ref["ids"], cpp["ids"]))
        checks.append(cmp("seg", ref["seg"], cpp["seg"]))
        checks.append(cmp("pos", ref["pos"], cpp["pos"]))
        checks.append(cmp("opt", ref["opt"], cpp["opt"]))
        checks.append(cmp("decide_idx", ref["decide_idx"], cpp["decide_idx"]))
        checks.append(cmp("opt_idx", ref["opt_idx"], cpp["opt_idx"]))
        checks.append(cmp("state_len(Ls)", len(Ls_ref), int(cpp["ls"])))
        # rows
        cpp_rows = [(int(r["decide"]), r["opts"]) for r in cpp["rows"]]
        for k, (rref, rcpp) in enumerate(zip(rows_ref, cpp_rows)):
            checks.append(cmp(f"row{k}.decide/opts", (rref["decide"], rref["opts"]), rcpp))
        this = all(ok for ok, _ in checks)
        allok &= this
        for ok, msg in checks:
            if not ok: print("  [DIFF] " + msg)
        print("  RESULT:", "PASS" if this else "FAIL")

    print("ALL:", "PASS" if allok else "FAIL")
    return 0 if allok else 1

if __name__ == "__main__":
    sys.exit(main())