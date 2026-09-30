#!/usr/bin/env python
"""Tokenizer parity probe: llama_tokenize (GGUF) vs HF fast tokenizer on kev user_tokens strings.

For each sample, applies the kev <|x|>-><¦x¦> rewrite (the text actually passed to the tokenizer),
tokenizes with HF fast (add_special_tokens=False), and with llama.cpp's llama_tokenize via the
C++ headbench.exe --mode tokens. Compares id sequences.

Runs C++ via files (no subprocess pipe capture -> sandbox EPERM).
"""
import json, os, subprocess, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "build", "bin", "Release", "headbench.exe")
GGUF = r"D:\share-models\Qwen3.5-0.8B-Q8_0.gguf"
BASE = r"D:\share-models\qwen3.5-0.8b"
TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "_probe")

_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")

def llm_ids(text_byte):
    # write text (as-is, utf-8 bytes) to a file, run C++ tokens, parse stdout [..]
    os.makedirs(TMP, exist_ok=True)
    tf = os.path.join(TMP, "s.txt")
    with open(tf, "wb") as f:
        f.write(text_byte)
    sy = os.path.join(TMP, "so.log"); se = os.path.join(TMP, "se.log")
    with open(sy, "wb") as so, open(se, "wb") as ee:
        subprocess.run([EXE, "--mode", "tokens", "--model", GGUF, "--text-file", tf, "--add-special", "false"],
                       stdout=so, stderr=ee)
    line = open(sy, encoding="utf-8", errors="replace").read().strip().splitlines()
    line = [l for l in line if l.startswith("[")]
    if not line:
        return None, open(se, encoding="utf-8", errors="replace").read()[-400:]
    return json.loads(line[0]), None

def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    samples = [
        "The key insight is causality.",
        "Pick the best follow-up.",
        "It happened first.",
        "Then we <|must|> wait for <|box_start|> conditions.",
        "dtype=f32, shape=[2,4], max=2.5e-3",
        "Réponse en français — déjà vu.",
        "state:<|x|> and <|y|> rewritten<|z|>",
        "12345, a b c, file_1_name",
    ]
    allok = True
    for s in samples:
        rewritten = _SPECIAL_RE.sub(r"<¦\1¦>", s)          # what kev's user_tokens passes to the tokenizer
        hf = tok(rewritten, add_special_tokens=False).input_ids
        lm, err = llm_ids(rewritten.encode("utf-8"))
        if lm is None:
            print(f"  [{s!r}] C++ err: {err[:200]}"); allok = False; continue
        same = (hf == lm)
        allok &= same
        print(f"[{'MATCH' if same else 'DIFF '}] {len(hf)} tokens")
        if not same:
            print(f"   hf : {hf}")
            print(f"   lm : {lm}")
            print(f"   hl: {[tok.decode([t]) for t in hf]}")
            print(f"   lm: {[tok.decode([t]) for t in lm]}")
    print("RESULT:", "PASS" if allok else "FAIL")
    return 0 if allok else 1

if __name__ == "__main__":
    sys.exit(main())