#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""比较 separate 与 merged 两份 lora_ab 输出 (PR 行数对齐), 计算 argmax 一致率 / max|dp| / avg|dp|。
用法: python compare_lora_ab.py out_separate.txt out_merged.txt [aligned_only]"""
import sys, math

def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line.startswith("PR "): continue
        parts = line.split()
        # PR <reqIdx> <qIdx> <argmax> p1 p2 ...
        req = int(parts[1]); qi = int(parts[2]); am = int(parts[3])
        probs = [float(x) for x in parts[4:]]
        rows.append((req, qi, am, probs))
    return rows

def main():
    a = load(sys.argv[1])
    b = load(sys.argv[2])
    """至少一个可选参数: 排除某 req 索引(comma-separated)。
    用法: compare_lora_ab.py outA outB [exclude reqidx,...]"""
    excl = set()
    if len(sys.argv) > 3:
        for t in sys.argv[3].split(","):
            if t.strip().isdigit(): excl.add(int(t.strip()))
    n = min(len(a), len(b))
    if len(a) != len(b):
        print("WARN: line counts differ separate=%d merged=%d; compare first %d aligned" % (len(a), len(b), n))
    same = 0; total = 0; excl_n = 0
    maxdp = 0.0; sumdp = 0.0; maxdp_q = None
    flips = []
    near_ties = []   # separate 侧 top-2 margin < 0.05, 用于解释翻转发生在"本就不确定"处
    for i in range(n):
        ra, rb = a[i], b[i]
        if ra[0] != rb[0] or ra[1] != rb[1]:
            print("WARN: order mismatch at %d: %s vs %s" % (i, ra[:2], rb[:2])); continue
        if ra[0] in excl: excl_n += 1; continue
        pa, pb = ra[3], rb[3]
        for ai, (x, y) in enumerate(zip(pa, pb)):
            d = abs(x - y)
            if d > maxdp: maxdp = d; maxdp_q = (ra[0], ra[1], ai)
            sumdp += d
        # separate 侧 top-2 margin
        srt = sorted(pa, reverse=True)
        margin = srt[0] - srt[1]
        if margin < 0.05: near_ties.append((ra[0], ra[1], margin))
        if ra[2] == rb[2]:
            same += 1
        else:
            flips.append((ra[0], ra[1], ra[2], rb[2], ra[3], rb[3], margin))
        total += 1
    opts = len(a[0][3])
    avg_opt = sumdp / (total * opts)
    avg_q = sumdp / total
    print("="*70)
    print("questions=%d (excluded=%d)  options_per_q=%d" % (total, excl_n, opts))
    print("argmax 一致率 = %d/%d = %.2f%%" % (same, total, 100.0*same/total))
    print("max |dp| = %.6f (at req/q/opt %s)" % (maxdp, maxdp_q))
    print("avg |dp| (per option element) = %.6f" % avg_opt)
    print("avg |dp| (per question sum)   = %.6f" % avg_q)
    near = sorted(near_ties)
    n_near = len(near)
    flip_near = sum(1 for f in flips if f[6] < 0.05)
    print("separate 侧 near-tie(margin<0.05) 问数 = %d/%d" % (n_near, total))
    print("-"*70)
    if flips:
        print("argmax 翻转样本 (%d): 其中 near-tie=%d" % (len(flips), flip_near))
        for (r, q, sa, sb, pa, pb, margin) in flips:
            print("  req_%02d q%d  separate->%d merged->%d  (sep-tie-margin=%.4f)" % (r, q, sa, sb, margin))
            print("    sep = " + "%.4f "*len(pa) % tuple(pa))
            print("    mrg = " + "%.4f "*len(pb) % tuple(pb))
    else:
        print("argmax 翻转样本: 0")
    return 0

if __name__ == "__main__":
    sys.exit(main())