# -*- coding: utf-8 -*-
"""
kevcpp 判别正确性测试 harness v2 (临时, 放 build/)

数据三种原生类型, 均直接按 kevcpp 原生格式发送, 零转换:
  - score : criteria=list, expected=int(index)  -> 比对 argmax(probabilities) 索引
  - choice: criteria=object, expected=str(key)   -> 比对返回 choice 键
  - noul  : 无 criteria, expected=bool           -> 比对 round(noul)

用法:
  python build/correctness_harness.py --port 18080 --model 0.8B
"""
import json, os, sys, time, urllib.request, argparse, collections, re

def post(url, body, timeout=None):
    # 超时对长文档很关键: 超时中断会在服务端留下未释放的并发槽位, 后续请求全部 422
    # "no free concurrent sequence slot"。35B CPU 上 4k-token 文档可能需数分钟, 故默认给足。
    if timeout is None:
        timeout = int(os.environ.get("KEV_HARNESS_TIMEOUT", "3600"))
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    st = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data, (time.time() - st) * 1000.0

def judge(qtype, expected, an):
    """返回 (hit:bool, pred_repr, detail) 对齐 expected"""
    if qtype == "score":
        probs = an.get("probabilities", {})
        idxs = sorted(int(k) for k in probs)
        pred = max(idxs, key=lambda i: probs[str(i)]) if idxs else None
        return pred == expected, pred, {"probs": {str(k): round(probs.get(str(k),0),4) for k in idxs},
                                        "sv": an.get("score")}
    if qtype == "choice":
        ch = an.get("choice")
        return ch == expected, ch, {"probabilities": an.get("probabilities"),
                                    "conf": an.get("confidence")}
    if qtype == "noul":
        n = an.get("noul")
        pred_bool = n >= 0.5
        return pred_bool == expected, bool(pred_bool), {"noul": n}
    return False, None, {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cases", default="tests/bench-60cases.jsonl",
                    help="输入用例集(jsonl): 默认本地 85 题; 传 build/kev-hard-v1-test.jsonl 可跑 kev 官方集")
    ap.add_argument("--out", default=None, help="结果输出路径(默认 build/correctness_<model>.json)")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}/v1/systemone"
    cases = [json.loads(l) for l in open(args.cases, encoding="utf-8-sig") if l.strip()]
    OUT = args.out or f"build/correctness_{args.model}.json"

    results = []
    tot = hit = 0
    pertype = collections.Counter(); pertype_hit = collections.Counter()
    for c in cases:
        sid = c["id"]; cat = c["category"]; state = c["state"]
        for qk, q in c["questions"].items():
            qtype = q["type"]
            # 原生 body (三类型都无需转换; choice 的 criteria 本就是对象, noul 无 criteria, score 为数组)
            bq = {"type": qtype, "instructions": q.get("instructions", "")}
            if qtype in ("score", "choice"):
                bq["criteria"] = q["criteria"]
            body = {"state": state, "questions": {qk: bq}}
            expected = c["expected"][qk]
            row = {"id": sid, "category": cat, "qtype": qtype, "key": qk,
                   "state_summary": state[:40], "criteria": q.get("criteria"),
                   "expected": expected}
            try:
                d, lat = post(base, body)
                an = d["answers"][qk]
                h, pred, det = judge(qtype, expected, an)
                row.update({"hit": h, "pred": pred, "detail": det,
                            "latency_ms": round(lat, 1), "per_tot_note": ""})
                tot += 1; hit += h
                pertype[qtype] += 1; pertype_hit[qtype] += h
            except Exception as e:
                row.update({"hit": False, "pred": None, "err": "%s: %s" % (type(e).__name__, e),
                            "latency_ms": None})
                tot += 1; pertype[qtype] += 1
            results.append(row)

    acc = hit / tot if tot else 0
    summary = {"model": args.model, "port": args.port, "total_questions": tot,
               "hit": hit, "accuracy": round(acc, 4)}
    bycat = collections.defaultdict(lambda: [0, 0])
    for r in results: bycat[r["category"]][0] += r["hit"]; bycat[r["category"]][1] += 1
    summary["by_category"] = {k: {"hit": v[0], "tot": v[1], "acc": round(v[0]/v[1], 4)}
                              for k, v in sorted(bycat.items())}
    summary["by_qtype"] = {k: {"hit": pertype_hit[k], "tot": pertype[k],
                               "acc": round(pertype_hit[k]/pertype[k], 4)} for k in pertype}
    bylen = collections.defaultdict(lambda: [0, 0])
    for r in results:
        if r["qtype"] == "score":
            L = len(r["criteria"]); bylen[L][0] += r["hit"]; bylen[L][1] += 1
    summary["by_critlen(score)"] = {k: {"hit": v[0], "tot": v[1]} for k, v in bylen.items()}

    fn = OUT
    with open(fn, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)

    md = [f"# kevcpp 判别正确性明细 — {args.model} (backend=vulkan, port {args.port})",
          "",
          f"- **总体: {hit}/{tot} = {acc:.3f}**",
          f"- 按类型: " + "; ".join(f"{k} {pertype_hit[k]}/{pertype[k]}={pertype_hit[k]/pertype[k]:.3f}" for k in pertype),
          "",
          "| id | category | type | key | state | criteria/expected | pred | hit | lat_ms |",
          "|----|----------|------|-----|-------|-------------------|------|-----|--------|"]
    for r in results:
        crit = json.dumps(r["criteria"], ensure_ascii=False)[:40] if r["criteria"] is not None else "-"
        exp = str(r["expected"])
        md.append(f"| {r['id']} | {r['category']} | {r['qtype']} | {r['key']} | {r['state_summary']} | "
                  f"{crit} → {exp} | {r.get('pred','ERR')} | {'✔' if r['hit'] else '✘'} | {r.get('latency_ms','-')} |")
    mdfn = f"build/correctness_{args.model}.md"
    with open(mdfn, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"=== {args.model} 汇总 (port {args.port}) ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"明细: {mdfn} / {fn}")

if __name__ == "__main__":
    main()