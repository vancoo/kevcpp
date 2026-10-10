#!/usr/bin/env python
"""Send the A/B request fixtures to a KEV systemone server and print per-question
choice / full probabilities / confidence, plus the raw JSON answers.
Usage: python ab_drive.py <base_url> <req_file...>"""
import sys, json
import httpx, math

def post(cl, body):
    r = cl.post("/v1/systemone", json=body)
    return r.status_code, r.json()

def main():
    base, files = sys.argv[1], sys.argv[2:]
    cl = httpx.Client(base_url=base, timeout=180)
    for f in files:
        body = json.load(open(f, encoding="utf-8"))
        code, r = post(cl, body)
        print(f"===== file={f} base={base} status={code} =====")
        if code != 200:
            print(json.dumps(r, ensure_ascii=False)[:2000]); continue
        for qid, a in r.get("answers", {}).items():
            if a.get("type") == "choice":
                probs = a.get("probabilities", {})
                choice = a.get("choice")
                conf = a.get("confidence")
                items = "  ".join(f"{k}={v:.4f}" for k, v in probs.items())
                print(f"[{qid}] choice={choice!r} confidence={conf:.4f}")
                print(f"    probs: {items}")
            else:
                print(f"[{qid}] type={a.get('type')} -> {json.dumps(a, ensure_ascii=False)[:500]}")
        print(json.dumps(r.get("usage", {}), ensure_ascii=False))

if __name__ == "__main__":
    main()