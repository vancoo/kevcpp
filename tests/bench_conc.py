#!/usr/bin/env python3
"""Single-server concurrency benchmark (B2 validation).

Runs ONE server at a time (caller starts batched or serial server on a fresh port).
Warms up the model + prefix cache, then fires C identical (/v1/systemone) requests per
round (prefix LCP-hit + tails batched — the production regime) or C DISTINCT requests
(--mix) for R rounds, reporting burst wall and per-request p50/p95 latency.

Usage: python bench_conc.py <port> --conc C --rounds R [--same|--mix] [--warm R]
"""
import argparse, concurrent.futures, json, statistics, time
import httpx

BASE_SAME = [
    {"state": "My running shoes arrived in the wrong size; can I exchange them for a size 10?",
     "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?",
                                  "criteria": {"billing": "charges", "shipping": "delivery status",
                                               "returns": "wrong/damaged items", "account": "login", "other": "none"}}}}]
BASE_MIX = [
    {"state": "My running shoes arrived in the wrong size; can I exchange them for a size 10?",
     "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?",
                                  "criteria": {"billing": "charges", "shipping": "delivery status", "returns": "items", "other": "none"}}}},
    {"state": "I was charged twice for a subscription I cancelled.",
     "questions": {"billing": {"type": "choice", "instructions": "Is this about billing?",
                               "criteria": {"yes": "charges", "no": "other"}}}},
    {"state": "The package was marked delivered but never arrived.",
     "questions": {"topic": {"type": "choice", "instructions": "Subject?",
                             "criteria": {"delivery": "status", "returns": "refund", "billing": "charges"}}}},
    {"state": "客户反馈缺一件商品,需要补发或退款。",
     "questions": {"department": {"type": "choice", "instructions": "部门?",
                                  "criteria": {"物流": "shipping", "售后": "returns", "账务": "billing"}}}},
    {"state": "How long does a refund take and who pays return shipping?",
     "questions": {"urgency": {"type": "score", "instructions": "urgency?", "criteria": ["can wait", "this week", "today"]}}},
    {"state": "The laptop lid is cracked and the customer wants a replacement.",
     "questions": {"intent": {"type": "choice", "instructions": "intent?",
                              "criteria": {"replace": "replace", "repair": "repair", "refund": "refund"}}}},
    {"state": "取消一个尚未发货的订单并全额退款。",
     "questions": {"intent": {"type": "choice", "instructions": "intent?",
                              "criteria": {"cancel": "cancel", "status": "status", "other": "other"}}}},
    {"state": "A third charge has no matching order record.",
     "questions": {"dept": {"type": "choice", "instructions": "dept?",
                            "criteria": {"billing": "billing", "tech": "tech", "returns": "returns"}}}},
]

def fire(base, bodies, workers):
    lat = []
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        def one(b):
            lt0 = time.perf_counter()
            r = httpx.post(f"{base}/v1/systemone", json=b, timeout=60.0)
            lat.append((time.perf_counter() - lt0) * 1000.0)
            return r.status_code
        statuses = list(ex.map(one, bodies))
    wall = (time.perf_counter() - t0) * 1000.0
    return wall, lat, statuses

def quant(a, p):
    if not a: return 0.0
    a = sorted(a); return a[min(len(a)-1, int((len(a)-1)*p))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--mix", action="store_true")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    pool = BASE_MIX if a.mix else BASE_SAME
    bodies = [pool[i % len(pool)] for i in range(a.conc)]
    # warm up: build prefix cache + graphs
    for _ in range(2):
        for b in bodies:
            httpx.post(f"{base}/v1/systemone", json=b, timeout=60.0)
    walls = []; all_lat = []
    for rnd in range(a.rounds):
        w, lat, st = fire(base, bodies, a.conc)
        walls.append(w); all_lat += lat
        ok = all(s == 200 for s in st)
        print(f"[round {rnd+1}] wall={w:.0f}ms p50={quant(lat,0.50):.0f} p95={quant(lat,0.95):.0f} ok={ok}")
    walls = walls[2:] if len(walls) > 2 else walls   # drop first 2 warmup-rounds noise
    print("=" * 55)
    print(f"conc={a.conc} mode={'mix' if a.mix else 'same'} rounds={a.rounds}")
    print(f"burst_wall: avg={statistics.mean(walls):.0f}ms min={min(walls):.0f} max={max(walls):.0f}")
    print(f"per_req:   p50={quant(all_lat,0.50):.0f} p95={quant(all_lat,0.95):.0f} avg={statistics.mean(all_lat):.0f}")

if __name__ == "__main__":
    main()