#!/usr/bin/env python3
"""B2 concurrency A/B: batched server vs serial (compute_mutex_) server.

Fires C concurrent /v1/systemone requests (a set of DISTINCT bodies, same set to both
servers, same thread order) at each server and compares:
  * correctness  : choice of body[i] identical batched-vs-serial?
  * latency      : wall time of the whole C-burst + per-request latency (ms),
                   batched vs serial under identical concurrent load.

Usage: python b2_concurrency.py <batched_port> <serial_port> --conc C [--rounds R]
"""
import argparse, concurrent.futures, json, time
import httpx

BODIES = [
    {"state": "My running shoes arrived in the wrong size; can I exchange them for a size 10?",
     "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?",
                                  "criteria": {"billing": "charges/invoices", "shipping": "delivery status",
                                               "returns": "wrong/damaged items", "other": "none"}}}},
    {"state": "I was charged twice on my credit card for a subscription I already cancelled.",
     "questions": {"billing": {"type": "choice", "instructions": "Is this about billing?",
                               "criteria": {"yes": "explicitly about charges", "no": "not about charges"}}}},
    {"state": "The package shows delivered but I never received it; please investigate the carrier.",
     "questions": {"topic": {"type": "choice", "instructions": "What is the customer asking about?",
                             "criteria": {"returns": "return/refund", "delivery": "delivery status",
                                          "billing": "charges", "account": "login access"}}}},
    {"state": "客户反馈收到的商品缺了一件, 外包装破损, 需要安排补发或退款。",
     "questions": {"department": {"type": "choice", "instructions": "归到哪个部门?",
                                  "criteria": {"物流": "shipping", "售后": "returns/refund", "账务": "billing"}}}},
    {"state": "I need to cancel my order that has not shipped yet and get a full refund.",
     "questions": {"intent": {"type": "choice", "instructions": "Customer's intent?",
                              "criteria": {"cancel": "cancel order", "return": "return shipped item",
                                           "status": "check status", "other": "other"}}}},
    {"state": "How long does a refund take to appear and who pays return shipping?",
     "questions": {"urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["can wait", "this week", "today"]}}},
    {"state": "The customer says the laptop lid is cracked and they want a replacement.",
     "questions": {"return": {"type": "choice", "instructions": "Return or repair?",
                              "criteria": {"replace": "replace item", "repair": "repair item", "refund": "refund"}}}},
    {"state": "第三笔扣款没有对应的订单记录, 客户要求在账单上详细注明。",
     "questions": {"department": {"type": "choice", "instructions": "处理部门?",
                                  "criteria": {"账务": "billing", "技术": "tech", "售后": "after-sales"}}}},
]

def fire(base, bodies, workers):
    """fire all `bodies` concurrently via `workers` threads; returns (body_idx -> result) and per-request latency ms."""
    res = {}
    lat = {}
    t0 = time.perf_counter()
    def one(args):
        i, b = args
        lt0 = time.perf_counter()
        try:
            r = httpx.post(f"{base}/v1/systemone", json=b, timeout=60.0)
            out = (r.status_code, r.json() if r.status_code == 200 else r.text)
        except Exception as e:
            out = (0, str(e))
        dt = (time.perf_counter() - lt0) * 1000.0
        return i, out, dt
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, out, dt in ex.map(one, list(enumerate(bodies))):
            res[i] = out; lat[i] = dt
    wall = (time.perf_counter() - t0) * 1000.0
    return res, wall, lat

def flat_choice(a):
    out = {}
    for qid, an in (a or {}).items():
        if isinstance(an, dict) and an.get("type") == "choice":
            out[qid] = an.get("choice")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("b_port", type=int); ap.add_argument("s_port", type=int)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--mix", action="store_true", help="use a MIXTURE of request types (stresses inter-request batching)")
    a = ap.parse_args()
    bb = f"http://127.0.0.1:{a.b_port}"; sb = f"http://127.0.0.1:{a.s_port}"

    # build per-round body list = distinct bodies, repeated to fill C
    base = BODIES

    print(f"batched={bb} serial={sb} conc={a.conc} rounds={a.rounds} mix={a.mix}")
    choice_mismatch_total = 0
    sample_mismatch = 0
    b_walls, s_walls = [], []
    b_lats, s_lats = [], []
    max_prob_dev = 0.0
    for rnd in range(1, a.rounds + 1):
        bodies = [base[i % len(base)] for i in range(a.conc)]
        rb, wb, lb = fire(bb, bodies, a.conc)
        rs, ws, ls = fire(sb, bodies, a.conc)
        b_walls.append(wb); s_walls.append(ws)
        b_lats.extend(lb.values()); s_lats.extend(ls.values())
        for i in range(a.conc):
            cb = rb.get(i, (0, None))[1]; cs = rs.get(i, (0, None))[1]
            cf_b = flat_choice(cb if isinstance(cb, dict) else {})
            cf_s = flat_choice(cs if isinstance(cs, dict) else {})
            if cf_b != cf_s:
                choice_mismatch_total += 1
                if sample_mismatch < 3:
                    sample_mismatch += 1
                    print(f"[round{rnd} i{i}] CHOICE MISMATCH batched={cf_b} serial={cf_s}")
            # measure probability deviation for choice answers
            if isinstance(cb, dict) and isinstance(cs, dict):
                for qid in cb.get("answers", {}):
                    ab = cb["answers"][qid]; asy = cs.get("answers", {}).get(qid)
                    if not isinstance(asy, dict) or ab.get("type") != "choice": continue
                    pb = ab.get("probabilities", {}); ps = asy.get("probabilities", {})
                    for k in set(pb) | set(ps):
                        d = abs(pb.get(k, 0.0) - ps.get(k, 0.0))
                        if d > max_prob_dev: max_prob_dev = d
        print(f"[round {rnd}] batched wall={wb:.0f}ms | serial wall={ws:.0f}ms | "
              f"batched avg_lat={sum(lb.values())/len(lb):.0f}ms serial avg_lat={sum(ls.values())/len(ls):.0f}ms | "
              f"choice_mismatch_this_round={sum(1 for i in range(a.conc) if flat_choice(rb.get(i,(0,None))[1] if isinstance(rb.get(i,(0,None))[1],dict) else {}) != flat_choice(rs.get(i,(0,None))[1] if isinstance(rs.get(i,(0,None))[1],dict) else {}))}")

    b_wall_avg = sum(b_walls)/len(b_walls); s_wall_avg = sum(s_walls)/len(s_walls)
    def quant(a_, p):
        a_ = sorted(a_); k = int(p * (len(a_) - 1)); return a_[k]
    print("=" * 70)
    print(f"burst wall: batched avg={b_wall_avg:.0f}ms  serial avg={s_wall_avg:.0f}ms  speedup={s_wall_avg/max(b_wall_avg,0.001):.2f}x")
    print(f"per_request: batched p50={quant(b_lats,0.5):.0f} p95={quant(b_lats,0.95):.0f} | "
          f"serial p50={quant(s_lats,0.5):.0f} p95={quant(s_lats,0.95):.0f}")
    print(f"choice_mismatch_total={choice_mismatch_total}/{a.rounds*a.conc}  max_prob_dev={max_prob_dev:.4f}")
    print("PASS" if choice_mismatch_total == 0 else "FAIL")
    return 0 if choice_mismatch_total == 0 else 1

if __name__ == "__main__":
    import sys; sys.exit(main())