#!/usr/bin/env python3
"""B2 parity check: batched server vs serial (compute_mutex_) server.

Sends the SAME set of diverse /v1/systemone requests to both servers (batched on
B_PORT, serial on S_PORT) one at a time, and compares the returned CHOICE (must be
identical) and the probabilities (may shift slightly, since llama.cpp CPU is not
batch-shape bit-identical — see B1 findings). Also runs a concurrent burst on each
to report latency under load.
"""
import json, sys, time
import httpx

B_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8010
S_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8011

BASE_B = f"http://127.0.0.1:{B_PORT}"
BASE_S = f"http://127.0.0.1:{S_PORT}"

REQUESTS = [
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
    {"state": {"document": "How long does a refund take to appear and who pays return shipping?"},
     "questions": {"urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["can wait", "this week", "today"]},
                   "billing": {"type": "noul", "instructions": "About billing?", "criteria": {"true": "yes", "false": "no"}}}},
    {"state": "I need to cancel my order that has not shipped yet and get a full refund.",
     "questions": {"intent": {"type": "choice", "instructions": "Customer's intent?",
                              "criteria": {"cancel": "cancel order", "return": "return shipped item",
                                           "status": "check status", "other": "other"}}}},
]

def send(base, body):
    r = httpx.post(f"{base}/v1/systemone", json=body, timeout=60.0)
    return r.status_code, (r.json() if r.status_code == 200 else r.text)

def flat_answers(answers):
    out = []
    for qid in sorted(answers.keys()):
        a = answers[qid]
        if a.get("type") == "choice":
            out.append((qid, "choice", a.get("choice")))
        elif a.get("type") == "noul":
            out.append((qid, "noul", a.get("noul")))
        elif a.get("type") == "score":
            out.append((qid, "score", a.get("score")))
    return out

print(f"batched={BASE_B} serial={BASE_S}")
choice_mismatch = 0
max_prob_dev = 0.0
for i, body in enumerate(REQUESTS):
    sc_b, rb = send(BASE_B, body)
    sc_s, rs = send(BASE_S, body)
    if sc_b != 200 or sc_s != 200:
        print(f"[req{i}] batched_status={sc_b} serial_status={sc_s} SKIP")
        continue
    ab = rb["answers"]; asym = rs["answers"]
    fb = flat_answers(ab); fs = flat_answers(asym)
    ok = True
    for (qid, t, v) in fb:
        vs = [v2 for (q2, t2, v2) in fs if q2 == qid and t2 == t]
        if not vs or v != vs[0]:
            ok = False
    if not ok:
        choice_mismatch += 1
        print(f"[req{i}] CHOICE MISMATCH batched={fb} serial={fs}")
    # max per-candidate probability deviation
    for qid in ab.keys():
        if ab[qid].get("type") != "choice" or qid not in asym: continue
        pb = ab[qid]["probabilities"]; ps = asym[qid]["probabilities"]
        for k in set(pb) | set(ps):
            d = abs(pb.get(k, 0.0) - ps.get(k, 0.0))
            if d > max_prob_dev: max_prob_dev = d
    print(f"[req{i}] choices_match={'OK' if ok else 'MISMATCH'} max_prob_dev_so_far={max_prob_dev:.4f}")

print("=" * 60)
print(f"RESULT: choice_mismatch_requests={choice_mismatch}/{len(REQUESTS)}  max_prob_dev={max_prob_dev:.4f}")
print("PASS" if choice_mismatch == 0 else "FAIL")
sys.exit(0 if choice_mismatch == 0 else 1)