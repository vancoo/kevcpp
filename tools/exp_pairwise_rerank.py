"""Experiment: pointwise binary scoring + pairwise rerank of the top-2 candidates.

Not part of the production runner. Reuses the same /v1/systemone endpoint:
a bare noul question renders one direct yes/no prompt, so asking
'Is "A" a better answer than "B"?' yields P(A better than B) from the yes logit.

Usage:
  python tools/exp_pairwise_rerank.py DATASET.jsonl [...]
Env: KEV_ENDPOINT, KEV_RESULTS, KEV_MARGIN (default 0.2)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_jevbench_kevcpp as base
from jevbench.scoring import score_task, validate_probs, InvalidDistribution


MARGIN = float(os.environ.get("KEV_MARGIN", "0.2"))


def pairwise(state, first, second):
    """Return P(first is a better answer than second)."""
    instructions = (
        f'Compare the two candidate answers below. '
        f'Is "{first}" a better answer than "{second}"?'
    )
    raw, latency, error, status = base.post({
        "state": state,
        "questions": {"decision": {"type": "noul", "instructions": instructions}},
    })
    if error or status != 200:
        return None, latency
    body = base.parse_json(raw)
    return float(body["answers"]["decision"]["noul"]), latency


def rerank(task, probs_pointwise):
    """Return (probs, n_pairwise, latency)."""
    labels = sorted(probs_pointwise, key=probs_pointwise.get, reverse=True)
    if len(labels) < 2:
        return probs_pointwise, 0, 0.0
    first, second = labels[0], labels[1]
    margin = probs_pointwise[first] - probs_pointwise[second]
    if margin >= MARGIN:
        return probs_pointwise, 0, 0.0
    criteria = task.question["criteria"]
    text = {k: (k if v is None else f"{k}: {json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v}")
            for k, v in criteria.items()}
    ab, t1 = pairwise(task.state, text[first], text[second])
    ba, t2 = pairwise(task.state, text[second], text[first])
    if ab is None or ba is None:
        return probs_pointwise, 0, t1 + t2
    p_first = (ab + (1.0 - ba)) / 2.0
    out = dict(probs_pointwise)
    if p_first >= 0.5:
        out[first], out[second] = max(out[first], out[second]), min(out[first], out[second])
    else:
        out[first], out[second] = min(out[first], out[second]), max(out[first], out[second])
    total = sum(out.values())
    out = {k: v / total for k, v in out.items()}
    return out, 2, t1 + t2


def main():
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit("usage: exp_pairwise_rerank.py DATASET.jsonl [...]")
    tasks, digest = base.load_tasks(paths)
    results_path = os.environ.get("KEV_RESULTS", "build/exp-pairwise.json")
    rows = []
    base.save_results(results_path, base.summarize([], len(tasks), digest))
    for index, task in enumerate(tasks, 1):
        raw, latency, error, status = base.post(base.build_body(task))
        row = {"id": task.id, "family": task.family, "type": task.question["type"],
               "expected": task.expected,
               "scorable": task.expected is not None and not task.provenance.get("exclude_reason"),
               "latency_s": latency, "pairwise": 0, "reranked": False,
               "ok": False, "valid": False,
               "correct": False, "strict_valid": False, "strict_correct": False,
               "predicted": None, "error": error, "probs": None}
        try:
            if error or status != 200:
                raise ValueError(error or f"HTTP {status}")
            answers = base.parse_json(raw)["answers"]["decision"]
            qtype = task.question["type"]
            if qtype == "noul":
                probs = {"yes": answers["noul"], "no": 1.0 - answers["noul"]}
            else:
                probs = answers["probabilities"]
            if qtype == "choice":
                probs, npair, extra = rerank(task, probs)
                row["pairwise"] = npair
                row["latency_s"] += extra
                row["reranked"] = npair > 0
            scored = score_task(probs, task)
            row.update({"ok": True, "valid": scored["valid"], "strict_valid": scored["strict_valid"],
                        "correct": scored["correct"], "predicted": scored["predicted"],
                        "probs": scored.get("probs")})
            row["strict_correct"] = bool(scored["strict_valid"] and scored["correct"])
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        summary = base.summarize(rows, len(tasks), digest)
        base.save_results(results_path, summary)
        if index % 10 == 0 or index == len(tasks):
            print(f"{index}/{len(tasks)} correct={summary['strict_correct']} "
                  f"valid={summary['strict_valid']} accuracy={summary['accuracy']}", flush=True)
    print(f"wrote {results_path}", flush=True)


if __name__ == "__main__":
    main()
